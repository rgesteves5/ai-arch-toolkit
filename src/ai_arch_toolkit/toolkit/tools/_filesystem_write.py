"""Filesystem write tools bound to a ``FilesystemPolicy``, and ``filesystem_tools`` (C07, D64).

``filesystem_tools(policy)`` gives the three reads bound to the policy's read roots and four
writes bound to its write roots: ``write_file``, ``append_file``, ``make_directory`` and
``move_path``. None of them deletes (C07.7). Each checks its paths with the policy right before it
acts, reaches them from the root down, one folder at a time and never through a link, and acts
inside the folder it opened.

A write is atomic (C07.5): the text goes to a new temporary file in the target's folder and is
synced to disk, then takes the target's name in one step, and the folder is synced. Without
``overwrite`` that step is ``os.link``, which refuses a name that exists, so of two writes racing
to one new path one wins and the other fails; with it, ``os.replace``, and the file keeps the old
one's permissions. An append is not atomic. A move never copies, so it is refused across volumes,
and never brings a file the agent cannot read where it reads (``check_move``). The four run one at
a time in a process, so no two of them interleave. POSIX only (C07.8).
"""

from __future__ import annotations

import difflib
import errno
import os
import secrets
import stat
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from ai_arch_toolkit.core import ToolResult
from ai_arch_toolkit.core._tools._definition import ToolPreview
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._filesystem import bound_reads, governed, path_failure
from ai_arch_toolkit.toolkit.tools._filesystem_policy import (
    FilesystemPolicy,
    FilesystemPolicyError,
    check_move,
    checked,
    checks_paths,
    open_beneath,
    opened_folder,
)

_WRITE_REASON = "Writing a file creates or replaces data on this machine."
_APPEND_REASON = "Appending to a file changes data on this machine."
_MAKE_REASON = "Making a folder changes the files on this machine."
_MOVE_REASON = "Moving a file or folder changes where data is, and can replace data."
# What a write's temporary file is named after, so one left by a crash is recognisable.
_TEMPORARY = ".ai-arch-"
# Errors that say the path holds the wrong kind of thing, not that the filesystem failed: a
# folder, a pipe with no reader or a device where a file was meant, and the other way round.
_NOT_A_FILE = frozenset({errno.EISDIR, errno.ENXIO})
_WRONG_KIND = frozenset({errno.ENOTDIR, errno.ENOTEMPTY, errno.EEXIST})

# One write at a time in this process (M2). A move by link and unlink is two steps, and so is an
# append (open, then write): a replace landing between them makes the move remove a file it was
# never asked to move, or the append go to the file replaced. ReAct runs a turn's calls in
# parallel, so the four write tools, of any policy, take turns. A write waits ``_WAIT_S`` seconds
# at most: one that takes longer is stuck (a dead network mount), and the next fails, retryable,
# instead of running after its call gave up.
_ONE_WRITE = threading.Lock()
_WAIT_S = 30.0


@contextmanager
def _one_write() -> Iterator[None]:
    """Hold the process's write turn.

    Raises:
        ToolFailure: upstream, retryable, when another write has held it for ``_WAIT_S``.
    """
    if not _ONE_WRITE.acquire(timeout=_WAIT_S):
        msg = (
            f"another write of this process has been running for over {_WAIT_S:g} s, and this "
            "one waits for it; try again in a while."
        )
        raise ToolFailure("upstream", msg, retryable=True)
    try:
        yield
    finally:
        _ONE_WRITE.release()


def filesystem_tools(policy: FilesystemPolicy) -> tuple[Callable[..., str | ToolResult], ...]:
    """The filesystem tools bound to ``policy``: the file tools to give an agent.

    With ``read_roots``, ``read_file``, ``list_directory`` and ``search_files``, with the names,
    schemas and governance of the module-level ones; with ``write_roots``, ``write_file``,
    ``append_file``, ``make_directory`` and ``move_path``. Each one checks its paths with
    ``policy`` right before it acts, whatever gates the run has, so a path an approver changed,
    or a folder swapped for a link while a person decided, fails the call (D64). All of them
    need approval, call by call (D4). The writes take turns: one at a time in the process.

    Raises:
        ValueError: The policy has neither ``read_roots`` nor ``write_roots``.
        NotImplementedError: The policy has ``write_roots`` and this is Windows: the writes need
            ``dir_fd`` and ``O_NOFOLLOW``, which Windows lacks (C07.8). The reads work there.
    """
    if not policy.read_roots and not policy.write_roots:
        msg = "the FilesystemPolicy has no read_roots and no write_roots: it gives no tools"
        raise ValueError(msg)
    reads = bound_reads(policy) if policy.read_roots else ()
    if policy.write_roots and sys.platform == "win32":
        msg = (
            "the filesystem write tools are POSIX only (they need dir_fd and O_NOFOLLOW); on "
            "Windows, build the FilesystemPolicy with read_roots alone"
        )
        raise NotImplementedError(msg)
    writes = tuple(build(policy) for build in _WRITES) if policy.write_roots else ()
    # Each checks its own paths with the policy: the gate leaves it what is not a path (L1).
    return tuple(checks_paths(fn, policy) for fn in (*reads, *writes))


# --- The tools -----------------------------------------------------------------------------------


def _write_file_tool(policy: FilesystemPolicy) -> Callable[..., str]:
    @governed(_WRITE_REASON, preview=_previewed(_write_preview, policy))
    def write_file(
        path: str, content: str, overwrite: bool = False, create_parents: bool = False
    ) -> str:
        """Write a text file whole, in UTF-8, exactly as given (line ends included).

        The file takes its new text in one step, so no one ever reads half of it. Without
        ``overwrite``, a file that exists is left as it is and the call fails. The answer gives
        the file's full path and its size in bytes.

        Args:
            path: The file, relative to the working folder or absolute.
            content: The whole text of the file.
            overwrite: Replace the file if it exists, keeping its permissions.
            create_parents: Make the folders on the way that do not exist yet.

        Raises:
            ToolFailure: permission_denied when the path is outside the folders this run may
                write to, or is a link; validation_error when the file exists without
                ``overwrite``, the path is a folder or goes through a file, or ``content`` is
                not text or is too long; not_found when its folder is missing; upstream when
                the OS fails the write.
        """
        with _one_write():
            return _write(
                policy, path, content, overwrite=overwrite, create_parents=create_parents
            )

    return write_file


def _append_file_tool(policy: FilesystemPolicy) -> Callable[..., str]:
    @governed(_APPEND_REASON, preview=_previewed(_append_preview, policy))
    def append_file(path: str, content: str) -> str:
        """Add text, in UTF-8, at the end of a text file that exists.

        The answer gives how many bytes were added and the file's new size. ``write_file``
        makes a file that does not exist yet.

        Args:
            path: The file, relative to the working folder or absolute.
            content: The text to add, line ends included.

        Raises:
            ToolFailure: permission_denied when the path is outside the folders this run may
                write to, is a link, or has other hard links; not_found when the file is
                missing; validation_error when it is not a regular file, the path goes through
                a file, or ``content`` is not text or is too long; upstream when the OS fails
                the write.
        """
        with _one_write():
            return _append(policy, path, content)

    return append_file


def _make_directory_tool(policy: FilesystemPolicy) -> Callable[..., str]:
    @governed(_MAKE_REASON, preview=_previewed(_make_preview, policy))
    def make_directory(path: str, parents: bool = False) -> str:
        """Make a folder. One that exists already is fine, and the answer says so.

        Args:
            path: The folder, relative to the working folder or absolute.
            parents: Make the folders on the way that do not exist yet.

        Raises:
            ToolFailure: permission_denied when the path is outside the folders this run may
                write to, or is a link; not_found when a folder on the way is missing;
                validation_error when a file is in the way; upstream when the OS fails.
        """
        with _one_write():
            return _make(policy, path, parents=parents)

    return make_directory


def _move_path_tool(policy: FilesystemPolicy) -> Callable[..., str]:
    @governed(_MOVE_REASON, preview=_previewed(_move_preview, policy))
    def move_path(source: str, destination: str, overwrite: bool = False) -> str:
        """Move or rename a file or a folder, within one volume: it never copies.

        Without ``overwrite``, an existing destination is left as it is and the call fails.
        A file the agent cannot read never moves to a folder it reads.

        Args:
            source: The file or folder to move.
            destination: Its new path, name included.
            overwrite: Replace what is at the destination: a file, or an empty folder.

        Raises:
            ToolFailure: permission_denied when either path is outside the folders this run
                may write to, or is a link, or the move would let the agent read what it could
                not; not_found when the source is missing; validation_error when the
                destination exists without ``overwrite``, is inside the source, is the same
                file under another name, or is on another volume; upstream when the OS fails
                the move.
        """
        with _one_write():
            return _move(policy, source, destination, overwrite=overwrite)

    return move_path


_WRITES = (_write_file_tool, _append_file_tool, _make_directory_tool, _move_path_tool)


# --- The previews (C07c) -------------------------------------------------------------------------
#
# What a call will do, for the person who approves it and for a dry run: the action, the canonical
# path and the size change, and for a file replaced the lines that change. A preview checks its
# paths with the policy and looks from the root down without following a link, as the tool will;
# it never writes. It is a picture of the files when it ran, not a lock on them.

type _Picture = Callable[[FilesystemPolicy, dict[str, Any]], str]

_DIFF_LINES = 80
_DIFF_BYTES = 8 * 1024
_DIFF_FILE_BYTES = 256 * 1024  # a file, or new text, larger than this gets no diff


def _previewed(picture: _Picture, policy: FilesystemPolicy) -> ToolPreview:
    """The preview hook of a write tool: what ``picture`` sees the call doing, or why it will
    fail, in the words the tool would use."""

    def preview(arguments: dict[str, Any]) -> str:
        try:
            return picture(policy, arguments)
        except FilesystemPolicyError as e:  # a link where the call would act
            failure = ToolFailure("permission_denied", str(e))
        except ToolFailure as e:
            failure = e
        return f"will fail ({failure.error.type}): {failure.error.message}"

    return preview


def _write_preview(policy: FilesystemPolicy, arguments: dict[str, Any]) -> str:
    """``create /…/a.md (12 bytes)``, or ``replace /…/a.md (1204 → 1311 bytes)`` and the diff."""
    data = _encoded(policy, arguments.get("content"))
    target = checked(policy, arguments.get("path"), "write")
    found = _look(policy, target, "write", makes_folders=bool(arguments.get("create_parents")))
    if found is None:
        return f"create {target} ({len(data)} bytes)"
    _refuse_unless_a_file(found, target)
    if not arguments.get("overwrite"):
        raise ToolFailure("validation_error", _exists(target, "pass overwrite=true"))
    summary = f"replace {target} ({found.st_size} → {len(data)} bytes)"
    return "\n".join((summary, *_diff(policy, target, found.st_size, data)))


def _append_preview(policy: FilesystemPolicy, arguments: dict[str, Any]) -> str:
    data = _encoded(policy, arguments.get("content"))
    target = checked(policy, arguments.get("path"), "write")
    found = _look(policy, target, "append to")
    if found is None:
        raise ToolFailure("not_found", f"{target} does not exist; write_file makes it.")
    _refuse_unless_a_file(found, target)
    if found.st_nlink != 1:
        msg = (
            f"{target} has {found.st_nlink} hard links, and append_file never changes a file "
            "other paths reach too."
        )
        raise ToolFailure("permission_denied", msg)
    size = found.st_size
    return f"append {len(data)} bytes to {target} ({size} → {size + len(data)} bytes)"


def _make_preview(policy: FilesystemPolicy, arguments: dict[str, Any]) -> str:
    target = checked(policy, arguments.get("path"), "write")
    found = _look(policy, target, "make", makes_folders=bool(arguments.get("parents")))
    if found is None:
        return f"make folder {target}"
    if stat.S_ISDIR(found.st_mode):
        return f"{target} already exists; nothing changes"
    if stat.S_ISLNK(found.st_mode):
        raise FilesystemPolicyError(_a_link(target))
    raise ToolFailure(
        "validation_error", f"{target} exists and is not a folder; pick another name."
    )


def _move_preview(policy: FilesystemPolicy, arguments: dict[str, Any]) -> str:
    origin, target = _move_paths(policy, arguments.get("source"), arguments.get("destination"))
    found = _look(policy, origin, "move")
    if found is None:
        msg = f"{origin} does not exist; list_directory shows what is there."
        raise ToolFailure("not_found", msg)
    if stat.S_ISLNK(found.st_mode):
        raise FilesystemPolicyError(_a_link(origin))
    summary = f"move {_kind(found)} {origin} to {target}"
    there = _look(policy, target, "move")
    if there is None:
        return summary
    if os.path.samestat(found, there):
        _refuse_two_names_now(policy, origin, target)
    if not arguments.get("overwrite"):
        raise ToolFailure("validation_error", _exists(target, "pass overwrite=true"))
    size = "" if stat.S_ISDIR(there.st_mode) else f" ({there.st_size} bytes)"
    return f"{summary}, replacing the {_kind(there)} there{size}"


def _refuse_two_names_now(policy: FilesystemPolicy, origin: Path, target: Path) -> None:
    """``_refuse_two_names`` for a preview, its folders opened from their root down."""
    try:
        with (
            opened_folder(policy, origin.parent, "write") as source_folder,
            opened_folder(policy, target.parent, "write") as target_folder,
        ):
            _refuse_two_names((source_folder, origin), (target_folder, target))
    except (OSError, ValueError) as e:
        raise _failure(e, "move", origin) from e


def _kind(found: os.stat_result) -> str:
    return "folder" if stat.S_ISDIR(found.st_mode) else "file"


def _look(
    policy: FilesystemPolicy, target: Path, verb: str, *, makes_folders: bool = False
) -> os.stat_result | None:
    """What is at the canonical ``target`` now, reached from its root down as the tool will,
    links not followed; ``None`` when nothing is there, or its folder is missing and the call
    ``makes_folders``.

    Raises:
        ToolFailure: not_found when its folder is missing and the call does not make it; what
            ``_failure`` says when the walk fails (a link on the way is permission_denied).
    """
    try:
        with opened_folder(policy, target.parent, "write") as folder:
            return _entry(folder, target.name)
    except FileNotFoundError as e:
        if makes_folders:
            return None
        raise ToolFailure("not_found", f"the folder {target.parent} does not exist.") from e
    except (OSError, ValueError) as e:
        raise _failure(e, verb, target) from e


def _diff(policy: FilesystemPolicy, target: Path, size: int, data: bytes) -> list[str]:
    """The lines a replace of the ``size``-byte file by ``data`` changes, as a unified diff cut at
    80 lines or 8 KB, or why there is none."""
    try:
        old = _text_now(policy, target, max(size, len(data)))
    except _NoDiff as e:
        return [f"(no diff: {e})"]
    new = data.decode("utf-8").splitlines()
    lines = list(
        difflib.unified_diff(old.splitlines(), new, str(target), str(target), lineterm="")
    )
    if not lines:
        return ["(the same lines of text)"]
    shown = "\n".join(lines[:_DIFF_LINES]).encode()
    if len(lines) <= _DIFF_LINES and len(shown) <= _DIFF_BYTES:
        return lines
    cut = shown[:_DIFF_BYTES].decode("utf-8", errors="ignore")
    note = f"[the diff goes on: {len(lines)} lines in all, cut at {_DIFF_LINES} lines or 8 KB]"
    return [cut, note]


class _NoDiff(Exception):
    """Why a replace shows no diff."""


def _text_now(policy: FilesystemPolicy, target: Path, largest: int) -> str:
    """The file's text now, opened from its root down without following a link, as
    ``append_file`` opens it.

    Raises:
        FilesystemPolicyError: A link took the place of a folder on the way, or of the file.
        _NoDiff: The file or the new text (``largest`` is the larger size) is over 256 KB, or the
            file cannot be read, or is not UTF-8 text.
    """
    if largest > _DIFF_FILE_BYTES:
        raise _NoDiff("over 256 KB")
    try:
        descriptor = open_beneath(policy, target, "write", os.O_RDONLY)
    except FilesystemPolicyError:
        raise
    except (OSError, ValueError) as e:  # replacing a file needs no read of it
        raise _NoDiff("the file cannot be read") from e
    with open(descriptor, "rb") as handle:  # closes the descriptor
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):  # swapped since the look
            raise _NoDiff("the file is not text")
        raw = handle.read(_DIFF_FILE_BYTES + 1)
    if len(raw) > _DIFF_FILE_BYTES:
        raise _NoDiff("over 256 KB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _NoDiff("the file is not text") from e
    if "\0" in text:
        raise _NoDiff("the file is not text")
    return text


# --- write_file ----------------------------------------------------------------------------------


def _write(
    policy: FilesystemPolicy, path: str, content: object, *, overwrite: bool, create_parents: bool
) -> str:
    data = _encoded(policy, content)
    target = checked(policy, path, "write")
    try:
        with opened_folder(policy, target.parent, "write", create=create_parents) as folder:
            replaced = _publish(folder, target, data, overwrite=overwrite)
            note = _synced(folder)
    except FileNotFoundError as e:
        msg = (
            f"the folder {target.parent} does not exist; pass create_parents=true, or make it "
            "with make_directory."
        )
        raise ToolFailure("not_found", msg) from e
    except (OSError, ValueError) as e:
        raise _failure(e, "write", target) from e
    if replaced is None:
        return f"Created {target} ({len(data)} bytes){note}"
    return f"Replaced {target} ({replaced} → {len(data)} bytes){note}"


def _encoded(policy: FilesystemPolicy, content: object) -> bytes:
    """``content`` in UTF-8, if it is text within the policy's ``max_write_bytes``."""
    if not isinstance(content, str):
        msg = f"content must be text (a string); got {type(content).__name__}."
        raise ToolFailure("validation_error", msg)
    if len(content) > policy.max_write_bytes:  # each character takes a byte at least
        raise _too_long(policy, f"at least {len(content)}")
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as e:
        msg = (
            f"content is not valid UTF-8 text ({e.reason} at character {e.start}); send it "
            "without lone surrogates."
        )
        raise ToolFailure("validation_error", msg) from e
    if len(data) > policy.max_write_bytes:
        raise _too_long(policy, str(len(data)))
    return data


def _too_long(policy: FilesystemPolicy, size: str) -> ToolFailure:
    msg = (
        f"content is {size} bytes, and this run writes at most {policy.max_write_bytes} at a "
        "time; write the start with write_file and add the rest with append_file."
    )
    return ToolFailure("validation_error", msg)


def _publish(folder: int, target: Path, data: bytes, *, overwrite: bool) -> int | None:
    """Put ``data`` at ``target`` (inside ``folder``) in one step; the size of the file it
    replaced, or ``None`` when there was none. The caller syncs the folder."""
    existing = _entry(folder, target.name)
    if existing is not None:
        _refuse_unless_a_file(existing, target)
        if not overwrite:
            raise ToolFailure("validation_error", _exists(target, "pass overwrite=true"))
    mode = None if existing is None else stat.S_IMODE(existing.st_mode) & 0o777
    temporary = _temporary(folder, data, mode)
    try:
        if overwrite:
            os.replace(temporary, target.name, src_dir_fd=folder, dst_dir_fd=folder)
        else:  # another process's write since the look above makes this fail
            _link((folder, temporary), (folder, target))
    finally:
        _discard(folder, temporary)  # after os.replace the name is already gone
    return None if existing is None else existing.st_size


def _synced(*folders: int) -> str:
    """Sync each folder to disk, so the names just put in it survive a crash; a note for the
    answer when one cannot be, or ``""``.

    The names are in place by then, so the write is done: a filesystem that cannot sync a
    folder (``EINVAL`` on some) makes it a note, never a failure that a retry would find
    already done (L2).
    """
    failed: list[str] = []
    for folder in folders:
        try:
            os.fsync(folder)
        except OSError as e:
            failed.append(e.strerror or str(e))
    if not failed:
        return ""
    return (
        f"; the folder could not be synced to disk ({failed[0]}): the change is made, but may "
        "not survive a crash"
    )


def _temporary(folder: int, data: bytes, mode: int | None) -> str:
    """The name of a new file in ``folder`` that holds ``data``, synced to disk.

    It is made with ``O_EXCL`` and ``O_NOFOLLOW``, with mode ``0o666`` under the umask, or
    ``mode``; it is removed if anything fails.
    """
    name = f"{_TEMPORARY}{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(name, flags, 0o666, dir_fd=folder)
    try:
        try:
            if mode is not None:
                os.fchmod(descriptor, mode)
            _write_all(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        _discard(folder, name)
        raise
    return name


# What a filesystem without hard links (FAT, some network mounts) answers ``link`` with.
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP})


def _link(source: tuple[int, str], target: tuple[int, Path]) -> None:
    """Give the entry ``source`` (its folder, its name) the name of ``target`` too, which
    must not exist: ``os.link`` refuses a name that exists, in one step."""
    (source_folder, name), (target_folder, path) = source, target
    try:
        os.link(
            name,
            path.name,
            src_dir_fd=source_folder,
            dst_dir_fd=target_folder,
            follow_symlinks=False,
        )
    except FileExistsError as e:
        raise ToolFailure("validation_error", _exists(path, "pass overwrite=true")) from e
    except OSError as e:
        if e.errno not in _NO_HARD_LINKS:
            raise
        msg = (
            f"cannot make {path} without risking another file of that name: this filesystem "
            f"has no hard links ({e.strerror}). Pass overwrite=true if replacing a file there "
            "is fine."
        )
        raise ToolFailure("upstream", msg) from e


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view) :]


def _discard(folder: int, name: str) -> None:
    """Remove ``name`` from ``folder`` if it is there; a cleanup never hides the error."""
    with suppress(OSError):
        os.unlink(name, dir_fd=folder)


def _entry(folder: int, name: str) -> os.stat_result | None:
    """What ``name`` in ``folder`` is, links not followed; ``None`` when nothing is there."""
    try:
        return os.stat(name, dir_fd=folder, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _refuse_unless_a_file(found: os.stat_result, target: Path) -> None:
    if stat.S_ISLNK(found.st_mode):
        raise FilesystemPolicyError(_a_link(target))
    if not stat.S_ISREG(found.st_mode):
        raise ToolFailure("validation_error", _not_a_file(target))


def _not_a_file(target: Path) -> str:
    return f"{target} is not a regular file (a folder, a pipe or a device); pick a file."


def _a_link(target: Path) -> str:
    return (
        f"{target} is a symbolic link, and the filesystem tools never write through one or "
        "move one; use the path it points to."
    )


def _exists(target: Path, how: str) -> str:
    return f"{target} already exists; {how} to replace it, or pick another name."


def _failure(error: OSError | ValueError, verb: str, path: Path) -> ToolFailure:
    """The failure for an error a write raised: what ``path_failure`` says, unless the error
    is the write's own (another volume, the wrong kind of thing, the OS's refusal)."""
    code = None if isinstance(error, FilesystemPolicyError | ValueError) else error.errno
    if code == errno.EXDEV:
        msg = (
            f"{path} is on another volume than the destination, and move_path never copies; "
            "move it within one volume."
        )
        return ToolFailure("validation_error", msg)
    if code in _NOT_A_FILE:
        return ToolFailure("validation_error", _not_a_file(path))
    if code in _WRONG_KIND:
        msg = f"cannot {verb} {path}: {os.strerror(code or 0)}; pick another path."
        return ToolFailure("validation_error", msg)
    if code is not None and isinstance(error, PermissionError):
        msg = f"permission denied to {verb} {path}; pick a folder this process can write to."
        return ToolFailure("upstream", msg)
    return path_failure(error, verb, str(path))


# --- append_file ---------------------------------------------------------------------------------


def _append(policy: FilesystemPolicy, path: str, content: object) -> str:
    data = _encoded(policy, content)
    target = checked(policy, path, "write")
    try:
        descriptor = open_beneath(policy, target, "write", os.O_WRONLY | os.O_APPEND)
    except FileNotFoundError as e:
        msg = f"{target} does not exist; write_file makes it."
        raise ToolFailure("not_found", msg) from e
    except (OSError, ValueError) as e:
        raise _failure(e, "append to", target) from e
    try:
        size = _append_to(descriptor, target, data)
    except OSError as e:
        raise _failure(e, "append to", target) from e
    finally:
        os.close(descriptor)
    return f"Appended {len(data)} bytes to {target} (now {size} bytes)"


def _append_to(descriptor: int, target: Path, data: bytes) -> int:
    """Add ``data`` to the regular file open at ``descriptor``, if no other path reaches it;
    its new size."""
    found = os.fstat(descriptor)
    if not stat.S_ISREG(found.st_mode):
        raise ToolFailure("validation_error", _not_a_file(target))
    if found.st_nlink != 1:
        msg = (
            f"{target} has {found.st_nlink} hard links, so appending would change a file other "
            "paths reach too, perhaps outside the folders this run may write to; write a new "
            "file with write_file."
        )
        raise ToolFailure("permission_denied", msg)
    _write_all(descriptor, data)
    os.fsync(descriptor)
    return os.fstat(descriptor).st_size


# --- make_directory ------------------------------------------------------------------------------


def _make(policy: FilesystemPolicy, path: str, *, parents: bool) -> str:
    target = checked(policy, path, "write")
    try:
        with opened_folder(policy, target.parent, "write", create=parents) as folder:
            made = _made(folder, target)
            note = _synced(folder) if made else ""
    except FileNotFoundError as e:
        msg = f"the folder {target.parent} does not exist; pass parents=true to make it too."
        raise ToolFailure("not_found", msg) from e
    except (OSError, ValueError) as e:
        raise _failure(e, "make", target) from e
    return f"Created folder {target}{note}" if made else f"{target} already exists"


def _made(folder: int, target: Path) -> bool:
    """Make ``target`` in ``folder``, mode ``0o777`` under the umask; ``False`` when it is a
    folder already."""
    try:
        os.mkdir(target.name, 0o777, dir_fd=folder)
    except FileExistsError:
        found = os.stat(target.name, dir_fd=folder, follow_symlinks=False)
        if stat.S_ISDIR(found.st_mode):
            return False
        if stat.S_ISLNK(found.st_mode):
            raise FilesystemPolicyError(_a_link(target)) from None
        msg = f"{target} exists and is not a folder; pick another name."
        raise ToolFailure("validation_error", msg) from None
    return True


# --- move_path -----------------------------------------------------------------------------------


def _move(policy: FilesystemPolicy, source: str, destination: str, *, overwrite: bool) -> str:
    origin, target = _move_paths(policy, source, destination)
    try:
        with (
            opened_folder(policy, origin.parent, "write") as source_folder,
            opened_folder(policy, target.parent, "write") as target_folder,
        ):
            moved = (source_folder, origin), (target_folder, target)
            left = _relocate(*moved, overwrite=overwrite)
            note = _synced(source_folder, target_folder)
    except FileNotFoundError as e:
        msg = (
            f"{origin.parent} or {target.parent} does not exist; list_directory shows what is "
            "there, and make_directory makes a folder."
        )
        raise ToolFailure("not_found", msg) from e
    except (OSError, ValueError) as e:
        raise _failure(e, "move", origin) from e
    kept = f"; {origin} no longer names the file that moved, and is left as it is" if left else ""
    return f"Moved {origin} to {target}{kept}{note}"


def _move_paths(
    policy: FilesystemPolicy, source: object, destination: object
) -> tuple[Path, Path]:
    """The canonical source and destination of a move the policy allows.

    Raises:
        ToolFailure: What ``checked`` says of either; validation_error when the destination is
            the source or inside it; permission_denied when the move would let the agent read
            what it could not (``check_move``, M3).
    """
    origin = checked(policy, source, "write", argument="source")
    target = checked(policy, destination, "write", argument="destination")
    if target.is_relative_to(origin):
        msg = f"{target} is {origin} or inside it; pick a destination outside the source."
        raise ToolFailure("validation_error", msg)
    try:
        check_move(policy, origin, target)
    except FilesystemPolicyError as e:
        raise ToolFailure("permission_denied", str(e)) from e
    return origin, target


def _relocate(source: tuple[int, Path], destination: tuple[int, Path], *, overwrite: bool) -> bool:
    """Give the entry at ``source`` (its folder, its path) the name at ``destination``; whether
    the source's name was left in place because another file took it meanwhile.

    Without ``overwrite`` a file moves by ``os.link`` and ``unlink``, which refuses a name that
    exists in one step; a folder by ``os.rename`` after a look, so only an empty folder made in
    between could be replaced. The caller syncs both folders.
    """
    (source_folder, origin), (target_folder, target) = source, destination
    found = _entry(source_folder, origin.name)
    if found is None:
        msg = f"{origin} does not exist; list_directory shows what is there."
        raise ToolFailure("not_found", msg)
    if stat.S_ISLNK(found.st_mode):
        raise FilesystemPolicyError(_a_link(origin))
    there = _entry(target_folder, target.name)
    if there is not None and os.path.samestat(found, there):
        _refuse_two_names(source, destination)
    names = (origin.name, target.name)
    if overwrite:
        os.replace(*names, src_dir_fd=source_folder, dst_dir_fd=target_folder)
        return False
    if stat.S_ISREG(found.st_mode):
        return not _link_then_unlink(source, destination)
    if there is not None:
        raise ToolFailure("validation_error", _exists(target, "pass overwrite=true"))
    os.rename(*names, src_dir_fd=source_folder, dst_dir_fd=target_folder)
    return False


def _refuse_two_names(source: tuple[int, Path], destination: tuple[int, Path]) -> None:
    """Refuse a move between two hard links of one file, which POSIX ``rename`` leaves as they
    are while saying it moved (L3).

    The same file under one entry is not that: on a case-insensitive disk, ``a.txt`` and
    ``A.txt`` are one entry, and a rename that changes the case goes on.

    Raises:
        ToolFailure: validation_error when the two paths are two entries of one file.
    """
    (source_folder, origin), (target_folder, target) = source, destination
    if os.path.samestat(os.fstat(source_folder), os.fstat(target_folder)):
        if origin.name == target.name:
            return  # one entry, spelled in two ways through its folders
        entries = set(os.listdir(source_folder))
        if origin.name not in entries or target.name not in entries:
            return  # one entry under two spellings: a rename of its case or form
    msg = (
        f"{origin} and {target} are the same file, under two names (hard links), so a move "
        "between them would change nothing; leave it, or move it to another name."
    )
    raise ToolFailure("validation_error", msg)


def _link_then_unlink(source: tuple[int, Path], destination: tuple[int, Path]) -> bool:
    """Give the file at ``source`` the name at ``destination`` too, then take its old name away,
    unless another file has that name by then; whether it was taken away.

    The two steps are not one: between them another process may give the old name to another
    file, which the unlink would then remove (M2). The old name is removed only while it still
    names the file the new one does; the tools of this process never come between, since they
    take turns (``_one_write``).
    """
    (source_folder, origin), (target_folder, target) = source, destination
    _link((source_folder, origin.name), (target_folder, target))
    try:
        moved = os.stat(target.name, dir_fd=target_folder, follow_symlinks=False)
        left = _entry(source_folder, origin.name)
        if left is None or not os.path.samestat(left, moved):
            return False
        os.unlink(origin.name, dir_fd=source_folder)
    except BaseException:
        _discard(target_folder, target.name)  # the file stays where it was
        raise
    return True
