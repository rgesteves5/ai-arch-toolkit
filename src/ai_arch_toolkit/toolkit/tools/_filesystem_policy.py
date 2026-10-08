"""Where the filesystem tools may go: ``FilesystemPolicy``, its gate and its walk (C07, D64).

A policy names the folders each action may reach. Its ``check`` is the one test of a path, and it
runs twice for each call: in the ``PathScopeGate``, before anyone is asked to approve it, and in
the tool, right before its system call. The tool then opens what it touches from the root down, one
folder at a time and never through a link, so a folder swapped for a link after the check fails
the call instead of leading it out of the roots.
"""

from __future__ import annotations

import asyncio
import errno
import os
import stat
import sys
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal, cast, get_args

from ai_arch_toolkit.core import ExecutionContext, GateBlock, GateModify, GateResult

type FilesystemAction = Literal["read", "write", "delete"]
"""What a tool does to a path: read it, write it (create, replace, append, move), or delete it."""

_ACTIONS: frozenset[str] = frozenset(get_args(FilesystemAction.__value__))
_VERBS: dict[str, str] = {"read": "read", "write": "write to", "delete": "delete from"}

# The path arguments of the filesystem tools, and what each one does (D64, C07.2).
_DEFAULT_PATHS: Mapping[str, Mapping[str, FilesystemAction]] = {
    "read_file": {"path": "read"},
    "list_directory": {"path": "read"},
    "search_files": {"directory": "read"},
    "write_file": {"path": "write"},
    "append_file": {"path": "write"},
    "make_directory": {"path": "write"},
    "move_path": {"source": "write", "destination": "write"},
}


class FilesystemPolicyError(PermissionError):
    """A path the policy does not allow; the message says why and what to do instead."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FilesystemPolicy:
    """The folders a run may reach, by action.

    Each root must be an existing folder; it is stored canonical (links resolved, absolute), and
    so is ``cwd``. A relative root or ``cwd`` is taken from the process's working directory when
    the policy is built; the agent's relative paths start from ``cwd``.

    A path is read inside ``read_roots``; it is written or deleted inside a folder of
    ``write_roots`` or ``delete_roots``, and never as a root itself: a root, or a folder that
    holds one, is never created, moved or replaced.

    Attributes:
        read_roots: The folders whose files may be read, at any depth.
        write_roots: The folders inside which files and folders may be created, replaced,
            appended to and moved.
        delete_roots: The folders inside which an app's or an MCP server's tool may delete; no
            toolkit tool deletes (D64, C07.7).
        cwd: The folder relative paths start from; ``None`` is the process's working directory
            when the policy is built.
        max_write_bytes: The most bytes one write or append may carry.

    Raises:
        ValueError: A root or ``cwd`` is not an existing folder, or ``max_write_bytes`` is not a
            positive integer.
        TypeError: A roots field is a single path instead of a tuple of them.
    """

    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    delete_roots: tuple[Path, ...] = ()
    cwd: Path | None = None
    max_write_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        cwd = _existing_folder(Path.cwd() if self.cwd is None else self.cwd, "cwd")
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "read_roots", _roots(self.read_roots, "read_roots"))
        object.__setattr__(self, "write_roots", _roots(self.write_roots, "write_roots"))
        object.__setattr__(self, "delete_roots", _roots(self.delete_roots, "delete_roots"))
        size = self.max_write_bytes
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            msg = f"FilesystemPolicy.max_write_bytes must be a positive integer, got {size!r}"
            raise ValueError(msg)

    def roots(self, action: FilesystemAction) -> tuple[Path, ...]:
        """The canonical roots of ``action``."""
        if action not in _ACTIONS:
            msg = f"unknown filesystem action {action!r}; use one of {sorted(_ACTIONS)}"
            raise ValueError(msg)
        if action == "read":
            return self.read_roots
        return self.write_roots if action == "write" else self.delete_roots

    def check(self, path: str | os.PathLike[str], action: FilesystemAction) -> Path:
        """The canonical path of ``path``, if the policy lets ``action`` reach it.

        A relative path starts from ``cwd``, and ``~`` is the user's home. Only
        ``os.path.realpath(strict=os.path.ALLOW_MISSING)`` resolves it, so a path that does not
        exist yet resolves as far as it does. To read, the path's target must be inside a read
        root. To write or delete, the folder that holds it must be inside a root of the action,
        and its last part is not resolved: it is never a link, ``.`` or ``..``, and the path is
        never a root, nor a folder that holds one. Case is never folded: on a case-insensitive
        disk a path spelled in another case than its root is refused, never let through.

        Raises:
            FilesystemPolicyError: The policy does not allow it, or the path cannot be resolved
                (a part of it is a file, or links loop).
            ValueError: ``action`` is unknown, or the path is malformed (a NUL byte).
            TypeError: ``path`` is not a text path.
            OSError: The path cannot be read to resolve it (a name too long, a folder the
                process may not search).
        """
        roots = self.roots(action)
        given = os.path.expanduser(os.fspath(path))
        if not roots:
            raise FilesystemPolicyError(
                f"this policy lets the agent {_VERBS[action]} no folder, so {given!r} cannot be "
                f"reached; {action} only what the run's folders allow."
            )
        full = os.path.join(cast("Path", self.cwd), given)
        if action == "read":
            canonical = Path(_resolved(full, given))
            held = canonical  # the target itself must be inside
        else:
            held, canonical = self._leaf(full, given, action)  # the folder that holds it must be
        if not any(held.is_relative_to(root) for root in roots):
            names = ", ".join(map(str, roots))
            raise FilesystemPolicyError(
                f"{given!r} is {canonical}, outside the folders this policy lets the agent "
                f"{_VERBS[action]} ({names}); pick a path inside one of them."
            )
        if sys.platform == "win32" and os.path.isreserved(canonical):
            raise FilesystemPolicyError(f"{given!r} is a name Windows reserves; pick another.")
        return canonical

    def root_of(self, path: Path, action: FilesystemAction) -> Path:
        """The deepest root of ``action`` that holds the canonical ``path`` (or is it).

        Raises:
            FilesystemPolicyError: No root of ``action`` holds it.
        """
        held = [root for root in self.roots(action) if path.is_relative_to(root)]
        if not held:
            raise FilesystemPolicyError(f"{path} is outside the folders of this policy.")
        return max(held, key=lambda root: len(root.parts))

    def _leaf(self, full: str, given: str, action: str) -> tuple[Path, Path]:
        """The canonical folder that holds the path, and the path: that folder and the path's
        unresolved last part, which must name something, never a link nor a root."""
        head, leaf = os.path.split(full.rstrip("/") or "/")
        if not given or leaf in ("", ".", ".."):
            raise FilesystemPolicyError(
                f"{given!r} names no file or folder to {action}; give the path of the file or "
                "folder itself."
            )
        parent = Path(_resolved(head, given))
        canonical = parent / leaf
        if self._protects(canonical):
            raise FilesystemPolicyError(
                f"{canonical} is a folder this policy is rooted at, or holds one, and is never "
                f"created, moved or replaced; {action} a path inside it."
            )
        if _is_link(canonical, given):
            raise FilesystemPolicyError(
                f"{given!r} is a symbolic link, and this policy never writes through one or "
                "moves one; use the path it points to."
            )
        return parent, canonical

    def _protects(self, path: Path) -> bool:
        """Whether ``path`` is a root of any action or holds one, compared with case and Unicode
        form folded, so a case-insensitive disk is a false refusal, never a false acceptance."""
        folded = _folded(path)
        roots = (*self.read_roots, *self.write_roots, *self.delete_roots)
        return any(_folded(root).is_relative_to(folded) for root in roots)


def _folded(path: Path) -> PurePath:
    return PurePath(unicodedata.normalize("NFKC", str(path).casefold()))


def _existing_folder(path: str | os.PathLike[str], name: str) -> Path:
    """``path`` canonical, or ``ValueError`` unless it is an existing folder."""
    given = os.path.expanduser(os.fspath(path))
    try:
        canonical = os.path.realpath(given, strict=True)
    except (OSError, ValueError) as error:
        msg = f"FilesystemPolicy {name} {given!r} is not an existing folder: {error}"
        raise ValueError(msg) from error
    if not os.path.isdir(canonical):
        msg = f"FilesystemPolicy {name} {given!r} is not a folder"
        raise ValueError(msg)
    return Path(canonical)


def _roots(roots: object, name: str) -> tuple[Path, ...]:
    if isinstance(roots, str | os.PathLike) or not isinstance(roots, tuple | list):
        msg = f"FilesystemPolicy.{name} takes a tuple of folders, e.g. (Path('docs'),)"
        raise TypeError(msg)
    return tuple(_existing_folder(root, name) for root in cast("tuple[Any, ...]", roots))


def _resolved(path: str, given: str) -> str:
    """``path`` with every link and ``..`` resolved, as far as it exists (C07.4).

    Raises:
        FilesystemPolicyError: A part of it is a file, or its links loop.
    """
    try:
        return os.path.realpath(path, strict=os.path.ALLOW_MISSING)
    except OSError as error:
        if error.errno not in (errno.ENOTDIR, errno.ELOOP):
            raise
        raise FilesystemPolicyError(
            f"cannot resolve {given!r}: {error.strerror}; a part of it is a file, or its links "
            "loop. Pick another path."
        ) from error


def _is_link(path: Path, given: str) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False
    except NotADirectoryError as error:
        raise FilesystemPolicyError(
            f"cannot resolve {given!r}: the folder that would hold it is a file; pick another "
            "path."
        ) from error


# --- The walk from the root ---------------------------------------------------------------------
#
# POSIX only (D64, C07.8): every folder is opened relative to the one above it, with O_NOFOLLOW,
# so a link put in the place of a folder (or of the file) after the check fails the open. Opened
# with O_DIRECTORY, macOS reports such a link as ENOTDIR, not ELOOP.


def open_beneath(
    policy: FilesystemPolicy, path: Path, action: FilesystemAction, flags: int
) -> int:
    """A descriptor of the canonical ``path``, opened with ``flags`` from its root down.

    The last part is opened without following a link and without blocking (a pipe never holds
    the call); a root itself is opened as a folder. The caller closes the descriptor.

    On Windows, which has neither ``dir_fd`` nor ``O_NOFOLLOW``, only reads are bound, and the
    path is opened as it is: the time between the tool's check and this open stays open.

    Raises:
        FilesystemPolicyError: A link took the place of a folder, or of ``path``.
        OSError: The path cannot be opened.
    """
    if sys.platform == "win32":
        return os.open(path, flags | os.O_BINARY)
    root, names = _below(policy, path, action)
    folder = _walk(root, names[:-1], create=False)
    if not names:
        return folder
    try:
        return _opened(names[-1], flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, folder)
    finally:
        os.close(folder)


@contextmanager
def opened_folder(
    policy: FilesystemPolicy, folder: Path, action: FilesystemAction, *, create: bool = False
) -> Iterator[int]:
    """A descriptor of the canonical ``folder``, opened from its root down, folder by folder.

    With ``create``, a missing folder on the way is made (mode ``0o777`` under the umask) and
    then opened like the others, so a link made in its place in between fails the walk.

    Raises:
        FilesystemPolicyError: A link took the place of a folder on the way.
        FileNotFoundError: A folder on the way is missing, and ``create`` is false.
        OSError: A folder cannot be opened or made.
    """
    root, names = _below(policy, folder, action)
    descriptor = _walk(root, names, create=create)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _below(
    policy: FilesystemPolicy, path: Path, action: FilesystemAction
) -> tuple[Path, tuple[str, ...]]:
    """The root of ``path`` and the names that lead down from it to ``path``.

    The walk only goes down: a ``.`` or ``..`` among them (never in a path ``check`` gave) is
    refused, not followed.
    """
    root = policy.root_of(path, action)
    names = path.relative_to(root).parts
    if {".", ".."} & set(names):
        raise FilesystemPolicyError(f"{path} is not canonical; check it with the policy first.")
    return root, names


def _walk(root: Path, names: tuple[str, ...], *, create: bool) -> int:
    """The descriptor of ``root/names...``, each folder opened inside the one above it."""
    folder = _opened(str(root), _folder_flags(), None)
    try:
        for name in names:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(name, 0o777, dir_fd=folder)
            inner = _opened(name, _folder_flags(), folder)
            folder, outer = inner, folder
            os.close(outer)
    except BaseException:
        os.close(folder)
        raise
    return folder


def _folder_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _opened(name: str, flags: int, folder: int | None) -> int:
    try:
        return os.open(name, flags, dir_fd=folder)
    except OSError as error:
        if error.errno == errno.ELOOP or (
            error.errno == errno.ENOTDIR and _names_a_link(name, folder)
        ):
            raise FilesystemPolicyError(
                f"{name!r} became a symbolic link after the path was checked, and this policy "
                "never follows one; nothing was done. Check the folder and try again."
            ) from error
        raise


def _names_a_link(name: str, folder: int | None) -> bool:
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=folder, follow_symlinks=False).st_mode)
    except OSError:
        return False


# --- The gate -----------------------------------------------------------------------------------

_NO_DEFAULT = object()


class PathScopeGate:
    """Refuse, before anyone is asked, a call whose paths the policy does not allow (D64).

    Each tool's path arguments, and the action each one does, come from a map: the seven
    filesystem tools' (``path`` of the reads and writes, ``directory`` of ``search_files``,
    ``source`` and ``destination`` of ``move_path``), with ``paths`` added tool by tool (a tool
    it names gets that map, not a merge of the two). An omitted argument takes its schema
    default; one with no default is refused. A tool of ``capability="filesystem"`` with no map
    is refused.

    A refused call is a ``permission_denied`` block, never run nor metered, with what was
    refused under ``audit["filesystem"]``. An allowed one goes on with each path canonical, so
    the approver sees the paths that will be touched. The tools of ``filesystem_tools`` check
    again before they act: the gate spares the approver and the budget, the tool is the
    guarantee.

    Raises:
        ValueError: ``paths`` names an action that is not ``read``, ``write`` or ``delete``.
    """

    __slots__ = ("_paths", "_policy")

    def __init__(
        self,
        policy: FilesystemPolicy,
        *,
        paths: Mapping[str, Mapping[str, FilesystemAction]] | None = None,
    ) -> None:
        mapped = {**_DEFAULT_PATHS, **(paths or {})}
        for tool, arguments in mapped.items():
            wrong = set(arguments.values()) - _ACTIONS
            if wrong:
                msg = f"PathScopeGate paths[{tool!r}] names unknown actions {sorted(wrong)}"
                raise ValueError(msg)
        self._policy = policy
        self._paths = {tool: dict(arguments) for tool, arguments in mapped.items()}

    def check_sync(self, ctx: ExecutionContext) -> GateResult | None:
        name = ctx.tool_call.name
        mapped = self._paths.get(name)
        if mapped is None:
            if ctx.definition.policy.capability != "filesystem":
                return None
            return _refused(
                name,
                {"tool": name},
                f"The tool {name!r} reaches the filesystem, and this run maps none of its "
                "arguments to the folders it may reach, so it does not run. Map its path "
                f"arguments with PathScopeGate(paths={{{name!r}: {{'path': 'read'}}}}).",
            )
        arguments = dict(ctx.tool_call.input)
        checked: dict[str, dict[str, str]] = {}
        for argument, action in mapped.items():
            given = arguments.get(argument, _schema_default(ctx, argument))
            canonical = self._canonical(name, argument, action, given)
            if isinstance(canonical, GateBlock):
                return canonical
            arguments[argument] = str(canonical)
            checked[argument] = {"action": action, "path": str(canonical)}
        return GateModify(args=arguments, audit={"filesystem": checked})

    async def check(self, ctx: ExecutionContext) -> GateResult | None:
        # Resolving a path is I/O: off the loop.
        return await asyncio.to_thread(self.check_sync, ctx)

    def _canonical(
        self, name: str, argument: str, action: FilesystemAction, given: object
    ) -> GateBlock | Path:
        """The canonical path of ``given``, or the block that refuses it."""
        audit = {"tool": name, "argument": argument, "action": action, "path": given}
        if not isinstance(given, str):
            return _refused(
                name,
                audit,
                f"The tool {name!r} did not run: its argument {argument!r} must be a path, and "
                f"got {'nothing' if given is _NO_DEFAULT else type(given).__name__}. Give it.",
            )
        try:
            return self._policy.check(given, action)
        except FilesystemPolicyError as error:
            reason = str(error)
        except (OSError, ValueError) as error:
            reason = f"{given!r} cannot be checked ({error}); give another path."
        return _refused(name, audit, f"The tool {name!r} did not run: {reason}")


def _schema_default(ctx: ExecutionContext, argument: str) -> object:
    properties = ctx.definition.schema.input_schema.get("properties", {})
    spec = properties.get(argument) if isinstance(properties, dict) else None
    return spec.get("default", _NO_DEFAULT) if isinstance(spec, dict) else _NO_DEFAULT


def _refused(name: str, audit: dict[str, object], message: str) -> GateBlock:
    if audit.get("path") is _NO_DEFAULT:
        audit = {**audit, "path": None}
    return GateBlock(
        error_type="permission_denied",
        message=message,
        audit={"filesystem": {**audit, "refused": message}},
    )
