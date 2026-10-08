"""Filesystem tools — read files, list directories, search content (T09), and the same three
bound to a ``FilesystemPolicy``'s read roots (C07).

Every long answer goes through the window (D39): a file is read window by window, by characters
from its start; a listing and a search come a page at a time. A search gives each line's offset,
which ``read_file`` takes, so a match found is a call away from the text around it. Files are read
a chunk at a time, so neither a huge file nor one long line ever lands in memory whole.

The bound reads (``bound_reads``) keep the names, schemas and answers of the module's three. They
check each path with the policy first, open each file from its root down without following a link,
refuse a listing pattern that climbs out with ``..``, and leave out an entry whose target is
outside the roots.
"""

from __future__ import annotations

import errno
import functools
import io
import itertools
import os
import re
import stat
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TextIO

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure, line_cut
from ai_arch_toolkit.toolkit.tools._filesystem_policy import (
    FilesystemAction,
    FilesystemPolicy,
    FilesystemPolicyError,
    open_beneath,
)
from ai_arch_toolkit.toolkit.tools._window import Window, list_window

_DEFAULT_MAX_LINES = 200
_DEFAULT_MAX_RESULTS = 50
_MAX_LINES = 10_000
_MAX_RESULTS = 1_000
# The characters one window of a file shows at most, so one long line never comes back whole.
_WINDOW_CHARS = 100_000
# The characters read at a time to reach an offset or count what is left.
_CHUNK = 1 << 20
# A read counts what is left of a file up to here: past it, the footer gives no total, so a read
# never decodes a huge file to the end (text decodes at about 3 GB/s, other bytes at 0.1 GB/s).
_COUNT_CHARS = 100_000_000
_PAGE_ENTRIES = 1_000
# The matches a listing walks, inside the folder or not, before it asks for a narrower pattern.
_MAX_WALK = 100_000
# The longest piece of a line a search holds at once.
_PIECE = 1 << 16
# A matching line longer than this shows the part around its first match.
_EXCERPT = 300
_BEFORE = 100
# A search reads at most this many characters, in all its files: past them it stops and says
# where, so a huge tree or file takes seconds (text reads at about 120 million characters a
# second: 4 s), not past the executor's deadline.
_SCAN_CHARS = 500_000_000
# A file with a NUL in its first bytes is binary, as grep and git decide.
_SNIFF = 8192
# The folders and files a search names when it could not read them.
_UNREAD_NAMED = 3
_NARROWER = "search a narrower directory to see the rest"
_BINARY_SUFFIXES = frozenset({".pyc", ".pyo", ".so", ".dylib", ".exe", ".bin", ".gz", ".zip"})


def _is_regular_file(path: Path) -> bool:
    """Whether ``path`` is a file, without suppressing filesystem errors."""
    return stat.S_ISREG(path.stat().st_mode)


def _is_directory(path: Path) -> bool:
    """Whether ``path`` is a directory, without suppressing filesystem errors."""
    return stat.S_ISDIR(path.stat().st_mode)


def _error_text(error: Exception) -> str:
    """The useful OS detail, or the exception text for non-OS path errors."""
    return str(getattr(error, "strerror", None) or error)


# Errors that say the path itself is malformed, not that the filesystem failed.
_BAD_PATH_ERRNOS = frozenset({errno.ENAMETOOLONG, errno.EINVAL})


def path_failure(error: OSError | ValueError, action: str, path: str) -> ToolFailure:
    """The failure for an error the filesystem raised while ``action``-ing ``path``.

    A path a ``FilesystemPolicy`` refuses is permission_denied, in the policy's words; one that
    does not exist is not_found; a malformed one (a null byte, a name too long) is a
    validation_error; the OS's permission refusal or any other OS error is upstream.
    """
    if isinstance(error, FilesystemPolicyError):
        return ToolFailure("permission_denied", str(error))
    if isinstance(error, FileNotFoundError | NotADirectoryError):
        msg = f"no such file or directory: {path!r}; list_directory shows what is there."
        return ToolFailure("not_found", msg)
    if isinstance(error, PermissionError):
        msg = f"permission denied to {action} {path!r}; pick a path this process can read."
        return ToolFailure("upstream", msg)
    if isinstance(error, ValueError) or error.errno in _BAD_PATH_ERRNOS:
        msg = f"cannot {action} {path!r}: {_error_text(error)}; check the path."
        return ToolFailure("validation_error", msg)
    msg = f"cannot {action} {path!r}: {_error_text(error)}; try again, or pick another path."
    return ToolFailure("upstream", msg)


def checked(
    policy: FilesystemPolicy, path: object, action: FilesystemAction, *, argument: str = "path"
) -> Path:
    """The canonical ``path``, if ``policy`` lets ``action`` reach it: the tool's own check,
    right before its system call (D64, C07.1).

    Raises:
        ToolFailure: validation_error when ``path`` is not text or is malformed;
            permission_denied when the policy refuses it; what ``path_failure`` says when it
            cannot be resolved.
    """
    if not isinstance(path, str):
        msg = f"{argument} must be a path, as text; got {type(path).__name__}."
        raise ToolFailure("validation_error", msg)
    try:
        return policy.check(path, action)
    except (OSError, ValueError) as e:
        raise path_failure(e, "reach", path) from e


def governed(reason: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """The ``@tool`` of a filesystem tool: dangerous, high risk, approved call by call (D4)."""
    return tool(
        capability="filesystem", risk_level="high", requires_approval=True, approval_reason=reason
    )


_READ_REASON = "Reading local files can expose secrets or private data."
_LIST_REASON = "Listing local directories can reveal private file names and layout."
_SEARCH_REASON = "Searching local file contents can expose secrets or private data."


# --- read_file ---------------------------------------------------------------------------------


@governed(_READ_REASON)
def read_file(
    path: str,
    offset: Annotated[int, Range(0)] = 0,
    max_lines: Annotated[int, Range(1, _MAX_LINES)] = _DEFAULT_MAX_LINES,
) -> ToolResult:
    """Read a text file, window by window.

    A window holds up to ``max_lines`` lines, and at most 100,000 characters, ending on a line;
    its footer gives the offset that reads on, and the file's size in characters once fewer than
    100 million are left. Reaching an offset reads the file up to it, so a deep one in a huge
    file takes longer.

    Args:
        path: Path to the file (absolute or relative to cwd).
        offset: Where to start, in characters from the start of the file; the footer and
            ``search_files`` give offsets.
        max_lines: How many lines to return.

    Raises:
        ToolFailure: not_found when ``path`` does not exist; validation_error when it is not a
            regular file or is malformed; upstream when the OS refuses or fails the read.
    """
    try:
        p = Path(path).expanduser()
        if not _is_regular_file(p):
            raise ToolFailure("validation_error", _not_a_file(path))
        # newline="": the offsets are those of the file's own characters, line ends included.
        with p.open(encoding="utf-8", errors="replace", newline="") as handle:
            return _file_window(handle, offset, max_lines).result()
    except (OSError, ValueError) as e:
        raise path_failure(e, "read", path) from e


def _not_a_file(path: str) -> str:
    return f"{path!r} is not a regular file; list_directory lists a directory."


def _file_window(handle: TextIO, offset: int, max_lines: int) -> Window:
    """The window of the file ``handle`` reads, from ``offset``, and the file's size when what
    is left after it is within ``_COUNT_CHARS`` (counted without holding it)."""
    start = _skip(handle, offset)
    piece = handle.read(_WINDOW_CHARS + 1)
    rest = _skip(handle, _COUNT_CHARS)
    total = start + len(piece) + rest if not handle.read(1) else None
    end = start + _window_end(piece, max_lines)
    return Window(
        body=piece[: end - start],
        unit="chars",
        first=start,
        last=end,
        total=total,
        next_call={"offset": end} if total is None or end < total else None,
    )


def _skip(handle: TextIO, count: int) -> int:
    """Read past ``count`` characters, a chunk at a time; how many there were (fewer when the
    file ends first).

    A character offset has no byte position in UTF-8 without decoding what comes before it, so
    reaching one costs the file up to it: text decodes at about 3 GB/s, bytes that are not UTF-8
    at about 0.12 GB/s. A window 10 GB deep in such a file takes about 80 s, and paging through
    all of it costs the square of its size; the windows near the start stay cheap.
    """
    skipped = 0
    while skipped < count and (chunk := handle.read(min(count - skipped, _CHUNK))):
        skipped += len(chunk)
    return skipped


def _window_end(piece: str, max_lines: int) -> int:
    """Where a window that starts ``piece`` ends: after its ``max_lines``-th line, or within
    ``_WINDOW_CHARS`` characters (on a line, as the window cuts) when they come first."""
    end = 0
    for _ in range(max_lines):
        brk = piece.find("\n", end, _WINDOW_CHARS)
        if brk < 0:
            return line_cut(piece, 0, _WINDOW_CHARS)
        end = brk + 1
    return end


# --- list_directory ----------------------------------------------------------------------------


@governed(_LIST_REASON)
def list_directory(
    path: str = ".",
    pattern: str = "*",
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """List files and directories with sizes and types, a page of 1000 at a time, by name.

    Args:
        path: Directory path.
        pattern: Glob pattern to filter entries, e.g. "*.py", "*.md".
        offset: How many entries to skip; the footer gives the next offset.

    Raises:
        ToolFailure: not_found when ``path`` does not exist; validation_error when it is not a
            directory, is malformed, or ``pattern`` is unusable or matches too much; upstream
            when the OS refuses or fails the listing.
    """
    return _listing(Path(path).expanduser(), path, pattern, offset, shows=_anything)


def _anything(_entry: Path) -> bool:
    return True


def _listing(
    p: Path, path: str, pattern: str, offset: int, *, shows: Callable[[Path], bool]
) -> ToolResult:
    """The page from ``offset`` of the entries of ``p`` that ``pattern`` matches and ``shows``
    lets through."""
    try:
        is_directory = _is_directory(p)
        if is_directory:
            with os.scandir(p):  # a folder this process cannot read fails here, not as empty
                pass
    except (OSError, ValueError) as e:
        raise path_failure(e, "list", path) from e
    if not is_directory:
        raise ToolFailure("validation_error", _not_a_directory(path))
    entries = [entry for entry in _entries(p, path, pattern) if shows(entry)]
    if not entries:
        return ToolResult.success(f"No entries matching {pattern!r} in {path}")
    page = entries[offset : offset + _PAGE_ENTRIES]
    end = offset + len(page)
    window = list_window(
        [_entry_line(entry) for entry in page],
        first=offset + 1,
        total=len(entries),
        next_call={"offset": end} if end < len(entries) else None,
    )
    return window.result(heading=f"{path} ({len(entries)} entries):")


def _entries(p: Path, path: str, pattern: str) -> list[Path]:
    """The entries ``pattern`` matches in ``p``, sorted, whose folder is inside ``p``.

    A pattern may climb out (``../*``) or go through a link (``link/*``): only entries whose
    folder is inside ``p`` are listed (G-27). The walk stops past ``_MAX_WALK`` matches, inside
    or not, so such a pattern walks no further than any other.

    Raises:
        ToolFailure: validation_error for an unusable pattern, or one that matches more than
            the walk takes; what ``path_failure`` says for an OS error.
    """
    try:
        found = list(itertools.islice(p.glob(pattern), _MAX_WALK + 1))
        inside = [] if len(found) > _MAX_WALK else _inside(found, p.resolve())
    except OSError as e:
        raise path_failure(e, "list", path) from e
    except (ValueError, NotImplementedError) as e:
        msg = f"invalid pattern {pattern!r}: {e}; use a relative glob such as '*.py'."
        raise ToolFailure("validation_error", msg) from e
    if len(found) > _MAX_WALK:
        raise ToolFailure(
            "validation_error",
            f"{pattern!r} matches more than {_MAX_WALK} entries in {path!r}; narrow it, e.g. "
            "'*.py' or 'sub/*'.",
        )
    return sorted(inside)


def _inside(found: Iterable[Path], base: Path) -> list[Path]:
    """The entries whose folder resolves inside ``base`` (each folder resolved once)."""
    folders: dict[Path, bool] = {}
    inside: list[Path] = []
    for entry in found:
        if entry.parent not in folders:
            folders[entry.parent] = entry.parent.resolve().is_relative_to(base)
        if folders[entry.parent]:
            inside.append(entry)
    return inside


def _not_a_directory(path: str) -> str:
    return f"{path!r} is not a directory; read_file reads a file."


def _entry_line(entry: Path) -> str:
    try:
        if entry.is_dir():
            return f"  [dir]  {entry.name}/"
        return f"  {_human_size(entry.stat().st_size):>8s}  {entry.name}"
    except OSError:
        return f"  [denied] {entry.name}"


def _human_size(size: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            if unit == "B":
                return f"{size} {unit}"
            return f"{size:.1f} {unit}"
        size = int(size / 1024)
    return f"{size:.1f} TB"


# --- search_files ------------------------------------------------------------------------------


@governed(_SEARCH_REASON)
def search_files(
    directory: str,
    pattern: str,
    max_results: Annotated[int, Range(1, _MAX_RESULTS)] = _DEFAULT_MAX_RESULTS,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search the text files under a folder for the lines that contain a text (like grep -rn).

    Each line comes as ``path:line:offset: text``: the offset is where the text shown starts, in
    characters, and ``read_file(path, offset=...)`` reads from there. A line longer than 300
    characters shows the part around the match. Binary files, and links that point out of the
    folder, are skipped.

    Args:
        directory: Root directory to search in.
        pattern: The text to look for, ignoring case (a plain substring, not a regex).
        max_results: How many matching lines to return.
        offset: How many matching lines to skip; the footer gives the next offset.

    Raises:
        ToolFailure: not_found when ``directory`` does not exist; validation_error when it is
            not a directory or is malformed, or ``pattern`` is empty; upstream when the OS
            refuses or fails the search.
    """
    _check_pattern(pattern)
    scan = _Scan(Path(directory).expanduser(), Path(directory), _open_binary)
    return _search(scan, directory, pattern, max_results, offset)


def _check_pattern(pattern: str) -> None:
    if not pattern:
        raise ToolFailure("validation_error", "pattern cannot be empty; give the text to find.")


def _open_binary(path: Path) -> io.BufferedReader:
    return path.open("rb")


def _search(
    scan: _Scan, directory: str, pattern: str, max_results: int, offset: int
) -> ToolResult:
    """The page from ``offset`` of the lines under the scan's root (``directory`` as given)
    that contain ``pattern``."""
    root = scan.root
    try:
        if not _is_directory(root):
            raise ToolFailure("validation_error", _not_a_directory(directory))
        hits = _hits(root, pattern, scan)
        # Only the page is kept: the hits before it are counted, not held.
        skipped = sum(1 for _ in itertools.islice(hits, offset))
        found = [line.render(path) for path, line in itertools.islice(hits, max_results + 1)]
    except (OSError, ValueError) as e:
        raise path_failure(e, "search", directory) from e
    return _search_answer(
        directory,
        pattern,
        scan,
        offset,
        skipped,
        found[:max_results],
        more=len(found) > max_results,
    )


def _search_answer(
    directory: str,
    pattern: str,
    scan: _Scan,
    offset: int,
    skipped: int,
    page: list[str],
    *,
    more: bool,
) -> ToolResult:
    """A search's answer: its page of lines, with the window's footer, or what the search found
    before its scan budget ran out, and where it stopped."""
    unread = scan.unread_note()
    if not skipped and not page:
        if scan.stopped_in is None:
            return ToolResult.success(f"No matches for {pattern!r} in {directory}{unread}")
        return ToolResult.success(
            f"No matches for {pattern!r} in {directory} up to the scan limit ({scan.limit()}), "
            f"which stopped in {scan.stopped_in}; {_NARROWER}{unread}"
        )
    heading = f"Lines in {directory} that contain {pattern!r} (path:line:offset: text){unread}:"
    if scan.stopped_in is None:
        total = None if more else skipped + len(page)
        window = list_window(
            page,
            first=offset + 1,
            total=total,
            next_call={"offset": offset + len(page)} if more else None,
        )
        return window.result(heading=heading)
    # The scan stopped before the page was full: no call reads on, and no total is known.
    shown = (
        f"results {offset + 1}-{offset + len(page)}" if page else f"no results from {offset + 1}"
    )
    footer = (
        f"[{shown} | stopped at the scan limit ({scan.limit()}) in {scan.stopped_in}; {_NARROWER}]"
    )
    window = {
        "unit": "results",
        "first": offset + 1,
        "last": offset + len(page),
        "total": None,
        "next_call": None,
    }
    text = "\n".join([heading, *page, footer])
    return ToolResult.success(text, metadata={"window": window})


class _Scan:
    """How a search opens its files (``opener``) and how far it went: the characters and the
    files it read, the file where its budget ran out, and the folders and files it could not
    read. Paths are shown as the search's ``directory`` names them, so ``read_file`` takes each
    as it is."""

    __slots__ = ("_base", "chars", "files", "opener", "root", "stopped_in", "unread")

    def __init__(
        self, root: Path, base: Path, opener: Callable[[Path], io.BufferedReader]
    ) -> None:
        self.root, self._base, self.opener = root, base, opener
        self.chars = 0
        self.files = 0
        self.stopped_in: Path | None = None
        self.unread: list[Path] = []

    def shown(self, path: Path) -> Path:
        """``path`` (under the root) as the search shows it: under ``directory``, as given."""
        return self._base / path.relative_to(self.root)

    def limit(self) -> str:
        """The budget, and how many files it went to."""
        return f"{_SCAN_CHARS} characters in {self.files} file{'' if self.files == 1 else 's'}"

    def unread_note(self) -> str:
        """What the search could not read, or ``""``."""
        if not self.unread:
            return ""
        named = ", ".join(map(str, self.unread[:_UNREAD_NAMED]))
        more = len(self.unread) - _UNREAD_NAMED
        return f"; this process cannot read {named}" + (f" and {more} more" if more > 0 else "")


def _hits(root: Path, pattern: str, scan: _Scan) -> Iterator[tuple[Path, _Line]]:
    """Each line under ``root`` that contains ``pattern``, with its file's path as shown, until
    the scan budget runs out."""
    finder = re.compile(re.escape(pattern), re.IGNORECASE)
    base = root.resolve()
    for path in _files(root, scan):
        try:
            if path.suffix in _BINARY_SUFFIXES or not path.is_file():
                continue
            if path.is_symlink() and not path.resolve().is_relative_to(base):  # out (G-27)
                continue
        except OSError:
            scan.unread.append(scan.shown(path))
            continue
        shown = scan.shown(path)
        for line in _matching_lines(path, shown, finder, len(pattern), scan):
            yield shown, line
        if scan.stopped_in is not None:
            return


def _files(root: Path, scan: _Scan) -> Iterator[Path]:
    """The files under ``root``, folder by folder, each folder's in name order (so a page holds
    the same lines on every call). A link to a folder is not followed.

    Raises:
        OSError: ``root`` itself cannot be read; a subfolder that cannot is noted in ``scan``.
    """

    def unreadable(error: OSError) -> None:
        folder = Path(error.filename) if error.filename is not None else root
        if folder == root:
            raise error
        scan.unread.append(scan.shown(folder))

    for folder, folders, names in os.walk(root, onerror=unreadable):
        folders.sort()
        for name in sorted(names):
            yield Path(folder, name)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Line:
    """A line that contains a match: its number, the text shown and the offset it starts at,
    the line's length, and whether the text is only the part around the match."""

    number: int
    offset: int
    text: str
    length: int
    part: bool

    def render(self, name: Path) -> str:
        shown = f"{name}:{self.number}:{self.offset}: {self.text}"
        if not self.part:
            return shown
        reads_on = f"read_file(offset={self.offset}) reads on"
        return f"{shown} [part of a {self.length}-char line; {reads_on}]"


def _matching_lines(
    path: Path, shown: Path, finder: re.Pattern[str], width: int, scan: _Scan
) -> Iterator[_Line]:
    """The lines of ``path`` that contain a match: none when its first chunk has a NUL (a
    binary file), none from where it stops being UTF-8 text, and none past the scan budget (the
    file is then where the search stopped). A file that cannot be read is noted in ``scan``."""
    try:
        with scan.opener(path) as raw:
            if b"\0" in raw.peek(_SNIFF)[:_SNIFF]:
                return
            scan.files += 1
            with io.TextIOWrapper(raw, encoding="utf-8", errors="strict", newline="\n") as text:
                yield from _scan(text, finder, width, scan)
    except UnicodeDecodeError:
        return
    except OSError:
        scan.unread.append(shown)
        return
    if scan.chars > _SCAN_CHARS:
        scan.stopped_in = shown


def _scan(handle: TextIO, finder: re.Pattern[str], width: int, scan: _Scan) -> Iterator[_Line]:
    """Each line of ``handle`` that contains a match, once, until ``scan`` has read its budget.

    A line is read in pieces of at most ``_PIECE`` characters, so one of any length is searched
    in bounded memory; the last ``width - 1`` characters of a piece go with the next one, so a
    match across two pieces is found.
    """
    number, start, read, carried, last = 1, 0, 0, "", ""
    first: tuple[int, str] | None = None
    while piece := handle.readline(_PIECE):
        scan.chars += len(piece)
        if scan.chars > _SCAN_CHARS:
            return
        if first is None:
            first = _around(carried + piece, finder, read - len(carried))
        read, last = read + len(piece), piece
        if piece.endswith("\n"):
            if first is not None:
                yield _line(number, start, read, piece, first)
            number, start, first, carried = number + 1, read, None, ""
        elif first is None:
            carried = (carried + piece)[-(width - 1) :] if width > 1 else ""
    if first is not None:
        yield _line(number, start, read, last, first)


def _around(text: str, finder: re.Pattern[str], at: int) -> tuple[int, str] | None:
    """Where the part of ``text`` around its first match starts (``text`` starts at ``at``), and
    the part; ``None`` without a match."""
    match = finder.search(text)
    if match is None:
        return None
    start = match.start() - _BEFORE if match.start() > _BEFORE else 0
    return at + start, text[start : start + _EXCERPT].rstrip("\r\n")


def _line(number: int, start: int, end: int, piece: str, first: tuple[int, str]) -> _Line:
    """The line from ``start`` to ``end``, whose last piece is ``piece``: whole when it is short
    and that piece is all of it, else the part around its first match."""
    line = piece.rstrip("\r\n")
    length = end - start - (len(piece) - len(line))
    if length > _EXCERPT or end - len(piece) != start:
        return _Line(number=number, offset=first[0], text=first[1], length=length, part=True)
    indent = len(line) - len(line.lstrip())
    return _Line(
        number=number, offset=start + indent, text=line.strip(), length=length, part=False
    )


# --- Bound to a policy (C07) -------------------------------------------------------------------


def bound_reads(policy: FilesystemPolicy) -> tuple[Callable[..., ToolResult], ...]:
    """``read_file``, ``list_directory`` and ``search_files``, bound to ``policy``'s read roots.

    Each keeps its name, schema, docstring and governance. A relative path starts from the
    policy's ``cwd``; each path is checked right before it is used, and the answer names it
    canonical. A file is opened from its root down, never through a link. A listing refuses a
    pattern that climbs out with ``..`` and leaves out an entry whose target is outside the read
    roots; a search reads no file outside them.
    """

    def read(path: str, offset: int = 0, max_lines: int = _DEFAULT_MAX_LINES) -> ToolResult:
        canonical = checked(policy, path, "read")
        try:
            raw = _open_within(policy, canonical)
            with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as handle:
                return _file_window(handle, offset, max_lines).result()
        except _NotRegular as e:
            raise ToolFailure("validation_error", _not_a_file(str(canonical))) from e
        except (OSError, ValueError) as e:
            raise path_failure(e, "read", str(canonical)) from e

    def listing(path: str = ".", pattern: str = "*", offset: int = 0) -> ToolResult:
        if ".." in pattern.replace("\\", "/").split("/"):
            msg = (
                f"pattern {pattern!r} climbs out of the folder with '..', which this policy "
                "does not follow; list the other folder by its own path."
            )
            raise ToolFailure("permission_denied", msg)
        canonical = checked(policy, path, "read")
        shows = functools.partial(_within, policy)
        return _listing(canonical, str(canonical), pattern, offset, shows=shows)

    def search(
        directory: str, pattern: str, max_results: int = _DEFAULT_MAX_RESULTS, offset: int = 0
    ) -> ToolResult:
        _check_pattern(pattern)
        canonical = checked(policy, directory, "read", argument="directory")
        scan = _Scan(canonical, canonical, functools.partial(_found_within, policy))
        return _search(scan, str(canonical), pattern, max_results, offset)

    return (
        _like(read_file, _READ_REASON, read),
        _like(list_directory, _LIST_REASON, listing),
        _like(search_files, _SEARCH_REASON, search),
    )


def _like(
    original: Callable[..., ToolResult], reason: str, body: Callable[..., ToolResult]
) -> Callable[..., ToolResult]:
    """``body`` as a tool with ``original``'s name, signature, docstring and governance."""
    return governed(reason)(functools.wraps(original)(body))


class _NotRegular(OSError):
    """What a bound read opened is not a regular file (a folder, a pipe, a device)."""


def _open_within(policy: FilesystemPolicy, canonical: Path) -> io.BufferedReader:
    """The regular file at ``canonical``, opened from its read root down.

    Raises:
        _NotRegular: It is not a regular file.
        FilesystemPolicyError: A link took the place of a folder, or of the file.
        OSError: It cannot be opened.
    """
    descriptor = open_beneath(policy, canonical, "read", os.O_RDONLY)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _NotRegular(errno.EISDIR, "not a regular file", str(canonical))
        return open(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _found_within(policy: FilesystemPolicy, path: Path) -> io.BufferedReader:
    """The file a search found at ``path`` (a link resolved), if inside the read roots."""
    return _open_within(policy, policy.check(path, "read"))


def _within(policy: FilesystemPolicy, entry: Path) -> bool:
    """Whether the target of ``entry`` is inside the read roots."""
    try:
        policy.check(entry, "read")
    except (OSError, ValueError):
        return False
    return True
