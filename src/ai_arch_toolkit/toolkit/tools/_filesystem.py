"""Filesystem tools — read files, list directories, search content (T09).

Every long answer goes through the window (D39): a file is read window by window, by characters
from its start; a listing and a search come a page at a time. A search gives each line's offset,
which ``read_file`` takes, so a match found is a call away from the text around it. Files are read
a chunk at a time, so neither a huge file nor one long line ever lands in memory whole.
"""

from __future__ import annotations

import errno
import itertools
import os
import re
import stat
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TextIO

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure, line_cut
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

    A path that does not exist is not_found; a malformed one (a null byte, a name too long) is a
    validation_error; a permission refusal or any other OS error is upstream.
    """
    if isinstance(error, FileNotFoundError | NotADirectoryError):
        msg = f"no such file or directory: {path!r}; list_directory shows what is there."
        return ToolFailure("not_found", msg)
    if isinstance(error, PermissionError):
        msg = f"permission denied to {action} {path!r}; pick a path this process can read."
        return ToolFailure("upstream", msg)
    if isinstance(error, ValueError) or error.errno in _BAD_PATH_ERRNOS:
        msg = f"cannot {action} {path!r}: {_error_text(error)}; check the path."
        return ToolFailure("validation_error", msg)
    return ToolFailure("upstream", f"cannot {action} {path!r}: {_error_text(error)}.")


# --- read_file ---------------------------------------------------------------------------------


@tool(
    capability="filesystem",
    risk_level="high",
    requires_approval=True,
    approval_reason="Reading local files can expose secrets or private data.",
)
def read_file(
    path: str,
    offset: Annotated[int, Range(0)] = 0,
    max_lines: Annotated[int, Range(1, _MAX_LINES)] = _DEFAULT_MAX_LINES,
) -> ToolResult:
    """Read a text file, window by window.

    A window holds up to ``max_lines`` lines, and at most 100,000 characters, ending on a line;
    its footer gives the offset that reads on, and the file's size in characters once fewer than
    100 million are left.

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
        return _file_window(Path(path).expanduser(), path, offset, max_lines).result()
    except (OSError, ValueError) as e:
        raise path_failure(e, "read", path) from e


def _file_window(p: Path, path: str, offset: int, max_lines: int) -> Window:
    """The window of ``p`` from ``offset``, and the file's size when what is left after it is
    within ``_COUNT_CHARS`` (counted without holding it)."""
    if not _is_regular_file(p):
        msg = f"{path!r} is not a regular file; list_directory lists a directory."
        raise ToolFailure("validation_error", msg)
    # newline="": the offsets are those of the file's own characters, line ends included.
    with p.open(encoding="utf-8", errors="replace", newline="") as handle:
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
    file ends first)."""
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


@tool(
    capability="filesystem",
    risk_level="high",
    requires_approval=True,
    approval_reason="Listing local directories can reveal private file names and layout.",
)
def list_directory(
    path: str = ".",
    pattern: str = "*",
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """List files and directories with sizes and types, a page of 1000 at a time, by name.

    Args:
        path: Directory path. Defaults to current directory.
        pattern: Glob pattern to filter entries, e.g. "*.py", "*.md". Defaults to all.
        offset: How many entries to skip; the footer gives the next offset.

    Raises:
        ToolFailure: not_found when ``path`` does not exist; validation_error when it is not a
            directory, is malformed, or ``pattern`` is unusable or matches too much; upstream
            when the OS refuses or fails the listing.
    """
    p = Path(path).expanduser()
    try:
        is_directory = _is_directory(p)
    except (OSError, ValueError) as e:
        raise path_failure(e, "list", path) from e
    if not is_directory:
        raise ToolFailure("validation_error", _not_a_directory(path))
    entries = _entries(p, path, pattern)
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


@tool(
    capability="filesystem",
    risk_level="high",
    requires_approval=True,
    approval_reason="Searching local file contents can expose secrets or private data.",
)
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
    if not pattern:
        raise ToolFailure("validation_error", "pattern cannot be empty; give the text to find.")
    root = Path(directory).expanduser()
    try:
        if not _is_directory(root):
            raise ToolFailure("validation_error", _not_a_directory(directory))
        found = _first(_hits(root, pattern), offset + max_results + 1)
    except (OSError, ValueError) as e:
        raise path_failure(e, "search", directory) from e
    if not found:
        return ToolResult.success(f"No matches for {pattern!r} in {directory}")
    page = found[offset : offset + max_results]
    more = len(found) > offset + max_results
    window = list_window(
        page,
        first=offset + 1,
        total=None if more else len(found),
        next_call={"offset": offset + len(page)} if more else None,
    )
    heading = f"Lines in {directory} that contain {pattern!r} (path:line:offset: text):"
    return window.result(heading=heading)


def _first(hits: Iterator[str], count: int) -> list[str]:
    """The first ``count`` of ``hits``: the search stops there."""
    found: list[str] = []
    for hit in hits:
        found.append(hit)
        if len(found) >= count:
            break
    return found


def _hits(root: Path, pattern: str) -> Iterator[str]:
    """Each line under ``root`` that contains ``pattern``, as ``path:line:offset: text``."""
    finder = re.compile(re.escape(pattern), re.IGNORECASE)
    base = root.resolve()
    for path in _files(root):
        if path.suffix in _BINARY_SUFFIXES or not path.is_file():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(base):  # out (G-27)
            continue
        name = path.relative_to(root)
        for line in _matching_lines(path, finder, len(pattern)):
            yield line.render(name)


def _files(root: Path) -> Iterator[Path]:
    """The files under ``root``, folder by folder, each folder's in name order (so a page holds
    the same lines on every call). A link to a folder is not followed."""
    for folder, folders, names in os.walk(root):
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


def _matching_lines(path: Path, finder: re.Pattern[str], width: int) -> Iterator[_Line]:
    """The lines of ``path`` that contain a match; none from where it stops being UTF-8 text (a
    binary file) or cannot be read."""
    try:
        with path.open(encoding="utf-8", errors="strict", newline="\n") as handle:
            yield from _scan(handle, finder, width)
    except (UnicodeDecodeError, OSError):
        return


def _scan(handle: TextIO, finder: re.Pattern[str], width: int) -> Iterator[_Line]:
    """Each line of ``handle`` that contains a match, once.

    A line is read in pieces of at most ``_PIECE`` characters, so one of any length is searched
    in bounded memory; the last ``width - 1`` characters of a piece go with the next one, so a
    match across two pieces is found.
    """
    number, start, read, carried, last = 1, 0, 0, "", ""
    first: tuple[int, str] | None = None
    while piece := handle.readline(_PIECE):
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
