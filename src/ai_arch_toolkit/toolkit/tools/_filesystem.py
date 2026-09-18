"""Filesystem tools — read files, list directories, search content."""

from __future__ import annotations

import itertools
from pathlib import Path

from ai_arch_toolkit.core import tool

_DEFAULT_MAX_LINES = 200
_DEFAULT_MAX_RESULTS = 50
_MAX_LINES = 10_000
_MAX_RESULTS = 1_000
_MAX_ENTRIES = 1_000
# Reads stop here, so a file on one long line or a huge log never lands in memory whole.
_MAX_READ_CHARS = 100_000
_MAX_SCAN_CHARS = 1_000_000
_MAX_MATCH_CHARS = 300
_BINARY_SUFFIXES = frozenset({".pyc", ".pyo", ".so", ".dylib", ".exe", ".bin", ".gz", ".zip"})


def read_prefix(path: Path, max_chars: int, *, errors: str = "replace") -> tuple[str, bool]:
    """The first ``max_chars`` characters of a UTF-8 file, and whether that is all of it."""
    with path.open(encoding="utf-8", errors=errors) as handle:
        text = handle.read(max_chars + 1)
    return text[:max_chars], len(text) <= max_chars


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


@tool(
    capability="filesystem",
    risk_level="high",
    requires_approval=True,
    approval_reason="Reading local files can expose secrets or private data.",
)
def read_file(path: str, max_lines: int = _DEFAULT_MAX_LINES) -> str:
    """Read a file and return its contents.

    Args:
        path: Path to the file (absolute or relative to cwd).
        max_lines: Maximum number of lines to return (1-10000). Defaults to 200.
    """
    try:
        return _read_lines(Path(path).expanduser(), path, clamp(max_lines, 1, _MAX_LINES))
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as e:
        return f"Cannot read {path!r}: {e.strerror or e}"


def _read_lines(p: Path, path: str, max_lines: int) -> str:
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    text, whole = read_prefix(p, _MAX_READ_CHARS)
    lines = text.splitlines()
    if len(lines) > max_lines:
        total = f"{len(lines)} total lines" if whole else "the file is longer"
        return "\n".join(lines[:max_lines]) + f"\n\n[Truncated — {total}]"
    if not whole:
        return text + f"\n\n[Truncated — the file is longer than {_MAX_READ_CHARS} characters]"
    return text


@tool(
    capability="filesystem",
    risk_level="high",
    requires_approval=True,
    approval_reason="Listing local directories can reveal private file names and layout.",
)
def list_directory(path: str = ".", pattern: str = "*") -> str:
    """List files and directories with sizes and types.

    Args:
        path: Directory path. Defaults to current directory.
        pattern: Glob pattern to filter entries, e.g. "*.py", "*.md". Defaults to all.
    """
    p = Path(path).expanduser()
    try:
        if not p.exists():
            return f"Directory not found: {path}"
        if not p.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(itertools.islice(p.glob(pattern), _MAX_ENTRIES + 1))
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as e:
        return f"Cannot list {path!r}: {e.strerror or e}"
    except (ValueError, NotImplementedError) as e:
        return f"Invalid pattern {pattern!r}: {e}"
    if not entries:
        return f"No entries matching {pattern!r} in {path}"
    lines = [_entry_line(entry) for entry in entries[:_MAX_ENTRIES]]
    header = f"{path} ({len(lines)} entries):"
    if len(entries) > _MAX_ENTRIES:
        header = f"{path} (first {_MAX_ENTRIES} entries found):"
    return header + "\n" + "\n".join(lines)


def _entry_line(entry: Path) -> str:
    try:
        if entry.is_dir():
            return f"  [dir]  {entry.name}/"
        return f"  {_human_size(entry.stat().st_size):>8s}  {entry.name}"
    except OSError:
        return f"  [denied] {entry.name}"


@tool(
    capability="filesystem",
    risk_level="high",
    requires_approval=True,
    approval_reason="Searching local file contents can expose secrets or private data.",
)
def search_files(directory: str, pattern: str, max_results: int = _DEFAULT_MAX_RESULTS) -> str:
    """Search for text in files recursively (like grep -r).

    Args:
        directory: Root directory to search in.
        pattern: Text pattern to search for (case-insensitive substring match).
        max_results: Maximum number of matching lines to return (1-1000). Defaults to 50.
    """
    root = Path(directory).expanduser()
    max_results = clamp(max_results, 1, _MAX_RESULTS)
    try:
        if not root.exists():
            return f"Directory not found: {directory}"
        if not root.is_dir():
            return f"Not a directory: {directory}"
        matches = _matches(root, pattern.lower(), max_results)
    except OSError as e:
        return f"Cannot search {directory!r}: {e.strerror or e}"
    if not matches:
        return f"No matches for {pattern!r} in {directory}"
    if len(matches) >= max_results:
        return "\n".join(matches) + f"\n\n[Stopped at {max_results} results]"
    return "\n".join(matches)


def _matches(root: Path, needle: str, max_results: int) -> list[str]:
    matches: list[str] = []
    for filepath in root.rglob("*"):
        if filepath.suffix in _BINARY_SUFFIXES or not filepath.is_file():
            continue
        try:
            text, _ = read_prefix(filepath, _MAX_SCAN_CHARS, errors="strict")
        except (UnicodeDecodeError, OSError):  # binary or unreadable: skip it
            continue
        for line_num, line in enumerate(text.splitlines(), 1):
            if needle in line.lower():
                shown = line.strip()[:_MAX_MATCH_CHARS]
                matches.append(f"{filepath.relative_to(root)}:{line_num}: {shown}")
                if len(matches) >= max_results:
                    return matches
    return matches


def _human_size(size: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            if unit == "B":
                return f"{size} {unit}"
            return f"{size:.1f} {unit}"
        size = int(size / 1024)
    return f"{size:.1f} TB"
