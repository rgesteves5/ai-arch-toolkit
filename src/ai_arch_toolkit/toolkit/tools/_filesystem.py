"""Filesystem tools — read files, list directories, search content."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.core import tool

_DEFAULT_MAX_LINES = 200
_DEFAULT_MAX_RESULTS = 50


@tool
def read_file(path: str, max_lines: int = _DEFAULT_MAX_LINES) -> str:
    """Read a file and return its contents.

    Args:
        path: Path to the file (absolute or relative to cwd).
        max_lines: Maximum number of lines to return. Defaults to 200.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Permission denied: {path}"

    lines = text.splitlines()
    if len(lines) > max_lines:
        truncated = "\n".join(lines[:max_lines])
        return truncated + f"\n\n[Truncated — {len(lines)} total lines]"
    return text


@tool
def list_directory(path: str = ".", pattern: str = "*") -> str:
    """List files and directories with sizes and types.

    Args:
        path: Directory path. Defaults to current directory.
        pattern: Glob pattern to filter entries, e.g. "*.py", "*.md". Defaults to all.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"Directory not found: {path}"
    if not p.is_dir():
        return f"Not a directory: {path}"

    try:
        entries = sorted(p.glob(pattern))
    except PermissionError:
        return f"Permission denied: {path}"

    if not entries:
        return f"No entries matching {pattern!r} in {path}"

    lines: list[str] = []
    for entry in entries:
        try:
            stat = entry.stat()
            if entry.is_dir():
                lines.append(f"  [dir]  {entry.name}/")
            else:
                size = _human_size(stat.st_size)
                lines.append(f"  {size:>8s}  {entry.name}")
        except PermissionError:
            lines.append(f"  [denied] {entry.name}")

    header = f"{path} ({len(entries)} entries):"
    return header + "\n" + "\n".join(lines)


@tool
def search_files(directory: str, pattern: str, max_results: int = _DEFAULT_MAX_RESULTS) -> str:
    """Search for text in files recursively (like grep -r).

    Args:
        directory: Root directory to search in.
        pattern: Text pattern to search for (case-insensitive substring match).
        max_results: Maximum number of matching lines to return. Defaults to 50.
    """
    root = Path(directory).expanduser()
    if not root.exists():
        return f"Directory not found: {directory}"
    if not root.is_dir():
        return f"Not a directory: {directory}"

    lower_pattern = pattern.lower()
    matches: list[str] = []

    for filepath in root.rglob("*"):
        if not filepath.is_file():
            continue
        # Skip binary-looking files
        if filepath.suffix in {".pyc", ".pyo", ".so", ".dylib", ".exe", ".bin", ".gz", ".zip"}:
            continue
        try:
            text = filepath.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, PermissionError):
            continue

        for line_num, line in enumerate(text.splitlines(), 1):
            if lower_pattern in line.lower():
                rel = filepath.relative_to(root)
                matches.append(f"{rel}:{line_num}: {line.strip()}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n\n[Stopped at {max_results} results]"

    if not matches:
        return f"No matches for {pattern!r} in {directory}"
    return "\n".join(matches)


def _human_size(size: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            if unit == "B":
                return f"{size} {unit}"
            return f"{size:.1f} {unit}"
        size = int(size / 1024)
    return f"{size:.1f} TB"
