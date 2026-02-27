"""Text processing tools — regex, statistics, encoding."""

from __future__ import annotations

import base64
import re

from ai_arch_toolkit.core import tool


@tool
def regex_search(text: str, pattern: str) -> str:
    """Find all regex matches in text.

    Returns each match on a separate line with its position.

    Args:
        text: The text to search in.
        pattern: A regular expression pattern.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = list(compiled.finditer(text))
    if not matches:
        return "No matches found."

    lines: list[str] = []
    for m in matches:
        groups = m.groups()
        if groups:
            lines.append(f"  [{m.start()}:{m.end()}] {m.group()!r} groups={groups}")
        else:
            lines.append(f"  [{m.start()}:{m.end()}] {m.group()!r}")

    return f"{len(matches)} match(es):\n" + "\n".join(lines)


@tool
def text_stats(text: str) -> str:
    """Count words, characters, lines, and sentences in text.

    Args:
        text: The text to analyze.
    """
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", "").replace("\t", ""))
    words = len(text.split())
    lines = text.count("\n") + (1 if text else 0)
    sentences = len(re.findall(r"[.!?]+(?:\s|$)", text))
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])

    return (
        f"Characters: {chars} ({chars_no_spaces} without spaces)\n"
        f"Words: {words}\n"
        f"Lines: {lines}\n"
        f"Sentences: {sentences}\n"
        f"Paragraphs: {paragraphs}"
    )


@tool
def base64_encode(text: str) -> str:
    """Encode text to base64.

    Args:
        text: The text to encode.
    """
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


@tool
def base64_decode(encoded: str) -> str:
    """Decode a base64 string to text.

    Args:
        encoded: The base64-encoded string to decode.
    """
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception as e:
        return f"Decode error: {e}"
