"""Text processing tools — regex, statistics, encoding (T09)."""

from __future__ import annotations

import base64
import re
from collections.abc import Sequence
from typing import Annotated

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._window import list_window

# A regex match runs in C holding the GIL: no timeout can stop it, not even the executor's. The
# guards bound it before it starts. One unbounded quantifier over 20 000 characters stays under a
# second; a shape that backtracks exponentially is refused whatever its size.
_MAX_PATTERN_CHARS = 500
_MAX_TEXT_CHARS = 20_000
# A page of matches: at most this many, within this many characters (one match at least).
_PAGE_MATCHES = 1000
_PAGE_CHARS = 20_000
_QUANTIFIER = r"[*+?]|\{\d*(?:,\d*)?\}"
_TOKEN = re.compile(
    r"(?P<ref>\\[1-9]|\(\?P=|\(\?\()"
    r"|(?P<escape>\\.)"
    r"|(?P<klass>\[\^?\]?(?:\\.|[^\]\\])*\])"
    r"|(?P<skip>\(\?\#[^)]*\)|\(\?[aiLmsux]+\))"
    r"|(?P<open>\((?:\?(?:P<\w+>|[aiLmsux-]*:|<?[=!]|>))?)"
    rf"|(?P<close>\)(?:{_QUANTIFIER})?)"
    rf"|(?P<quantifier>{_QUANTIFIER})"
    r"|(?P<bar>\|)"
    r"|(?P<other>.)",
    re.DOTALL,
)


@tool(capability="compute")
def regex_search(text: str, pattern: str, offset: Annotated[int, Range(0)] = 0) -> ToolResult:
    """Find all regex matches in text, each on a line of its own with its position and groups.

    A page holds up to 1000 matches; the heading gives how many there are, and the footer the
    offset of the next page.

    Args:
        text: The text to search in (up to 20000 characters).
        pattern: A regular expression pattern (up to 500 characters). Back-references and groups
            that repeat while holding a quantifier or an alternation are refused.
        offset: How many matches to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the text or the pattern is too long, the pattern is
            not a valid regex, or its shape can backtrack exponentially.
    """
    matches = list(_compiled(text, pattern).finditer(text))
    if not matches:
        return ToolResult.success(f"No matches for {pattern!r}.")
    lines = _page(matches, offset)
    end = offset + len(lines)
    window = list_window(
        lines,
        first=offset + 1,
        total=len(matches),
        next_call={"offset": end} if end < len(matches) else None,
    )
    return window.result(heading=f"{len(matches)} match(es) for {pattern!r}:")


def _page(matches: Sequence[re.Match[str]], offset: int) -> list[str]:
    """The lines of the matches from ``offset``: up to ``_PAGE_MATCHES``, within ``_PAGE_CHARS``
    characters (the first one whatever its length)."""
    lines: list[str] = []
    used = 0
    for match in matches[offset : offset + _PAGE_MATCHES]:
        groups = match.groups()
        line = f"  [{match.start()}:{match.end()}] {match.group()!r}"
        line += f" groups={groups}" if groups else ""
        if lines and used + len(line) > _PAGE_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return lines


def _compiled(text: str, pattern: str) -> re.Pattern[str]:
    """``pattern``, compiled, once the guards let it run on ``text``.

    Raises:
        ToolFailure: validation_error when either is too long, the pattern is not a valid regex,
            or its shape can backtrack exponentially.
    """
    if len(text) > _MAX_TEXT_CHARS:
        raise ToolFailure(
            "validation_error",
            f"text refused: {len(text)} characters, longer than {_MAX_TEXT_CHARS}; search it in "
            "parts.",
        )
    if len(pattern) > _MAX_PATTERN_CHARS:
        raise ToolFailure(
            "validation_error",
            f"pattern refused: {len(pattern)} characters, longer than {_MAX_PATTERN_CHARS}; "
            "shorten it.",
        )
    try:
        compiled = re.compile(pattern)
    except (re.error, OverflowError, RecursionError) as e:  # a{4294967296}, deep nesting
        raise ToolFailure(
            "validation_error",
            f"invalid regex {pattern!r}: {e}; write it in Python's re syntax.",
        ) from e
    risk = _backtracking_risk(pattern)
    if risk:
        raise ToolFailure(
            "validation_error",
            f"pattern refused: {risk}, which can backtrack exponentially; rewrite it without "
            "nested repetition or back-references.",
        )
    return compiled


def _backtracking_risk(pattern: str) -> str:
    """Why a valid ``pattern`` could backtrack exponentially, or ``""`` when it cannot."""
    holds = [False]  # per open group: whether it holds a quantifier or an alternation
    for token in _TOKEN.finditer(pattern):
        kind, text = token.lastgroup, token.group()
        if kind == "ref":
            return "it uses a back-reference or a conditional"
        if kind == "open":
            holds.append(False)
        elif kind == "close" and len(holds) > 1:
            inner = holds.pop()
            if inner and _repeats(text[1:]):
                return "a repeated group holds a quantifier or an alternation"
            holds[-1] = holds[-1] or inner or len(text) > 1
        elif kind in ("quantifier", "bar"):
            holds[-1] = True
    return ""


def _repeats(quantifier: str) -> bool:
    """Whether a group's quantifier can take the group more than once."""
    if quantifier in ("", "?"):
        return False
    if quantifier in ("*", "+"):
        return True
    low, comma, high = quantifier[1:-1].partition(",")
    if not comma:
        return int(low or 0) > 1
    return not high or int(high) > 1


@tool(capability="compute")
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


@tool(capability="compute")
def base64_encode(text: str) -> str:
    """Encode text to base64.

    Args:
        text: The text to encode.
    """
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


@tool(capability="compute")
def base64_decode(encoded: str) -> str:
    """Decode a base64 string to text.

    Args:
        encoded: The base64-encoded string to decode.

    Raises:
        ToolFailure: validation_error when ``encoded`` is not base64.
    """
    try:
        decoded = base64.b64decode(encoded)
    except ValueError as e:  # binascii.Error, or a non-ASCII character
        raise ToolFailure("validation_error", f"not valid base64: {e}.") from e
    return decoded.decode("utf-8", errors="replace")
