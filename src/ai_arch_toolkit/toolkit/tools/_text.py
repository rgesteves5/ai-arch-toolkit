"""Text processing tools — regex, statistics, encoding."""

from __future__ import annotations

import base64
import re

from ai_arch_toolkit.core import tool

# A regex match runs in C holding the GIL: no timeout can stop it, not even the executor's. The
# guards bound it before it starts. One unbounded quantifier over 20 000 characters stays under a
# second; a shape that backtracks exponentially is refused whatever its size.
_MAX_PATTERN_CHARS = 500
_MAX_TEXT_CHARS = 20_000
_MAX_MATCHES = 1000
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
def regex_search(text: str, pattern: str) -> str:
    """Find all regex matches in text.

    Returns each match on a separate line with its position.

    Args:
        text: The text to search in (up to 20000 characters).
        pattern: A regular expression pattern (up to 500 characters). Back-references and groups
            that repeat while holding a quantifier or an alternation are refused.
    """
    if len(text) > _MAX_TEXT_CHARS:
        return f"Text refused: longer than {_MAX_TEXT_CHARS} characters."
    if len(pattern) > _MAX_PATTERN_CHARS:
        return f"Pattern refused: longer than {_MAX_PATTERN_CHARS} characters."
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"
    risk = _backtracking_risk(pattern)
    if risk:
        return f"Pattern refused: {risk}, which can backtrack exponentially."

    lines: list[str] = []
    for m in compiled.finditer(text):
        if len(lines) == _MAX_MATCHES:
            return f"{_MAX_MATCHES} match(es) shown; more not shown:\n" + "\n".join(lines)
        groups = m.groups()
        suffix = f" groups={groups}" if groups else ""
        lines.append(f"  [{m.start()}:{m.end()}] {m.group()!r}{suffix}")
    if not lines:
        return "No matches found."
    return f"{len(lines)} match(es):\n" + "\n".join(lines)


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
    """
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception as e:
        return f"Decode error: {e}"
