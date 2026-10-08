"""Text processing tools — regex, statistics, encoding (T09)."""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Annotated

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._window import list_window

# A regex match runs in C holding the GIL: no timeout can stop it in this process, not even the
# executor's. A static check cannot bound it either: two quantifiers over the same characters
# backtrack polynomially (a*a*b takes n³ steps, about 18 minutes on 20,000 characters), and a
# check strict enough to catch every such shape would refuse ordinary patterns such as \d+-\d+.
# So the match runs in a child Python process (-I -S: stdlib only, no site), which is killed
# past _MATCH_S seconds while this process waits without the GIL. That bounds every pattern: the
# worst case allowed is 5 s of a child's CPU, and a match costs the child's start, about 25 ms.
# The heaviest pattern with one quantifier measured on 20,000 characters, [a-z]+\d on "a" * 20000
# (n² steps), takes 1.8 s, well within. The guards below still refuse at once what is too long,
# back-references, and the shapes that backtrack exponentially.
_MATCH_S = 5.0
_WORKER = """
import json, re, sys
job = json.load(sys.stdin)
first, last, total, spans = job["offset"], job["offset"] + job["count"], 0, []
for match in re.finditer(job["pattern"], job["text"]):
    if first <= total < last:
        spans.append(match.regs)
    total += 1
json.dump({"total": total, "spans": spans}, sys.stdout)
"""
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
    offset of the next page. The match runs in a separate process, given 5 seconds.

    Args:
        text: The text to search in (up to 20000 characters).
        pattern: A regular expression pattern (up to 500 characters). Back-references and groups
            that repeat while holding a quantifier or an alternation are refused.
        offset: How many matches to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the text or the pattern is too long, the pattern is
            not a valid regex, its shape can backtrack exponentially, or matching it takes
            longer than 5 seconds; upstream when the process that matches cannot run.
    """
    _check(text, pattern)
    total, spans = _matches(text, pattern, offset)
    if not total:
        return ToolResult.success(f"No matches for {pattern!r}.")
    lines = _page(text, spans)
    end = offset + len(lines)
    window = list_window(
        lines,
        first=offset + 1,
        total=total,
        next_call={"offset": end} if end < total else None,
    )
    return window.result(heading=f"{total} match(es) for {pattern!r}:")


type _Spans = Sequence[Sequence[Sequence[int]]]


def _matches(text: str, pattern: str, offset: int) -> tuple[int, _Spans]:
    """How many matches ``pattern`` has in ``text``, and the spans (the match's, then each
    group's; ``-1`` for a group that took no part) of up to ``_PAGE_MATCHES`` from ``offset``,
    found in a child process within ``_MATCH_S`` seconds.

    Raises:
        ToolFailure: validation_error when the match takes longer; upstream when the child
            cannot run.
    """
    job = {"pattern": pattern, "text": text, "offset": offset, "count": _PAGE_MATCHES}
    try:
        done = subprocess.run(
            [_interpreter(), "-I", "-S", "-c", _WORKER],
            input=json.dumps(job),  # ASCII: any character, a lone surrogate too, is escaped
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_MATCH_S,
            check=False,
        )
    except subprocess.TimeoutExpired as e:  # the child is killed
        raise ToolFailure(
            "validation_error",
            f"pattern refused: matching it on this text took longer than {_MATCH_S:g}s, so its "
            "quantifiers backtrack (as a*a*b does on a long run of a); rewrite it so that no two "
            "repetitions can match the same characters, or search a shorter text.",
        ) from e
    except OSError as e:
        raise ToolFailure("upstream", f"could not start the process that matches: {e}") from e
    if done.returncode != 0:
        reason = done.stderr.strip().rsplit("\n", 1)[-1]
        raise ToolFailure("upstream", f"the process that matches failed: {reason}")
    answer = json.loads(done.stdout)
    return answer["total"], answer["spans"]


def _interpreter() -> str:
    """The Python that runs the match: this one.

    Raises:
        ToolFailure: upstream when this program has no Python to start (an embedded or frozen
            one, whose ``sys.executable`` is not a Python).
    """
    if not sys.executable or getattr(sys, "frozen", False):
        raise ToolFailure(
            "upstream",
            "regex_search matches in a child Python process, and this program has no Python "
            "interpreter to start (sys.executable), so it cannot match here.",
        )
    return sys.executable


def _page(text: str, spans: _Spans) -> list[str]:
    """The lines of the matches at ``spans``: up to ``_PAGE_MATCHES``, within ``_PAGE_CHARS``
    characters (the first one whatever its length)."""
    lines: list[str] = []
    used = 0
    for (start, end), *regs in spans:
        groups = tuple(None if low < 0 else text[low:high] for low, high in regs)
        line = f"  [{start}:{end}] {text[start:end]!r}"
        line += f" groups={groups}" if groups else ""
        if lines and used + len(line) > _PAGE_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return lines


def _check(text: str, pattern: str) -> None:
    """Refuse ``pattern`` or ``text`` before a child starts on them.

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
        re.compile(pattern)
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
