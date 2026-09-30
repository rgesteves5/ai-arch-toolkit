"""The window every cut of a tool's output goes through (D39).

A tool that returns part of something longer — a page of a document, a page of results, the
passages that mention a term — builds a :class:`Window`. Its text ends with a footer that says what
was shown, the total when it is known, and the exact call that reads on, so no cut leaves the agent
at a dead end::

    [chars 0-4000 of 34651 | next: offset=4000]
    [results 21-40 of 1234 | next: offset=40]
    [matches 1-3 of 5 for "1960" | next: find="1960", offset=16500]

``Window.result()`` hands the same facts to the application in ``metadata["window"]``. Numbers
carry no separators, so the value in the footer is the one the model passes back.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ai_arch_toolkit.core._tools._result import ToolResult, line_cut

type Unit = Literal["chars", "results", "matches"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Window:
    """A part of a longer text or list, and how to read the rest.

    Attributes:
        body: The part shown.
        unit: What ``first``, ``last`` and ``total`` count.
        first: For ``chars``, the offset where the body starts; otherwise the 1-based number of the
            first result or match shown (0 when a search shows none).
        last: For ``chars``, the offset where the body ends; otherwise the number of the last
            result or match shown.
        total: How many there are in all, or ``None`` when the source does not say.
        next_call: The arguments of the call that reads on, or ``None`` when nothing is left.
        label: The term a search looked for.
    """

    body: str
    unit: Unit
    first: int
    last: int
    total: int | None
    next_call: Mapping[str, object] | None = None
    label: str = ""

    def footer(self) -> str:
        """The line that says what was shown and how to read on; empty when nothing is missing."""
        if self.unit == "matches":
            return _matches_footer(self)
        if not _partial(self):
            return ""
        if self.last < self.first:
            return f"[no {self.unit} from {self.first}{_of(self.total)} | end]"
        return f"[{self.unit} {self.first}-{self.last}{_of(self.total)} | {_onward(self)}]"

    def text(self) -> str:
        """The body, then the footer on a line of its own."""
        footer = self.footer()
        if not footer or not self.body:
            return footer or self.body
        return self.body + ("" if self.body.endswith("\n") else "\n") + footer

    def result(self) -> ToolResult:
        """A successful tool result with :meth:`text` and the window in ``metadata["window"]``."""
        next_call = dict(self.next_call) if self.next_call is not None else None
        window = {
            "unit": self.unit,
            "first": self.first,
            "last": self.last,
            "total": self.total,
            "next_call": next_call,
        }
        return ToolResult.success(self.text(), metadata={"window": window})


def text_window(text: str, *, offset: int = 0, limit: int) -> Window:
    """The part of ``text`` from ``offset``, at most ``limit`` characters, ending on a line.

    The next window starts exactly where this one ends, so following the footers rebuilds the
    text. An offset past the end shows nothing and says so.
    """
    start = min(max(offset, 0), len(text))
    end = line_cut(text, start, min(start + max(limit, 1), len(text)))
    next_call = {"offset": end} if end < len(text) else None
    return Window(
        body=text[start:end],
        unit="chars",
        first=start,
        last=end,
        total=len(text),
        next_call=next_call,
    )


def find_window(
    text: str, needle: str, *, offset: int = 0, limit: int, context: int = 400
) -> Window:
    """The passages of ``text`` that contain ``needle``, from ``offset``, within ``limit`` chars.

    Matching ignores case. A passage is the whole lines within ``context`` characters of a match,
    headed by its position (``[at char 13860]``); passages that touch merge into one. The first
    passage is always shown; the next call searches on from the end of the last one shown.
    """
    positions = [match.start() for match in re.finditer(re.escape(needle), text, re.IGNORECASE)]
    later = [position for position in positions if position >= offset] if needle else []
    shown = _within(text, _passages(text, later, len(needle), context), limit)
    if not shown:
        return Window(body="", unit="matches", first=0, last=0, total=len(positions), label=needle)
    first = positions.index(shown[0].matches[0]) + 1
    last = positions.index(shown[-1].matches[-1]) + 1
    next_call = {"find": needle, "offset": shown[-1].end} if last < len(positions) else None
    return Window(
        body="\n".join(passage.render(text) for passage in shown),
        unit="matches",
        first=first,
        last=last,
        total=len(positions),
        next_call=next_call,
        label=needle,
    )


def list_window(
    lines: Sequence[str],
    *,
    first: int = 1,
    total: int | None = None,
    next_call: Mapping[str, object] | None = None,
) -> Window:
    """A page of results the source already cut.

    Args:
        lines: The results of this page, one line each.
        first: The 1-based number of the first of them.
        total: The source's count of all results, when it gives one.
        next_call: The arguments that fetch the following page; ``None`` on the last.
    """
    return Window(
        body="\n".join(lines),
        unit="results",
        first=first,
        last=first + len(lines) - 1,
        total=total,
        next_call=next_call,
    )


def page_window(
    items: Sequence[str], *, offset: int = 0, limit: int, param: str = "offset"
) -> Window:
    """A page of a list the tool holds whole: ``limit`` items from ``offset``, with the total.

    ``param`` names the tool's parameter that the footer tells the model to set.
    """
    start = min(max(offset, 0), len(items))
    page = items[start : start + max(limit, 1)]
    end = start + len(page)
    next_call = {param: end} if end < len(items) else None
    return list_window(page, first=start + 1, total=len(items), next_call=next_call)


@dataclass(frozen=True, slots=True)
class _Passage:
    start: int
    end: int
    matches: tuple[int, ...]

    def render(self, text: str) -> str:
        return f"[at char {self.start}]\n{text[self.start : self.end]}"


def _passages(text: str, positions: Sequence[int], length: int, context: int) -> list[_Passage]:
    passages: list[_Passage] = []
    for position in positions:
        start, end = _around(text, position, position + length, context)
        if passages and start <= passages[-1].end + 1:
            joined = passages[-1]
            end = max(joined.end, end)
            passages[-1] = _Passage(joined.start, end, (*joined.matches, position))
        else:
            passages.append(_Passage(start, end, (position,)))
    return passages


def _around(text: str, start: int, end: int, context: int) -> tuple[int, int]:
    """The whole lines within ``context`` characters of ``text[start:end]``."""
    low = max(0, start - context)
    brk = text.find("\n", low, start)
    first = brk + 1 if brk >= 0 and low > 0 else low
    high = min(len(text), end + context)
    brk = text.rfind("\n", end, high)
    last = brk if brk >= 0 and high < len(text) else high
    return first, last


def _within(text: str, passages: Sequence[_Passage], limit: int) -> list[_Passage]:
    shown: list[_Passage] = []
    used = 0
    for passage in passages:
        size = len(passage.render(text)) + 1
        if shown and used + size > limit:
            break
        shown.append(passage)
        used += size
    return shown


def _partial(window: Window) -> bool:
    start = 0 if window.unit == "chars" else 1
    more = window.total is not None and window.last < window.total
    return window.next_call is not None or window.first > start or more


def _of(total: int | None) -> str:
    return "" if total is None else f" of {total}"


def _onward(window: Window) -> str:
    if window.next_call is not None:
        arguments = (f"{name}={json.dumps(value)}" for name, value in window.next_call.items())
        return "next: " + ", ".join(arguments)
    if window.total is not None and window.last < window.total:
        return "the rest cannot be read here"
    return "end"


def _matches_footer(window: Window) -> str:
    term = json.dumps(window.label)
    if window.first == 0:
        if not window.total:
            return f"[no matches for {term}]"
        return f"[no more matches for {term} of {window.total} | end]"
    counted = f"matches {window.first}-{window.last} of {window.total} for {term}"
    return f"[{counted} | {_onward(window)}]"
