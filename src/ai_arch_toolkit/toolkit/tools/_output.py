"""The output of a run — a shell command's, a Python program's — within a limit (T09, D39).

What a run printed exists only for the call that made it: reading on would run it again, with its
effects. So a part that does not fit shows its start, ending on a line, and the window's footer
says its size and how to narrow the output, where other tools name the call that reads on::

    [chars 0-7998 of 48893 | the rest is not kept: run the command again narrowed, e.g. ...]

A stream is kept as it arrives (:class:`Head`): its start, and a count of the rest, so memory stays
within the limit however much the run prints.
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_arch_toolkit.core._tools._result import line_cut
from ai_arch_toolkit.toolkit.tools._window import Window


class Head:
    """The start of a stream as it arrives: its first ``keep`` characters, and how many came."""

    __slots__ = ("_kept", "_parts", "keep", "total")

    def __init__(self, keep: int) -> None:
        self.keep = keep
        self.total = 0
        self._kept = 0
        self._parts: list[str] = []

    def add(self, text: str) -> None:
        """Count ``text``, and keep what still fits."""
        self.total += len(text)
        if self._kept < self.keep:
            piece = text[: self.keep - self._kept]
            self._parts.append(piece)
            self._kept += len(piece)

    @property
    def text(self) -> str:
        """The characters kept: the whole stream when ``total`` is within ``keep``."""
        return "".join(self._parts)


def fit(heads: Sequence[Head], *, limit: int, rest: str) -> list[str]:
    """Each part's text, within ``limit`` characters in all.

    A part that fits shows whole. Of one that does not, the start shows, ending on a line, then
    the window's footer: what was shown, the part's size, and ``rest``, how to narrow the output.
    A part shorter than its even share of ``limit`` shows whole and leaves the rest of its share to
    the others. Each head must keep more than ``limit`` characters.
    """
    shares = _shares([head.total for head in heads], limit)
    return [_part(head, share, rest) for head, share in zip(heads, shares, strict=True)]


def _shares(sizes: Sequence[int], limit: int) -> list[int]:
    """How many characters of each part to show: the shortest first, each up to an even share
    of what the shorter ones left."""
    shares = [0] * len(sizes)
    left = limit
    order = sorted(range(len(sizes)), key=sizes.__getitem__)
    for done, index in enumerate(order):
        shares[index] = min(sizes[index], left // (len(order) - done))
        left -= shares[index]
    return shares


def _part(head: Head, share: int, rest: str) -> str:
    if head.total <= share:
        return head.text
    text = head.text
    end = line_cut(text, 0, share)
    window = Window(body=text[:end], unit="chars", first=0, last=end, total=head.total, rest=rest)
    return window.text()
