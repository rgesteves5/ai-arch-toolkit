"""The contract's debt: each tool that does not keep a point yet, and the points (T04b).

The contract test (``test_tool_contract.py``) fails when a tool owes a point not listed here, and
when a tool keeps a point it still lists: the list only shrinks. Whoever migrates a module (T05 to
T09) deletes its tools' points and nothing else.

Initial size (2026-10-05): 121 tools; window 96, errors 91, not_found 22, zero 42, limits 70.
"""

from __future__ import annotations


def _owes(*points: str) -> frozenset[str]:
    return frozenset(points)


# Definitions of the copied cut helpers (``_truncate``, ``_trim``) left in ``toolkit/tools``.
CUT_HELPERS = 0

DEBT: dict[str, frozenset[str]] = {}
