"""Deterministic lexical search for small knowledge registries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_arch_toolkit.toolkit.knowledge._registry import KnowledgeEntry

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeSearchResult:
    """One knowledge match with an explainable lexical relevance score."""

    entry: KnowledgeEntry
    score: float
    matched_terms: tuple[str, ...]


def search_entries(
    entries: tuple[KnowledgeEntry, ...],
    query: str,
    *,
    limit: int = 10,
) -> tuple[KnowledgeSearchResult, ...]:
    """Rank entries by exact token matches in domain fields and content."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("knowledge search query must be a non-empty string")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("knowledge search limit must be a positive integer")
    terms = tuple(dict.fromkeys(_tokens(query)))
    results: list[KnowledgeSearchResult] = []
    for entry in entries:
        key_tokens = _tokens(entry.key)
        category_tokens = _tokens(entry.category)
        tag_tokens = [token for tag in entry.tags for token in _tokens(tag)]
        content_tokens = _tokens(entry.content)
        matched = tuple(
            term
            for term in terms
            if term in key_tokens
            or term in category_tokens
            or term in tag_tokens
            or term in content_tokens
        )
        if not matched:
            continue
        score = sum(
            key_tokens.count(term) * 4
            + tag_tokens.count(term) * 3
            + category_tokens.count(term) * 2
            + content_tokens.count(term)
            for term in matched
        ) / len(terms)
        results.append(
            KnowledgeSearchResult(entry=entry, score=float(score), matched_terms=matched)
        )
    results.sort(key=lambda result: (-result.score, result.entry.key))
    return tuple(results[:limit])


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD.finditer(value)]


__all__ = ["KnowledgeSearchResult", "search_entries"]
