"""FoodOn tools — public food ontology search through EMBL-EBI OLS."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(base="https://www.ebi.ac.uk/ols4/api/search", name="EMBL-EBI OLS", timeout_s=15)
_MAX_RESULTS_LIMIT = 20
_TERM_RE = re.compile(r"^(FOODON[:_]\d{7,}|[A-Za-z]+[:_]\d+)$", re.IGNORECASE)
_DESCRIPTION_MAX_CHARS = 700


@dataclass(frozen=True, slots=True, kw_only=True)
class _FoodOnTerm:
    """Normalized FoodOn ontology term."""

    iri: str
    obo_id: str
    short_form: str
    label: str
    type: str
    ontology: str
    descriptions: tuple[str, ...]


@tool(capability="network")
def foodon_search(query: str, max_results: int = 10, start: int = 0) -> str:
    """Search FoodOn food ontology terms via EMBL-EBI OLS.

    Args:
        query: Food concept search text, e.g. "apple", "yogurt", or "fermented food".
        max_results: Number of terms to return (1-20). Defaults to 10.
        start: Zero-based result offset. Defaults to 0.
    """
    query = query.strip()
    if not query:
        return "FoodOn search failed: query cannot be empty."
    if start < 0:
        return "FoodOn search failed: start must be greater than or equal to 0."

    try:
        return _API.get_json(
            params=_search_params(query, max_results=max_results, start=start),
            parse=lambda data: _search_text(data, query, start),
        )
    except HttpError as e:
        return f"FoodOn search failed: {e}"


@tool(capability="network")
def foodon_term(term_id: str) -> str:
    """Fetch a FoodOn ontology term by OBO ID.

    Args:
        term_id: FoodOn OBO ID, e.g. "FOODON:00002473" or "FOODON_00002473".
    """
    normalized = term_id.strip().replace("_", ":").upper()
    if not _TERM_RE.fullmatch(term_id.strip()):
        return f"FoodOn term lookup failed: invalid term_id: {term_id!r}"

    try:
        return _API.get_json(
            params=_search_params(normalized, max_results=5, start=0),
            parse=lambda data: _term_text(data, normalized),
        )
    except HttpError as e:
        return f"FoodOn term lookup failed: {e}"


def _search_params(query: str, *, max_results: int, start: int) -> dict[str, str]:
    return {
        "q": query,
        "ontology": "foodon",
        "rows": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "start": str(start),
    }


def _search_text(data: dict[str, Any], query: str, start: int) -> str:
    terms = _terms_from_data(data)
    if not terms:
        return f"No FoodOn terms found for: {query!r}"
    total = _string(data.get("response", {}).get("numFound")) or "?"
    return (
        f"FoodOn terms for {query!r} (start {start}, returned {len(terms)}, total {total}):\n"
        + _format_terms(terms)
    )


def _term_text(data: dict[str, Any], normalized: str) -> str:
    terms = [term for term in _terms_from_data(data) if term.obo_id.upper() == normalized]
    if not terms:
        return f"FoodOn term not found: {normalized}"
    return f"FoodOn term {normalized}:\n" + _format_terms(
        [terms[0]],
        include_index=False,
        include_full_description=True,
    )


def _terms_from_data(data: dict[str, Any]) -> list[_FoodOnTerm]:
    docs = data.get("response", {}).get("docs", [])
    if not isinstance(docs, list):
        return []
    return [term for item in docs if isinstance(item, dict) if (term := _parse_term(item))]


def _parse_term(data: dict[str, Any]) -> _FoodOnTerm | None:
    label = _string(data.get("label"))
    obo_id = _string(data.get("obo_id"))
    if not label and not obo_id:
        return None
    return _FoodOnTerm(
        iri=_string(data.get("iri")),
        obo_id=obo_id,
        short_form=_string(data.get("short_form")),
        label=label or "(unlabeled)",
        type=_string(data.get("type")),
        ontology=_string(data.get("ontology_name")),
        descriptions=_string_tuple(data.get("description")),
    )


def _format_terms(
    terms: list[_FoodOnTerm],
    *,
    include_index: bool = True,
    include_full_description: bool = False,
) -> str:
    blocks: list[str] = []
    for index, term in enumerate(terms, start=1):
        title = f"{index}. {term.label}" if include_index else term.label
        lines = [title]
        meta = []
        if term.obo_id:
            meta.append(f"id: {term.obo_id}")
        if term.short_form:
            meta.append(f"short: {term.short_form}")
        if term.type:
            meta.append(f"type: {term.type}")
        if term.ontology:
            meta.append(f"ontology: {term.ontology}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if term.descriptions:
            description = " ".join(term.descriptions)
            if not include_full_description:
                description = _truncate(description, _DESCRIPTION_MAX_CHARS)
            lines.append(f"   Definition: {description}")
        if term.iri:
            lines.append(f"   IRI: {term.iri}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(_string(item) for item in value if _string(item))
    text = _string(value)
    return (text,) if text else ()


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
