"""FoodOn tools — public food ontology search through EMBL-EBI OLS."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit; research tool)"
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


@tool
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
        data = _fetch_search(query, max_results=max_results, start=start)
        terms = _terms_from_data(data)
    except urllib.error.HTTPError as e:
        return _http_error("FoodOn search failed", e)
    except urllib.error.URLError as e:
        return f"FoodOn search failed: URL error: {e.reason}"
    except TimeoutError:
        return "FoodOn search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"FoodOn search failed: could not parse API response: {e}"

    if not terms:
        return f"No FoodOn terms found for: {query!r}"
    total = _string(data.get("response", {}).get("numFound")) or "?"
    return (
        f"FoodOn terms for {query!r} (start {start}, returned {len(terms)}, total {total}):\n"
        + _format_terms(terms)
    )


@tool
def foodon_term(term_id: str) -> str:
    """Fetch a FoodOn ontology term by OBO ID.

    Args:
        term_id: FoodOn OBO ID, e.g. "FOODON:00002473" or "FOODON_00002473".
    """
    normalized = term_id.strip().replace("_", ":").upper()
    if not _TERM_RE.fullmatch(term_id.strip()):
        return f"FoodOn term lookup failed: invalid term_id: {term_id!r}"

    try:
        data = _fetch_search(normalized, max_results=5, start=0)
        terms = [term for term in _terms_from_data(data) if term.obo_id.upper() == normalized]
    except urllib.error.HTTPError as e:
        return _http_error("FoodOn term lookup failed", e)
    except urllib.error.URLError as e:
        return f"FoodOn term lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "FoodOn term lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"FoodOn term lookup failed: could not parse API response: {e}"

    if not terms:
        return f"FoodOn term not found: {normalized}"
    return f"FoodOn term {normalized}:\n" + _format_terms(
        [terms[0]],
        include_index=False,
        include_full_description=True,
    )


def _fetch_search(query: str, *, max_results: int, start: int) -> dict[str, Any]:
    params = {
        "q": query,
        "ontology": "foodon",
        "rows": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "start": str(start),
    }
    url = f"{_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 429:
        return f"{prefix}: rate limited by EMBL-EBI OLS (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


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
