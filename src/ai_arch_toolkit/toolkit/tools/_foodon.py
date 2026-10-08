"""FoodOn: search the food ontology's terms, and read one by ID, through EMBL-EBI's OLS (T07; D39).

OLS4's v1 search is Solr's: ``rows`` and ``start`` page it, and ``response.numFound`` counts the
matches; by default it returns each term's label, IDs, definition and synonyms, and terms FoodOn
imports from other ontologies (``NCBITaxon:3750``) come with their own prefix
(https://github.com/EBISPOT/ols4/blob/dev/backend/src/main/java/uk/ac/ebi/spot/ols/controller/api/v1/V1SearchController.java).
Its errors are ``{"status": …, "message": …}``, whose message the door quotes
(``GlobalExceptionHandler.java``, same repository). A definition is never cut (D39).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api
from ai_arch_toolkit.toolkit.tools._window import list_window

_API = Api(base="https://www.ebi.ac.uk/ols4/api/search", name="EMBL-EBI OLS", timeout_s=15)
_PAGE_MAX = 20
_TERM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*[:_]\d+$")
_SYNONYMS = (
    ("Exact synonyms", "exact_synonyms"),
    ("Related synonyms", "related_synonyms"),
    ("Narrow synonyms", "narrow_synonyms"),
    ("Broad synonyms", "broad_synonyms"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _FoodOnTerm:
    """A term as the tools read it."""

    iri: str
    obo_id: str
    short_form: str
    label: str
    type: str
    ontology: str
    definition: str
    synonyms: tuple[tuple[str, str], ...]


@tool(capability="network")
def foodon_search(
    query: str,
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = 10,
    start: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search FoodOn's food terms, with their definitions.

    Args:
        query: A food concept, e.g. "apple", "yogurt" or "fermented food".
        max_results: How many terms to list.
        start: How many terms to skip; the footer gives the next start.

    Raises:
        ToolFailure: validation_error when ``query`` is empty.
    """
    query = query.strip()
    if not query:
        msg = "query cannot be empty; pass a food concept, e.g. 'apple'."
        raise ToolFailure("validation_error", msg)
    params = {"q": query, "ontology": "foodon", "rows": str(max_results), "start": str(start)}
    return _API.get_json(
        params=params, parse=lambda data: _search_answer(data, query, start, max_results)
    )


@tool(capability="network")
def foodon_term(term_id: str) -> str:
    """Read a FoodOn term by its ID: label, definition and synonyms.

    Args:
        term_id: The term's ID, as foodon_search gives it, e.g. "FOODON:00002473"
            ("FOODON_00002473" works too).

    Raises:
        ToolFailure: validation_error when ``term_id`` is not an ontology ID; not_found when
            FoodOn has no term with it.
    """
    if not _TERM_RE.fullmatch(term_id.strip()):
        msg = f"invalid term_id {term_id!r}; a FoodOn term ID looks like FOODON:00002473."
        raise ToolFailure("validation_error", msg)
    wanted = term_id.strip().replace("_", ":", 1)
    params = {"q": wanted, "ontology": "foodon", "queryFields": "obo_id", "rows": "5"}
    return _API.get_json(params=params, parse=lambda data: _term_text(data, wanted))


def _search_answer(data: dict[str, Any], query: str, start: int, rows: int) -> ToolResult:
    terms = _terms(data)
    found = _dict(data, "response").get("numFound")
    total = found if isinstance(found, int) and not isinstance(found, bool) else None
    if not terms and total and start >= total:
        return ToolResult.success(
            f"start={start} is past the end: {total} FoodOn terms match {query!r}; the last page "
            f"is start={(total - 1) // rows * rows}."
        )
    if not terms:
        later = f" after the first {start}" if start else ""
        return ToolResult.success(f"No FoodOn terms match {query!r}{later}.")
    shown = start + len(terms)
    next_call = {"start": shown} if total is not None and shown < total else None
    entries = [_entry(number, term) for number, term in enumerate(terms, start=start + 1)]
    window = list_window(entries, first=start + 1, total=total, next_call=next_call)
    return window.result(heading=f"FoodOn terms for {query!r} (details: foodon_term):")


def _entry(number: int, term: _FoodOnTerm) -> str:
    head = [f"{number}. {term.label}", term.obo_id, term.type]
    head.append(f"ontology {term.ontology}" if term.ontology else "")
    lines = [" | ".join(part for part in head if part)]
    lines += [_labelled("Definition", term.definition), _labelled("IRI", term.iri)]
    return "\n   ".join(line for line in lines if line)


def _term_text(data: dict[str, Any], wanted: str) -> str:
    matches = [term for term in _terms(data) if term.obo_id.upper() == wanted.upper()]
    if not matches:
        msg = f"no FoodOn term with ID {wanted.upper()}; search with foodon_search."
        raise ToolFailure("not_found", msg)
    term = matches[0]
    facts = [
        _labelled("type", term.type),
        _labelled("ontology", term.ontology),
        _labelled("short form", term.short_form),
    ]
    lines = [
        f"FoodOn term {term.obo_id}: {term.label}",
        " | ".join(fact for fact in facts if fact),
        _labelled("Definition", term.definition),
        *(_labelled(label, synonyms) for label, synonyms in term.synonyms),
        _labelled("IRI", term.iri),
    ]
    return "\n   ".join(line for line in lines if line)


def _terms(data: dict[str, Any]) -> list[_FoodOnTerm]:
    docs = _dict(data, "response").get("docs")
    if not isinstance(docs, list):
        return []
    return [term for item in docs if isinstance(item, dict) if (term := _parse_term(item))]


def _parse_term(data: dict[str, Any]) -> _FoodOnTerm | None:
    label = _string(data.get("label"))
    obo_id = _string(data.get("obo_id"))
    if not label and not obo_id:
        return None
    synonyms = tuple(
        (name, ", ".join(found)) for name, key in _SYNONYMS if (found := _strings(data.get(key)))
    )
    return _FoodOnTerm(
        iri=_string(data.get("iri")),
        obo_id=obo_id,
        short_form=_string(data.get("short_form")),
        label=label or "(unlabeled)",
        type=_string(data.get("type")),
        ontology=_string(data.get("ontology_name")),
        definition=" ".join(_strings(data.get("description"))),
        synonyms=synonyms,
    )


def _labelled(label: str, value: str) -> str:
    return f"{label}: {value}" if value else ""


def _dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _strings(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [text for item in values if (text := _string(item))]


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
