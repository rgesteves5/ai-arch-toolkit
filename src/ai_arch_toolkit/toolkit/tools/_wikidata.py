"""Wikidata tools — public entity search, entity lookup, and SPARQL queries."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api
from ai_arch_toolkit.toolkit.tools._mediawiki import mediawiki_error

_API = Api(
    base="https://www.wikidata.org/w/api.php",
    name="Wikidata",
    timeout_s=15,
    error_reader=mediawiki_error,
)
_ENTITY_DATA = Api(
    base="https://www.wikidata.org/wiki/Special:EntityData", name="Wikidata", timeout_s=15
)
_SPARQL = Api(
    base="https://query.wikidata.org/sparql", name="Wikidata Query Service", timeout_s=15
)
_MAX_RESULTS_LIMIT = 20
_QID_RE = re.compile(r"^Q\d+$", re.IGNORECASE)
_LANG_RE = re.compile(r"^[a-z][a-z0-9-]{0,15}$", re.IGNORECASE)
_UNSAFE_SPARQL = re.compile(
    r"\b(INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|MOVE|COPY|ADD)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _WikidataSearchResult:
    """Normalized Wikidata search result."""

    qid: str
    label: str
    description: str
    match: str
    url: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _WikidataEntity:
    """Normalized Wikidata entity metadata."""

    qid: str
    label: str
    description: str
    aliases: tuple[str, ...]
    claims: tuple[str, ...]
    wikipedia_url: str
    wikidata_url: str


@tool(capability="network")
def wikidata_search(query: str, max_results: int = 5, language: str = "en") -> str:
    """Search Wikidata entities using the public Wikidata API.

    Args:
        query: Entity search text.
        max_results: Number of entities to return (1-20). Defaults to 5.
        language: Search language code. Defaults to "en".

    Raises:
        ToolFailure: validation_error when the query is empty or the language code is malformed.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "empty query; give the entity's name")
    language = _language(language)

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": language,
        "uselang": language,
        "type": "item",
        "limit": str(max_results),
        "format": "json",
    }
    results = _API.get_json(params=params, parse=_search_results)

    if not results:
        return f"No Wikidata results for: {query!r}"

    return f"Wikidata results for {query!r}:\n" + _format_search_results(results)


@tool(capability="network")
def wikidata_entity(qid: str, language: str = "en") -> str:
    """Fetch a Wikidata entity by QID.

    Args:
        qid: Wikidata item ID, e.g. "Q42".
        language: Preferred label/description language. Defaults to "en".

    Raises:
        ToolFailure: validation_error when the QID or the language code is malformed; not_found
            when Wikidata has no entity with that QID.
    """
    normalized = qid.strip().upper()
    if not _QID_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid QID {qid!r}; a QID looks like Q42 (find one with wikidata_search)",
        )
    language = _language(language)

    missing = f"Wikidata has no entity {normalized}; search for it with wikidata_search"
    entity = _ENTITY_DATA.get_json(
        f"{normalized}.json",
        parse=lambda data: _entity(data, normalized, language),
        missing=missing,
    )
    if entity is None:
        raise ToolFailure("not_found", missing)

    merged = f" (redirects to {entity.qid})" if entity.qid != normalized else ""
    return f"Wikidata entity {normalized}{merged}:\n" + _format_entity(entity)


@tool(capability="network")
def wikidata_sparql(query: str, max_results: int = 20) -> str:
    """Run a read-only Wikidata SPARQL SELECT or ASK query.

    Args:
        query: SPARQL SELECT or ASK query.
        max_results: LIMIT to append for SELECT queries without an explicit LIMIT (1-20).

    Raises:
        ToolFailure: validation_error when the query is empty, would change data, or is not a
            SELECT or ASK query; upstream when the query service rejects it.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "empty query; write a SPARQL SELECT or ASK query")
    if _UNSAFE_SPARQL.search(query):
        raise ToolFailure(
            "validation_error",
            "only read-only SELECT or ASK queries are allowed; remove the update keywords",
        )
    if not re.match(r"^(PREFIX\s+\w+:\s*<[^>]+>\s*)*(SELECT|ASK)\b", query, re.IGNORECASE):
        raise ToolFailure(
            "validation_error", "the query must start with SELECT or ASK (after any PREFIX lines)"
        )

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    sparql = query
    if re.match(r"^(PREFIX\s+\w+:\s*<[^>]+>\s*)*SELECT\b", query, re.IGNORECASE) and not re.search(
        r"\bLIMIT\s+\d+\b", query, re.IGNORECASE
    ):
        sparql = f"{query}\nLIMIT {max_results}"

    return _SPARQL.get_json(
        params={"query": sparql, "format": "json"},
        parse=lambda data: _sparql_text(data, max_results),
    )


def _language(language: str) -> str:
    """The stripped language code, "en" when empty; a malformed one raises ``ToolFailure``."""
    language = language.strip() or "en"
    if not _LANG_RE.fullmatch(language):
        raise ToolFailure(
            "validation_error", f"invalid language {language!r}; use a code such as 'en' or 'pt'"
        )
    return language


def _search_results(data: dict[str, Any]) -> list[_WikidataSearchResult]:
    items = data.get("search", [])
    results = [_parse_search_result(item) for item in items if isinstance(item, dict)]
    return [item for item in results if item is not None]


def _entity(data: dict[str, Any], qid: str, language: str) -> _WikidataEntity | None:
    """The entity ``qid`` names, or for a merged QID the one it redirects to.

    Special:EntityData follows the redirect (https://www.wikidata.org/wiki/Help:Redirects), so
    the answer holds only the target, under its own ID.
    """
    entities = data.get("entities", {})
    target, entity_data = qid, entities.get(qid)
    if entity_data is None and len(entities) == 1:
        target, entity_data = next(iter(entities.items()))
    if not isinstance(entity_data, dict) or "missing" in entity_data:
        return None
    return _parse_entity(target, entity_data, language)


def _sparql_text(data: dict[str, Any], max_results: int) -> str:
    if "boolean" in data:
        return f"Wikidata SPARQL result: {data['boolean']}"

    head = data.get("head", {})
    results = data.get("results", {}).get("bindings", [])
    variables = head.get("vars", [])
    if not isinstance(variables, list) or not isinstance(results, list):
        raise ToolFailure(
            "upstream", "the Wikidata Query Service answered without head.vars or bindings"
        )
    if not results:
        return "Wikidata SPARQL returned no rows."

    return "Wikidata SPARQL rows:\n" + _format_sparql_rows(variables, results[:max_results])


def _parse_search_result(data: dict[str, Any]) -> _WikidataSearchResult | None:
    qid = str(data.get("id", "")).strip()
    if not qid:
        return None
    return _WikidataSearchResult(
        qid=qid,
        label=str(data.get("label", "") or "").strip() or "(unlabeled)",
        description=str(data.get("description", "") or "").strip(),
        match=str(
            data.get("match", {}).get("text", "") if isinstance(data.get("match"), dict) else ""
        ),
        url=str(data.get("concepturi", "") or f"https://www.wikidata.org/wiki/{qid}").strip(),
    )


def _parse_entity(qid: str, data: dict[str, Any], language: str) -> _WikidataEntity:
    labels = data.get("labels", {})
    descriptions = data.get("descriptions", {})
    aliases = data.get("aliases", {})
    sitelinks = data.get("sitelinks", {})

    return _WikidataEntity(
        qid=qid,
        label=_localized_value(labels, language) or "(unlabeled)",
        description=_localized_value(descriptions, language),
        aliases=_aliases(aliases, language),
        claims=_claims(data.get("claims")),
        wikipedia_url=_wikipedia_url(sitelinks, language),
        wikidata_url=f"https://www.wikidata.org/wiki/{qid}",
    )


def _format_search_results(results: list[_WikidataSearchResult]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(results, start=1):
        lines = [f"{index}. {item.label} ({item.qid})"]
        if item.description:
            lines.append(f"   Description: {item.description}")
        if item.match and item.match != item.label:
            lines.append(f"   Match: {item.match}")
        lines.append(f"   URL: {item.url}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_entity(entity: _WikidataEntity) -> str:
    lines = [entity.label]
    if entity.description:
        lines.append(f"   Description: {entity.description}")
    if entity.aliases:
        lines.append("   Aliases: " + ", ".join(entity.aliases[:12]))
    if entity.claims:
        lines.append("   Claims:")
        for claim in entity.claims[:15]:
            lines.append(f"     - {claim}")
    if entity.wikipedia_url:
        lines.append(f"   Wikipedia: {entity.wikipedia_url}")
    lines.append(f"   Wikidata: {entity.wikidata_url}")
    return "\n".join(lines)


def _format_sparql_rows(variables: list[Any], rows: list[Any]) -> str:
    blocks: list[str] = []
    for index, row in enumerate(rows, start=1):
        parts: list[str] = []
        for variable in variables:
            value = row.get(str(variable), {}) if isinstance(row, dict) else {}
            if isinstance(value, dict):
                text = str(value.get("value", "")).strip()
                if text:
                    parts.append(f"{variable}: {text}")
        if parts:
            blocks.append(f"{index}. " + " | ".join(parts))
    return "\n".join(blocks) if blocks else "No displayable SPARQL values."


def _localized_value(values: Any, language: str) -> str:
    if not isinstance(values, dict):
        return ""
    for key in (language, "en"):
        item = values.get(key)
        if isinstance(item, dict):
            text = str(item.get("value", "")).strip()
            if text:
                return text
    for item in values.values():
        if isinstance(item, dict):
            text = str(item.get("value", "")).strip()
            if text:
                return text
    return ""


def _aliases(values: Any, language: str) -> tuple[str, ...]:
    if not isinstance(values, dict):
        return ()
    items = values.get(language) or values.get("en") or []
    if not isinstance(items, list):
        return ()
    return tuple(
        str(item.get("value", "")).strip()
        for item in items
        if isinstance(item, dict) and item.get("value")
    )


def _claims(values: Any) -> tuple[str, ...]:
    if not isinstance(values, dict):
        return ()
    claims: list[str] = []
    for prop, items in list(values.items())[:20]:
        if not isinstance(items, list) or not items:
            continue
        rendered_values: list[str] = []
        for item in items[:3]:
            if isinstance(item, dict):
                rendered = _claim_value(item)
                if rendered:
                    rendered_values.append(rendered)
        if rendered_values:
            claims.append(f"{prop}: {', '.join(rendered_values)}")
    return tuple(claims)


def _claim_value(claim: dict[str, Any]) -> str:
    mainsnak = claim.get("mainsnak")
    if not isinstance(mainsnak, dict):
        return ""
    datavalue = mainsnak.get("datavalue")
    if not isinstance(datavalue, dict):
        return ""
    value = datavalue.get("value")
    if isinstance(value, dict):
        if "id" in value:
            return str(value["id"])
        if "time" in value:
            return str(value["time"])
        if "amount" in value:
            return str(value["amount"])
        if "text" in value:
            return str(value["text"])
    if value is not None:
        return str(value)
    return ""


def _wikipedia_url(sitelinks: Any, language: str) -> str:
    if not isinstance(sitelinks, dict):
        return ""
    key = f"{language}wiki"
    item = sitelinks.get(key) or sitelinks.get("enwiki")
    if isinstance(item, dict):
        title = str(item.get("title", "")).strip()
        if title:
            lang = key.removesuffix("wiki") if key in sitelinks else "en"
            return (
                f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            )
    return ""
