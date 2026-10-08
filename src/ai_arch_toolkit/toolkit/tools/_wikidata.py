"""Wikidata tools: entity search, entity statements and SPARQL queries.

Every long answer goes through the window (D39), and every code comes with its label: an
entity's statements are read 40 at a time, and the properties, items and units on the page are
named by one ``wbgetentities`` request per 50 codes (its limit;
https://www.wikidata.org/w/api.php?action=help&modules=wbgetentities). Values follow the data
model (https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html): dates to their
precision in ISO 8601, quantities with their unit, coordinates as latitude and longitude. A
SPARQL answer keeps every row the query asked for, and reads on by ``offset``.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._mediawiki import mediawiki_error
from ai_arch_toolkit.toolkit.tools._values import decimal_text, plain
from ai_arch_toolkit.toolkit.tools._window import list_window, page_window


def _sparql_error(reply: Reply) -> ToolFailure | None:
    """The error the query service explains in an error status's text; ``None`` otherwise.

    A query it cannot parse names a ``MalformedQueryException``: the caller's to fix. A query
    past the service's 60 s deadline names a ``TimeoutException``; the same query would time
    out again (https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual#Query_limits).
    """
    body = reply.body if isinstance(reply.body, str) else ""
    if reply.status < 400 or not body:
        return None
    if "TimeoutException" in body:
        return ToolFailure(
            "upstream",
            "the query ran past the query service's 60 s limit (TimeoutException); narrow it "
            "(fewer patterns, a LIMIT, no label service on many rows) and try again",
        )
    malformed = re.search(r"MalformedQueryException: ([^\n]+)", body)
    if malformed:
        said = " ".join(malformed[1].split())  # one line, of an error body the door caps
        return ToolFailure(
            "validation_error",
            f"the query service could not parse the query ({said}); fix the query",
        )
    return None


_API = Api(
    base="https://www.wikidata.org/w/api.php",
    name="Wikidata",
    timeout_s=15,
    error_reader=mediawiki_error,
)
_ENTITY_DATA = Api(
    base="https://www.wikidata.org/wiki/Special:EntityData", name="Wikidata", timeout_s=15
)
# The service lets a query run 60 s (its user manual); the client waits a little longer.
_SPARQL = Api(
    base="https://query.wikidata.org/sparql",
    name="Wikidata Query Service",
    timeout_s=65,
    error_reader=_sparql_error,
)
# wbsearchentities continues up to the 10,000th result (its ``continue``).
_SEARCH_DEPTH = 10_000
_STATEMENTS_PAGE = 40
_LABELS_BATCH = 50
# A page of 40 statements names at most 120 codes (properties, values, units), and its
# qualifiers': four requests label all but the most qualified pages' (the rest stay bare codes).
_LABEL_REQUESTS = 4
# The rows a SELECT without a LIMIT of its own may return.
_ROW_CAP = 1000
_ID_RE = re.compile(r"^[QP]\d+$")
_LANG_RE = re.compile(r"^[a-z][a-z0-9-]{0,15}$", re.IGNORECASE)
# The update keywords, as keywords: not a variable (``?add``) nor part of a prefixed name
# (``ex:move``, ``copy:x``).
_UNSAFE_SPARQL = re.compile(
    r"(?<![?$:])\b(INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|MOVE|COPY|ADD)\b(?!:)",
    re.IGNORECASE,
)
_PREFIXES = r"^(PREFIX\s+\w*:\s*<[^>]+>\s*)*"
# The tokens of a query whose text is not its syntax: strings, IRIs and comments, which run from
# a ``#`` to the line's end (https://www.w3.org/TR/sparql11-query/#grammar, STRING_LITERAL*,
# IRIREF and section 19.4).
_OPAQUE = re.compile(
    r'"""(?:[^"\\]|\\.|"(?!""))*"""'
    r"|'''(?:[^'\\]|\\.|'(?!''))*'''"
    r'|"(?:[^"\\\n]|\\.)*"'
    r"|'(?:[^'\\\n]|\\.)*'"
    r'|<[^<>"{}|^`\\\s]*>'
    r"|#[^\n]*",
    re.DOTALL,
)
# A VALUES clause after the WHERE clause, which ends the query (the grammar's ValuesClause).
_TRAILING_VALUES = re.compile(
    r"\bVALUES\s*(?:[?$]\w+|\((?:\s*[?$]\w+)*\s*\))\s*\{[^{}]*\}\s*$", re.IGNORECASE
)
_ENTITY_URI = "http://www.wikidata.org/entity/"
_EARTH = f"{_ENTITY_URI}Q2"
_JULIAN = f"{_ENTITY_URI}Q1985786"
_NUMERIC_TYPES = ("#double", "#float", "#decimal", "#integer")
# Time precision below a year (https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html).
_COARSE = {
    8: "decade",
    7: "century",
    6: "millennium",
    5: "10000 years",
    4: "100000 years",
    3: "million years",
    2: "10 million years",
    1: "100 million years",
    0: "billion years",
}

# A piece of a value: text, or a code (``Q5``) to show with its label.
type _Part = tuple[bool, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Claim:
    """One statement, read inside the door: its property, its value as parts, its rank and its
    qualifiers (``(property, value parts)``, in the answer's order)."""

    prop: str
    value: tuple[_Part, ...]
    rank: str
    qualifiers: tuple[tuple[str, tuple[_Part, ...]], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Entity:
    """An entity's terms, its links and its statements, in the entity's property order."""

    id: str
    label: str
    description: str
    aliases: tuple[str, ...]
    wikipedia_url: str
    claims: tuple[_Claim, ...]


@tool(capability="network")
def wikidata_search(
    query: str,
    max_results: Annotated[int, Range(1, 50)] = 5,
    offset: Annotated[int, Range(0, _SEARCH_DEPTH)] = 0,
    language: str = "en",
) -> ToolResult:
    """Search Wikidata items by label or alias.

    Args:
        query: The item's name, or part of it.
        max_results: How many items to show.
        offset: How many results to skip; the footer gives the next offset.
        language: The language of the labels to search and show, e.g. "en" or "pt".

    Raises:
        ToolFailure: validation_error when the query is empty or the language code is malformed.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "empty query; give the entity's name")
    language = _language(language)
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": language,
        "uselang": language,
        "type": "item",
        "limit": str(max_results),
        "continue": str(offset),
        "format": "json",
    }
    return _API.get_json(params=params, parse=lambda data: _search_answer(data, query, offset))


@tool(capability="network")
def wikidata_entity(
    qid: str, language: str = "en", offset: Annotated[int, Range(0)] = 0
) -> ToolResult:
    """Read a Wikidata item or property: its label, description, aliases, Wikipedia link and
    statements, each code with its label.

    Args:
        qid: The item's or property's ID, e.g. "Q42" or "P31".
        language: The language of the labels, e.g. "en" or "pt" (English when one is missing).
        offset: How many statements to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the ID or the language code is malformed; not_found
            when Wikidata has no entity with that ID.
    """
    normalized = qid.strip().upper()
    if not _ID_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid ID {qid!r}; a Wikidata ID looks like Q42 (an item) or P31 (a property); "
            "find one with wikidata_search",
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
    page = entity.claims[offset : offset + _STATEMENTS_PAGE]
    labels, unlabelled = _labels(_codes(page), language)
    return _entity_answer(entity, normalized, page, labels, offset, unlabelled)


@tool(capability="network")
def wikidata_sparql(
    query: str,
    max_results: Annotated[int, Range(1, 100)] = 20,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Run a read-only Wikidata SPARQL SELECT or ASK query; every row it returns can be read.

    A SELECT without a LIMIT gets ``LIMIT 1000``, and the answer says when it reached it.

    Args:
        query: SPARQL SELECT or ASK query.
        max_results: How many rows to show.
        offset: How many rows to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the query is empty, would change data, is not a
            SELECT or ASK query, or the query service cannot parse it; upstream when the query
            service fails or the query runs past its time limit.
    """
    query = query.strip()
    if not query:
        raise ToolFailure("validation_error", "empty query; write a SPARQL SELECT or ASK query")
    shape = _shape(query)
    if _UNSAFE_SPARQL.search(shape):
        raise ToolFailure(
            "validation_error",
            "only read-only SELECT or ASK queries are allowed; remove the update keywords",
        )
    if not re.match(_PREFIXES + r"(SELECT|ASK)\b", shape.lstrip(), re.IGNORECASE):
        raise ToolFailure(
            "validation_error", "the query must start with SELECT or ASK (after any PREFIX lines)"
        )
    sparql, capped = _with_limit(query, shape)
    return _SPARQL.get_json(
        params={"query": sparql, "format": "json"},
        parse=lambda data: _sparql_answer(data, query, capped, max_results, offset),
    )


def _language(language: str) -> str:
    """The stripped language code, "en" when empty; a malformed one raises ``ToolFailure``."""
    language = language.strip() or "en"
    if not _LANG_RE.fullmatch(language):
        raise ToolFailure(
            "validation_error", f"invalid language {language!r}; use a code such as 'en' or 'pt'"
        )
    return language


# --- Search ------------------------------------------------------------------------------------


def _search_answer(data: dict[str, Any], query: str, offset: int) -> ToolResult:
    items = [item for item in data.get("search", []) if isinstance(item, dict) and item.get("id")]
    if not items and offset == 0:
        return ToolResult.success(f"No Wikidata items match {query!r}.")
    lines = [_search_line(number, item) for number, item in enumerate(items, offset + 1)]
    onward = data.get("search-continue")
    next_call = {"offset": onward} if isinstance(onward, int) and onward > offset else None
    window = list_window(lines, first=offset + 1, next_call=next_call)
    return window.result(
        heading=f"Wikidata items that match {query!r} (wikidata_entity reads one):"
    )


def _search_line(number: int, item: dict[str, Any]) -> str:
    label = _string(item.get("label")) or "(no label)"
    line = f"{number}. {label} ({_string(item.get('id'))})"
    if description := _string(item.get("description")):
        line += f": {description}"
    match = item.get("match")
    matched = _string(match.get("text")) if isinstance(match, dict) else ""
    return line + (f"\n   matched: {matched}" if matched and matched != label else "")


# --- Entity ------------------------------------------------------------------------------------


def _entity(data: dict[str, Any], qid: str, language: str) -> _Entity | None:
    """The entity ``qid`` names, or for a merged ID the one it redirects to.

    Special:EntityData follows the redirect (https://www.wikidata.org/wiki/Help:Redirects), so
    the answer holds only the target, under its own ID.
    """
    entities = data.get("entities", {})
    target, entity_data = qid, entities.get(qid)
    if entity_data is None and len(entities) == 1:
        target, entity_data = next(iter(entities.items()))
    if not isinstance(entity_data, dict) or "missing" in entity_data:
        return None
    claims = entity_data.get("claims")
    return _Entity(
        id=str(target),
        label=_localized(entity_data.get("labels"), language),
        description=_localized(entity_data.get("descriptions"), language),
        aliases=_aliases(entity_data.get("aliases"), language),
        wikipedia_url=_wikipedia_url(entity_data.get("sitelinks"), language),
        claims=tuple(_claims(claims if isinstance(claims, dict) else {})),
    )


def _claims(claims: Mapping[str, Any]) -> Iterable[_Claim]:
    for prop, statements in claims.items():
        for statement in statements if isinstance(statements, list) else ():
            if isinstance(statement, dict):
                yield _claim(str(prop), statement)


def _claim(prop: str, statement: dict[str, Any]) -> _Claim:
    qualifiers = statement.get("qualifiers")
    shown = [
        (str(key), _snak_parts(snak))
        for key, snaks in (qualifiers.items() if isinstance(qualifiers, dict) else ())
        for snak in (snaks if isinstance(snaks, list) else ())
    ]
    return _Claim(
        prop=prop,
        value=_snak_parts(statement.get("mainsnak")),
        rank=_string(statement.get("rank")) or "normal",
        qualifiers=tuple(shown),
    )


def _datavalue(snak: object) -> dict[str, Any]:
    value = snak.get("datavalue") if isinstance(snak, dict) else None
    return value if isinstance(value, dict) else {}


def _snak_parts(snak: object) -> tuple[_Part, ...]:
    """A snak's value as parts: text, and the codes to name with their labels."""
    kind = snak.get("snaktype") if isinstance(snak, dict) else None
    if kind == "somevalue":
        return ((False, "unknown value"),)
    if kind == "novalue":
        return ((False, "no value"),)
    datavalue = _datavalue(snak)
    reader = _VALUE_READERS.get(_string(datavalue.get("type")))
    value = datavalue.get("value")
    if reader is not None and isinstance(value, dict):
        return reader(value)
    return ((False, _string(value) or "(no value)"),)


def _entity_parts(value: dict[str, Any]) -> tuple[_Part, ...]:
    code = _string(value.get("id"))
    return ((True, code),) if code else ((False, "(no value)"),)


def _quantity_parts(value: dict[str, Any]) -> tuple[_Part, ...]:
    amount: tuple[_Part, ...] = ((False, decimal_text(_string(value.get("amount")))),)
    unit = _code_of(_string(value.get("unit")))
    return (*amount, (False, " "), (True, unit)) if unit else amount


def _coordinate_parts(value: dict[str, Any]) -> tuple[_Part, ...]:
    where: tuple[_Part, ...] = (
        (
            False,
            f"latitude {_number_text(value.get('latitude'))}, longitude "
            f"{_number_text(value.get('longitude'))}",
        ),
    )
    globe = _string(value.get("globe"))
    if globe and globe != _EARTH and (code := _code_of(globe)):
        return (*where, (False, " on "), (True, code))
    return where


def _monolingual_parts(value: dict[str, Any]) -> tuple[_Part, ...]:
    language = _string(value.get("language"))
    return ((False, _string(value.get("text")) + (f" ({language})" if language else "")),)


def _time_parts(value: dict[str, Any]) -> tuple[_Part, ...]:
    return ((False, _time_text(value)),)


_VALUE_READERS: dict[str, Callable[[dict[str, Any]], tuple[_Part, ...]]] = {
    "wikibase-entityid": _entity_parts,
    "quantity": _quantity_parts,
    "globecoordinate": _coordinate_parts,
    "monolingualtext": _monolingual_parts,
    "time": _time_parts,
}


def _time_text(value: dict[str, Any]) -> str:
    """A Wikibase time to its precision: ``1952-03-11``, ``1952-03``, ``1952``, or the year and
    how coarse it is (``1001 (to the century)``); the Julian calendar is named."""
    stamp = _string(value.get("time"))
    found = re.match(r"^([+-])(\d+)-(\d\d)-(\d\d)T", stamp)
    if found is None:
        return stamp or "(no time)"
    sign, year, month, day = found.groups()
    year = ("-" if sign == "-" else "") + year.lstrip("0").rjust(4, "0")
    precision = value.get("precision")
    precision = precision if isinstance(precision, int) else 11
    notes = [f"to the {_COARSE[precision]}"] if precision in _COARSE else []
    if _string(value.get("calendarmodel")) == _JULIAN:
        notes.append("Julian calendar")
    text = (
        year
        if precision <= 9
        else f"{year}-{month}"
        if precision == 10
        else f"{year}-{month}-{day}"
    )
    return text + (f" ({', '.join(notes)})" if notes else "")


def _codes(claims: Iterable[_Claim]) -> list[str]:
    """The codes to label for ``claims``: properties, values, units, and the qualifiers'
    properties and values, in order, once each."""
    codes: dict[str, None] = {}
    for claim in claims:
        codes[claim.prop] = None
        parts = [*claim.value, *(part for _, value in claim.qualifiers for part in value)]
        codes.update(dict.fromkeys(text for is_code, text in parts if is_code))
        codes.update(dict.fromkeys(prop for prop, _ in claim.qualifiers))
    return list(codes)


def _labels(codes: list[str], language: str) -> tuple[dict[str, str], str]:
    """The labels of ``codes`` in ``language``, with English where it has none, and a note on
    the codes a failed request left bare (empty when none did).

    Only items and properties are asked (``wbgetentities`` refuses a whole batch for an ID it
    does not hold, such as an EntitySchema's ``E10``): one request per 50 codes, at most
    ``_LABEL_REQUESTS``. Labels are best effort, since the statements are read already: a code
    past the requests, or in a batch the API refuses (a 429, a ``maxlag``), stays bare, which
    ``wikidata_entity`` reads.
    """
    askable = [code for code in codes if _ID_RE.fullmatch(code)]
    languages = language if language == "en" else f"{language}|en"
    labels: dict[str, str] = {}
    bare = 0
    reasons: dict[str, None] = {}
    for start in range(0, min(len(askable), _LABELS_BATCH * _LABEL_REQUESTS), _LABELS_BATCH):
        batch = askable[start : start + _LABELS_BATCH]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": languages,
            "languagefallback": "1",
            "format": "json",
        }
        try:
            labels.update(
                _API.get_json(params=params, parse=lambda data: _label_map(data, language))
            )
        except ToolFailure as failure:
            bare += len(batch)
            reasons[failure.error.message] = None
    note = (
        f"{bare} code{'' if bare == 1 else 's'} without their label; the label request failed: "
        + "; ".join(reasons)
        if bare
        else ""
    )
    return labels, note


def _label_map(data: dict[str, Any], language: str) -> dict[str, str]:
    entities = data.get("entities")
    labels: dict[str, str] = {}
    for code, entity in entities.items() if isinstance(entities, dict) else ():
        if isinstance(entity, dict) and (label := _localized(entity.get("labels"), language)):
            labels[str(code)] = label
    return labels


def _entity_answer(
    entity: _Entity,
    asked: str,
    page: tuple[_Claim, ...],
    labels: Mapping[str, str],
    offset: int,
    unlabelled: str,
) -> ToolResult:
    merged = f" (redirects to {entity.id})" if entity.id != asked else ""
    title = f"Wikidata entity {asked}{merged}: {entity.label or '(no label)'}"
    if not entity.claims:
        return ToolResult.success("\n".join([title, *_details(entity), "Statements: none."]))
    if offset:
        heading = f"{title}, statements:"
    else:
        heading = "\n".join(
            [title, *_details(entity), "Statements (wikidata_entity reads any Q or P code below):"]
        )
    if unlabelled:
        heading += f"\n({unlabelled})"
    lines = [_claim_text(number, claim, labels) for number, claim in enumerate(page, offset + 1)]
    end = offset + len(page)
    next_call = {"offset": end} if page and end < len(entity.claims) else None
    window = list_window(lines, first=offset + 1, total=len(entity.claims), next_call=next_call)
    return window.result(heading=heading)


def _details(entity: _Entity) -> list[str]:
    lines = []
    if entity.description:
        lines.append(f"   description: {entity.description}")
    if entity.aliases:
        lines.append("   aliases: " + ", ".join(entity.aliases))
    if entity.wikipedia_url:
        lines.append(f"   Wikipedia: {entity.wikipedia_url}")
    lines.append(f"   Wikidata: https://www.wikidata.org/wiki/{entity.id}")
    return lines


def _claim_text(number: int, claim: _Claim, labels: Mapping[str, str]) -> str:
    def named(code: str) -> str:
        return f"{labels[code]} ({code})" if code in labels else code

    def text(parts: tuple[_Part, ...]) -> str:
        return "".join(named(part) if is_code else part for is_code, part in parts)

    line = f"{number}. {named(claim.prop)}: {text(claim.value)}"
    for prop, value in claim.qualifiers:
        line += f" ({labels.get(prop, prop)}: {text(value)})"
    return line + (f" [{claim.rank}]" if claim.rank != "normal" else "")


def _localized(values: object, language: str) -> str:
    if not isinstance(values, dict):
        return ""
    for key in (language, "en"):
        item = values.get(key)
        if isinstance(item, dict) and (text := _string(item.get("value"))):
            return text
    for item in values.values():
        if isinstance(item, dict) and (text := _string(item.get("value"))):
            return text
    return ""


def _aliases(values: object, language: str) -> tuple[str, ...]:
    if not isinstance(values, dict):
        return ()
    items = values.get(language) or values.get("en") or []
    if not isinstance(items, list):
        return ()
    return tuple(
        _string(item.get("value"))
        for item in items
        if isinstance(item, dict) and item.get("value")
    )


def _wikipedia_url(sitelinks: object, language: str) -> str:
    if not isinstance(sitelinks, dict):
        return ""
    key = f"{language}wiki"
    item = sitelinks.get(key) or sitelinks.get("enwiki")
    if isinstance(item, dict) and (title := _string(item.get("title"))):
        lang = key.removesuffix("wiki") if key in sitelinks else "en"
        return f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    return ""


def _code_of(uri: str) -> str:
    """The code at the end of an entity URI (``…/entity/Q11573``); empty for anything else."""
    code = uri.removeprefix(_ENTITY_URI)
    return code if code != uri and re.fullmatch(r"[A-Z]\d+", code) else ""


def _number_text(value: object) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return plain(value)
    return decimal_text(_string(value))


# --- SPARQL ------------------------------------------------------------------------------------


def _shape(query: str) -> str:
    """``query``'s syntax, at the same length: comments as spaces, and the text inside strings
    and IRIs as ``_``, so no word or brace in them reads as the query's."""

    def blank(token: re.Match[str]) -> str:
        text = token[0]
        if text.startswith("#"):
            return " " * len(text)
        return text[0] + "_" * (len(text) - 2) + text[-1]

    return _OPAQUE.sub(blank, query)


def _with_limit(query: str, shape: str) -> tuple[str, bool]:
    """The query to send, and whether the tool added its LIMIT: a SELECT whose solution
    modifiers (after the WHERE clause's last ``}``, before a trailing VALUES clause) have none
    gets ``LIMIT 1000``, where the modifiers go."""
    if not re.match(_PREFIXES + r"SELECT\b", shape.lstrip(), re.IGNORECASE):
        return query, False
    values = _TRAILING_VALUES.search(shape)
    end = values.start() if values else len(shape)
    if re.search(r"\bLIMIT\s+\d+", shape[shape.rfind("}", 0, end) + 1 : end], re.IGNORECASE):
        return query, False
    if values is None:
        return f"{query}\nLIMIT {_ROW_CAP}", True
    return f"{query[:end].rstrip()}\nLIMIT {_ROW_CAP}\n{query[end:]}", True


def _sparql_answer(
    data: dict[str, Any], query: str, capped: bool, max_results: int, offset: int
) -> ToolResult:
    if "boolean" in data:
        return ToolResult.success(f"Wikidata SPARQL answer: {str(bool(data['boolean'])).lower()}")
    head = data.get("head", {})
    rows = data.get("results", {}).get("bindings", [])
    variables = head.get("vars", [])
    if not isinstance(variables, list) or not isinstance(rows, list):
        raise ToolFailure(
            "upstream", "the Wikidata Query Service answered without head.vars or bindings"
        )
    if not rows:
        return ToolResult.success(f"No rows for the Wikidata query: {' '.join(query.split())}")
    lines = [_row_text(number, variables, row) for number, row in enumerate(rows, start=1)]
    heading = "Wikidata SPARQL rows:"
    if capped and len(rows) >= _ROW_CAP:
        heading = (
            f"Wikidata SPARQL rows (the query had no LIMIT: the tool added LIMIT {_ROW_CAP}, and "
            "the answer reached it; to read past it, add ORDER BY, LIMIT and OFFSET to the query):"
        )
    return page_window(lines, offset=offset, limit=max_results).result(heading=heading)


def _row_text(number: int, variables: list[Any], row: object) -> str:
    parts = []
    for variable in variables:
        value = row.get(str(variable)) if isinstance(row, dict) else None
        if isinstance(value, dict) and (text := _binding_text(value)):
            parts.append(f"{variable}: {text}")
    return f"{number}. " + (" | ".join(parts) or "(no values)")


def _binding_text(value: dict[str, Any]) -> str:
    """A SPARQL value as the tools read it: a Wikidata entity as its code (which
    ``wikidata_entity`` takes), a number in plain digits, anything else as it is."""
    text = _string(value.get("value"))
    if value.get("type") == "uri":
        return _code_of(text) or text
    if _string(value.get("datatype")).endswith(_NUMERIC_TYPES):
        return decimal_text(text)
    return text


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
