"""GBIF tools: resolve a scientific name, search taxa, read a taxon, and search occurrence
records (T07b; D39).

The match service resolves scientific names only: its ``name`` is "the scientific name to fuzzy
match against" (https://techdocs.gbif.org/en/openapi/v1/species; the source,
https://github.com/gbif/matching-ws, ``MatchV1Controller``). The species search covers "the
scientific and vernacular names" (https://github.com/gbif/checklistbank, ``SpeciesResource``), so
it is where a common name goes. Lists page by ``limit`` and ``offset`` with a ``count``; species
search takes offsets up to 100,000, occurrence search up to 100,000 records in all, 300 a page
(https://github.com/gbif/occurrence, ``OccurrenceSearchResource``). A refused request answers
400 with its reason as plain text (https://github.com/gbif/gbif-common-ws,
``IllegalArgumentExceptionMapper``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, NoReturn

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import list_window


def _gbif_error(reply: Reply) -> ToolFailure | None:
    """A 400's reason, which GBIF sends as plain text (or in a JSON ``message``):
    ``validation_error`` in GBIF's words; ``None`` for anything else."""
    if reply.status != 400:
        return None
    body = reply.body
    said = (
        (_string(body.get("message")) or _string(body.get("error")))
        if isinstance(body, dict)
        else _string(body)
    )
    if not said or said.startswith("<"):  # an HTML page explains nothing
        return None
    return ToolFailure(
        "validation_error", f"GBIF refused the request: {said.rstrip('.')}; change what it names"
    )


_API = Api(base="https://api.gbif.org/v1", name="GBIF", timeout_s=15, error_reader=_gbif_error)
# Species search refuses only an offset past 100,000 (``SpeciesResource.checkDeepPaging``,
# ``DEEP_PAGING_OFFSET_LIMIT``); occurrence search serves offset + limit up to 100,000
# (``OccurrenceSearchResource``).
_SPECIES_DEPTH = 100_000
_OCCURRENCE_DEPTH = 100_000
_SPECIES_REST = (
    f"GBIF's species search pages no further than offset {_SPECIES_DEPTH}; narrow the query, "
    "the rank or the higher taxon for the rest"
)
_OCCURRENCE_REACHED = f"GBIF's search reaches the first {_OCCURRENCE_DEPTH}"
_OCCURRENCE_REST = "narrow the filters, or use GBIF's download service, for the rest"
# GBIF reads a taxon key as a 32-bit integer (``@PathVariable int usageKey``).
_MAX_KEY = 2**31 - 1
_NAME_CHARS = 300
_KEY_RE = re.compile(r"\d{1,10}")
_CODE_RE = re.compile(r"[A-Za-z_ -]{0,80}")
_KINGDOM_RE = re.compile(r"[A-Za-z ]{1,80}")
_RANK_HINT = "pass a rank name such as SPECIES, GENUS or FAMILY"
_KEY_HINT = "a GBIF taxon key is a number; gbif_species_match finds it"
_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


@tool(capability="network")
def gbif_species_match(name: str, rank: str = "", kingdom: str = "") -> str:
    """Resolve a scientific name to the GBIF backbone taxon it names, fuzzily: its key, rank,
    status and classification. For a common name, use gbif_species_search.

    Args:
        name: A scientific name, with or without its authorship, e.g. "Puma concolor".
        rank: The name's rank, to tell homonyms apart, e.g. "species" or "genus".
        kingdom: The name's kingdom, to tell homonyms apart, e.g. "Animalia" or "Plantae".

    Raises:
        ToolFailure: validation_error when an argument is malformed; not_found when the
            backbone has no single taxon for the name.
    """
    text = _free_text("name", name, example="'Puma concolor'")
    params = {"name": text}
    if rank.strip():
        params["rank"] = _rank(rank)
    if kingdom.strip():
        if not _KINGDOM_RE.fullmatch(kingdom.strip()):
            _invalid(f"invalid kingdom {kingdom[:100]!r}; pass a kingdom name, e.g. 'Animalia'")
        params["kingdom"] = kingdom.strip()
    return _API.get_json(
        "species", "match", params=params, parse=lambda data: _match_text(data, text)
    )


@tool(capability="network")
def gbif_species_search(
    query: str,
    rank: str = "",
    highertaxon_key: str = "",
    max_results: Annotated[int, Range(1, 50)] = 10,
    offset: Annotated[int, Range(0, _SPECIES_DEPTH)] = 0,
) -> ToolResult:
    """Search GBIF taxa by scientific or common name, across GBIF's checklists.

    Args:
        query: A name or part of one, scientific or common, e.g. "Puma" or "cougar".
        rank: Only taxa of this rank, e.g. "SPECIES", "GENUS" or "FAMILY".
        highertaxon_key: Only taxa under this taxon key.
        max_results: How many taxa to list.
        offset: How many taxa to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when an argument is malformed or GBIF refuses it.
    """
    text = _free_text("query", query, example="'Puma'")
    params = {"q": text, "limit": str(max_results), "offset": str(offset)}
    if rank.strip():
        params["rank"] = _rank(rank)
    if highertaxon_key.strip():
        params["highertaxonKey"] = _key("highertaxon_key", highertaxon_key)

    def read(data: dict[str, Any]) -> ToolResult:
        return _list_answer(
            data,
            offset,
            lambda number, item: _taxon_line(number, item, text),
            reach=_Reach(last_offset=_SPECIES_DEPTH, rest=_SPECIES_REST),
            heading=f"GBIF taxa that match {text!r} (scientific and common names):",
            nothing=f"No GBIF taxa match {text!r}.",
        )

    return _API.get_json("species", "search", params=params, parse=read)


@tool(capability="network")
def gbif_species(taxon_key: str) -> str:
    """Read a GBIF taxon by its key: name, rank, status, common name, classification, parent.

    Args:
        taxon_key: A GBIF taxon key, from gbif_species_match or gbif_species_search.

    Raises:
        ToolFailure: validation_error when ``taxon_key`` is not a GBIF key; not_found when
            GBIF has no taxon with it.
    """
    key = _key("taxon_key", taxon_key)
    return _API.get_json(
        "species",
        key,
        parse=lambda data: _taxon_text(data, key),
        missing=f"GBIF has no taxon {key}; find its key with gbif_species_match",
    )


@tool(capability="network")
def gbif_occurrence_search(
    taxon_key: str = "",
    country: str = "",
    year: str = "",
    has_coordinate: bool = True,
    max_results: Annotated[int, Range(1, 50)] = 10,
    offset: Annotated[int, Range(0, _OCCURRENCE_DEPTH - 1)] = 0,
) -> ToolResult:
    """Search GBIF occurrence records (observations and specimens) of a taxon, a country or a
    year.

    Args:
        taxon_key: Only records of this GBIF backbone taxon and the taxa under it: a key from
            gbif_species_match, or a search hit's backbone key.
        country: Only records from this ISO 3166-1 alpha-2 country, e.g. "PT".
        year: Only records of this year or range of years, e.g. "2020" or "2020,2024".
        has_coordinate: Only georeferenced records; false for records with or without
            coordinates.
        max_results: How many records to list.
        offset: How many records to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when an option is malformed, none of ``taxon_key``,
            ``country`` and ``year`` is given, or GBIF refuses the filters.
    """
    filters = _occurrence_filters(taxon_key, country, year)
    limit, reach = _occurrence_page(offset, max_results)
    params = {"limit": str(limit), "offset": str(offset), **filters}
    if has_coordinate:
        params["hasCoordinate"] = "true"
    label = _occurrence_label(filters, has_coordinate=has_coordinate)

    def read(data: dict[str, Any]) -> ToolResult:
        return _list_answer(
            data,
            offset,
            _occurrence_line,
            reach=reach,
            heading=f"GBIF occurrences of {label}:",
            nothing=f"No GBIF occurrences of {label}.",
        )

    return _API.get_json("occurrence", "search", params=params, parse=read)


# --- Arguments ---------------------------------------------------------------------------------


def _invalid(message: str) -> NoReturn:
    raise ToolFailure("validation_error", message)


def _free_text(name: str, value: str, *, example: str) -> str:
    """``value`` stripped: refused only when empty, too long or holding a control character."""
    text = value.strip()
    if not text or len(text) > _NAME_CHARS or any(ord(char) < 32 for char in text):
        _invalid(
            f"invalid {name} {value[:100]!r}; give 1 to {_NAME_CHARS} characters on one line, "
            f"e.g. {example}"
        )
    return text


def _rank(rank: str) -> str:
    if not _CODE_RE.fullmatch(rank.strip()):
        _invalid(f"invalid rank {rank[:100]!r}; {_RANK_HINT}")
    return rank.strip().upper()


def _key(name: str, value: str) -> str:
    """A taxon key as GBIF takes it: digits, at most a 32-bit integer."""
    key = value.strip()
    if not _KEY_RE.fullmatch(key) or int(key) > _MAX_KEY:
        _invalid(f"invalid {name} {value[:100]!r}; {_KEY_HINT}")
    return key


def _occurrence_filters(taxon_key: str, country: str, year: str) -> dict[str, str]:
    """GBIF's filters, by its parameter names; at least one is required."""
    if not any((taxon_key.strip(), country.strip(), year.strip())):
        _invalid("provide taxon_key, country, or year to narrow the occurrences")
    if country.strip() and not re.fullmatch(r"[A-Za-z]{2}", country.strip()):
        _invalid(f"invalid country code {country[:100]!r}; use ISO 3166-1 alpha-2, e.g. 'PT'")
    if year.strip() and not re.fullmatch(r"\d{4}(,\d{4})?", year.strip()):
        _invalid(f"invalid year {year[:100]!r}; use YYYY or YYYY,YYYY, e.g. '2020,2024'")
    filters = {
        "taxonKey": _key("taxon_key", taxon_key) if taxon_key.strip() else "",
        "country": country.strip().upper(),
        "year": year.strip(),
    }
    return {name: value for name, value in filters.items() if value}


def _occurrence_page(offset: int, max_results: int) -> tuple[int, _Reach]:
    """The limit to ask for at ``offset``, and how deep the search reaches.

    GBIF serves offset + limit up to 100,000: the page that reaches that depth asks for fewer
    than ``max_results``, and its footer says so.
    """
    limit = min(max_results, _OCCURRENCE_DEPTH - offset)
    short = f" (max_results={max_results} stops there: {limit} on this page)"
    reach = _Reach(
        last_offset=_OCCURRENCE_DEPTH - 1,
        rest=f"{_OCCURRENCE_REACHED}{short if limit < max_results else ''}; {_OCCURRENCE_REST}",
        heading=f"{_OCCURRENCE_REACHED}; {_OCCURRENCE_REST}",
    )
    return limit, reach


def _occurrence_label(filters: dict[str, str], *, has_coordinate: bool) -> str:
    """The filters as answers name them: ``taxon 2435099, country PT, years 2020-2024``."""
    years = filters.get("year", "")
    parts = [
        f"taxon {filters['taxonKey']}" if "taxonKey" in filters else "",
        f"country {filters['country']}" if "country" in filters else "",
        ("years " + years.replace(",", "-") if "," in years else f"year {years}") if years else "",
        "with coordinates" if has_coordinate else "",
    ]
    return ", ".join(part for part in parts if part)


# --- Answers -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _Reach:
    """How deep a GBIF search pages, and what to do for the results past it.

    Attributes:
        last_offset: The largest offset the search takes.
        rest: What the footer says when the next page is past ``last_offset``.
        heading: What the heading says when there are more results than that; ``rest`` when
            empty.
    """

    last_offset: int
    rest: str
    heading: str = ""


def _list_answer(
    data: dict[str, Any],
    offset: int,
    line: Callable[[int, dict[str, Any]], str],
    *,
    reach: _Reach,
    heading: str,
    nothing: str,
) -> ToolResult:
    """A page of a GBIF list, numbered from ``offset``, with its ``count``; past
    ``reach.last_offset`` the heading and the footer say where the rest is."""
    items = [item for item in data.get("results") or [] if isinstance(item, dict)]
    if not items and offset == 0:
        return ToolResult.success(nothing)
    count = data.get("count")
    total = count if isinstance(count, int) and not isinstance(count, bool) else None
    lines = [line(offset + number, item) for number, item in enumerate(items, start=1)]
    shown = offset + len(lines)
    more = data.get("endOfRecords") is False or (total is not None and shown < total)
    next_call = {"offset": shown} if lines and more and shown <= reach.last_offset else None
    if total is not None and total > reach.last_offset:
        heading = heading.removesuffix(":") + f" ({reach.heading or reach.rest}):"
    window = list_window(lines, first=offset + 1, total=total, next_call=next_call)
    return replace(window, rest=reach.rest).result(heading=heading)


def _match_text(data: dict[str, Any], name: str) -> str:
    key = _string(data.get("usageKey"))
    kind = _string(data.get("matchType"))
    if not key or kind == "NONE":
        raise ToolFailure("not_found", _no_match(name, _string(data.get("note"))))
    rank = _string(data.get("rank"))
    confidence = _string(data.get("confidence"))
    lines = [
        f"GBIF match for {name!r}:",
        " | ".join(
            [
                _string(data.get("scientificName")) or _string(data.get("canonicalName")) or "?",
                f"key {key}",
                rank or "?",
                _string(data.get("status")) or "?",
            ]
        ),
        f"Match: {_match_kind(kind, name, rank)}"
        + (f", confidence {confidence}" if confidence else ""),
    ]
    accepted = _string(data.get("acceptedUsageKey"))
    if accepted and accepted != key:
        lines.append(f"Synonym of the taxon with key {accepted} (read it with gbif_species)")
    if note := _string(data.get("note")):
        lines.append(f"Note: {note}")
    if classification := _classification(data):
        lines.append(f"Classification: {classification}")
    return "\n".join(lines)


def _no_match(name: str, note: str) -> str:
    if note:
        return (
            f"GBIF's backbone has no single scientific name matching {name!r} ({note}); give "
            "the kingdom or the authorship to choose one, or search with "
            f"gbif_species_search(query={name!r})"
        )
    return (
        f"GBIF's backbone has no scientific name matching {name!r}; for a common name, search "
        f"with gbif_species_search(query={name!r})"
    )


def _match_kind(kind: str, name: str, rank: str) -> str:
    """What a ``matchType`` means: EXACT, FUZZY (another spelling) or HIGHERRANK (the name is
    not in the backbone, only a taxon above it)."""
    if kind == "FUZZY":
        return "fuzzy (the spelling differs)"
    if kind == "HIGHERRANK":
        return f"higher rank only ({name!r} is not in the backbone; this is its {rank.lower()})"
    return kind.lower() or "?"


def _taxon_line(number: int, item: dict[str, Any], query: str) -> str:
    """A taxon a search found; one from another checklist than the backbone names its backbone
    key too, the one occurrence search takes."""
    key, backbone = _string(item.get("key")), _string(item.get("nubKey"))
    parts = [
        f"{number}. {_string(item.get('scientificName')) or '?'}",
        f"key {key or '?'}" + (f" (backbone key {backbone})" if backbone not in {"", key} else ""),
        _string(item.get("rank")) or "?",
        _string(item.get("taxonomicStatus")) or "?",
        _classification(item, ranks=_RANKS[:-1]),
    ]
    if names := _common_names(item.get("vernacularNames"), query):
        parts.append(f"common names: {names}")
    return " | ".join(part for part in parts if part)


def _common_names(names: object, query: str) -> str:
    """The common names that hold the query (why a search by one matched), each spelling
    once."""
    wanted = query.casefold()
    found: dict[str, str] = {}
    for entry in names if isinstance(names, list) else []:
        name = _string(entry.get("vernacularName")) if isinstance(entry, dict) else ""
        if wanted in name.casefold():
            found.setdefault(name.casefold(), name)
    return ", ".join(found.values())


def _taxon_text(data: dict[str, Any], key: str) -> str:
    lines = [
        f"GBIF taxon {key}:",
        " | ".join(
            [
                _string(data.get("scientificName")) or _string(data.get("canonicalName")) or "?",
                _string(data.get("rank")) or "?",
                _string(data.get("taxonomicStatus")) or "?",
            ]
        ),
    ]
    facts = (
        ("Common name", _string(data.get("vernacularName"))),
        ("Classification", _classification(data)),
        ("Parent", _related(data.get("parent"), data.get("parentKey"))),
        ("Accepted name", _related(data.get("accepted"), data.get("acceptedKey"), key)),
        ("Published in", _string(data.get("publishedIn"))),
    )
    lines.extend(f"{label}: {value}" for label, value in facts if value)
    return "\n".join(lines)


def _related(name: object, key: object, own: str = "") -> str:
    """``Puma (key 2435098)``; empty without a key, or when it is the taxon's own."""
    related = _string(key)
    if not related or related == own:
        return ""
    return f"{_string(name) or '?'} (key {related})"


def _occurrence_line(number: int, item: dict[str, Any]) -> str:
    place = ", ".join(
        part
        for part in (
            _string(item.get("locality")),
            _string(item.get("stateProvince")),
            _string(item.get("country")),
        )
        if part
    )
    latitude, longitude = item.get("decimalLatitude"), item.get("decimalLongitude")
    coordinates = (
        f"{_decimal(latitude)}, {_decimal(longitude)}"
        if latitude is not None and longitude is not None
        else ""
    )
    key = _string(item.get("key"))
    parts = [
        f"{number}. {_string(item.get('scientificName')) or _string(item.get('species')) or '?'}",
        _string(item.get("eventDate")) or _string(item.get("year")),
        place,
        coordinates,
        _string(item.get("basisOfRecord")).replace("_", " ").lower(),
        _string(item.get("datasetName")),
        f"https://www.gbif.org/occurrence/{key}" if key else "",
    ]
    return " | ".join(part for part in parts if part)


def _classification(data: dict[str, Any], ranks: tuple[str, ...] = _RANKS) -> str:
    return " > ".join(value for value in (_string(data.get(rank)) for rank in ranks) if value)


def _decimal(value: object) -> str:
    """A number as written, never in scientific notation (``-1e-05`` is ``-0.00001``)."""
    try:
        return format(Decimal(str(value)), "f")
    except (InvalidOperation, ValueError):
        return _string(value)


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
