"""ROR tools: search research organizations, and read one organization's record (T06).

A search gives ``number_of_results`` and a page of 20 organizations, pages 1 to 500 (the first
10,000; https://ror.readme.io/v2/docs/rest-api, and ror-community/ror-api,
rorapi/settings.py: PAGE_SIZE 20, MAX_PAGE 500): the page is the source's, shown whole. An
organization's record gives every name, location, link, external ID and relationship, through
the window (D39).
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._records import DEFAULT_CHARS, MAX_CHARS, record
from ai_arch_toolkit.toolkit.tools._window import list_window


def _ror_error(reply: Reply) -> ToolFailure | str | None:
    """The error a ROR answer reports, from its ``errors`` list; ``None`` without one.

    ROR explains a refused request in ``{"errors": ["..."]}`` (https://ror.readme.io/v2/docs/
    rest-api; the words in ror-community/ror-api, rorapi/common/queries.py), a list the door's
    generic reading does not take. A 400 is a parameter ROR does not accept (the query, or a
    filter built from ``country`` or ``org_type``): ``validation_error``.
    """
    if reply.status < 400 or not isinstance(reply.body, dict):
        return None
    errors = reply.body.get("errors")
    said = "; ".join(" ".join(str(e).split()) for e in errors) if isinstance(errors, list) else ""
    if not said:
        return None
    if reply.status == 400:
        return ToolFailure(
            "validation_error", f"ROR refused the request: {said}; correct that parameter"
        )
    return said


_API = Api(
    base="https://api.ror.org/v2/organizations", name="ROR", timeout_s=20, error_reader=_ror_error
)
_PAGE_SIZE = 20
_MAX_PAGE = 500
_QUERY_CHARS = 200
_ROR_RE = re.compile(r"^(?:https://ror\.org/)?0[a-z0-9]{8}$", re.IGNORECASE)


@tool(capability="network")
def ror_search(
    query: str,
    country: str = "",
    org_type: str = "",
    page: Annotated[int, Range(1, _MAX_PAGE)] = 1,
) -> ToolResult:
    """Search ROR research organizations by name, acronym, alias or domain: ROR's pages of 20,
    numbered, with the total.

    Args:
        query: An organization's name, acronym, alias or domain.
        country: An ISO 3166-1 alpha-2 country code to keep, e.g. "PT".
        org_type: A ROR type to keep, e.g. "education", "funder" or "healthcare".
        page: Which page of 20 to show; the footer gives the next page.

    Raises:
        ToolFailure: validation_error when the query, ``country`` or ``org_type`` is invalid
            (here or for ROR).
    """
    text = query.strip()
    if not text or len(text) > _QUERY_CHARS or not text.isprintable():
        raise ToolFailure(
            "validation_error",
            f"invalid query {query[:100]!r}; give 1-{_QUERY_CHARS} printable characters of an "
            "organization's name, acronym or domain.",
        )
    if country and not re.fullmatch(r"^[A-Za-z]{2}$", country.strip()):
        raise ToolFailure(
            "validation_error",
            f"invalid country {country!r}; give an ISO 3166-1 alpha-2 code, e.g. 'PT'.",
        )
    if org_type and not re.fullmatch(r"^[A-Za-z_-]{1,40}$", org_type.strip()):
        raise ToolFailure(
            "validation_error",
            f"invalid org_type {org_type!r}; give a ROR type such as 'education', 'funder' or "
            "'healthcare'.",
        )
    filters = []
    if country.strip():
        filters.append(f"country.country_code:{country.strip().lower()}")
    if org_type.strip():
        filters.append(f"types:{org_type.strip().lower()}")
    params = {"query": text, "page": str(page)}
    if filters:
        params["filter"] = ",".join(filters)
    return _API.get_json(params=params, parse=lambda data: _search_answer(data, text, page))


@tool(capability="network")
def ror_organization(
    ror_id: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read a ROR organization's record: every name, location, link, external ID and
    relationship.

    Args:
        ror_id: A ROR ID or URL, e.g. "https://ror.org/01c27hj86".
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``ror_id`` is not a ROR ID; not_found when ROR has
            no organization with it.
    """
    normalized = _normalize_ror_id(ror_id)
    if not normalized:
        raise ToolFailure(
            "validation_error",
            f"invalid ror_id {ror_id!r}; a ROR ID is 0 and eight letters or digits, e.g. "
            "'01c27hj86' or 'https://ror.org/01c27hj86'; find one with ror_search.",
        )
    text = _API.get_json(
        normalized,
        parse=_record_text,
        missing=f"ROR has no organization {normalized}; find organizations with ror_search",
    )
    heading = f"ROR organization {normalized}:"
    return record(text, heading=heading, offset=offset, max_chars=max_chars)


def _search_answer(data: dict[str, Any], query: str, page: int) -> ToolResult:
    items = [item for item in data.get("items", []) if isinstance(item, dict)]
    if not items and page == 1:
        return ToolResult.success(f"No ROR organizations match {query!r}.")
    first = (page - 1) * _PAGE_SIZE
    total = data.get("number_of_results")
    total = total if isinstance(total, int) and not isinstance(total, bool) else None
    blocks = [_result_block(first + n, item) for n, item in enumerate(items, start=1)]
    more = total is not None and first + len(items) < total
    next_call = {"page": page + 1} if items and more and page < _MAX_PAGE else None
    window = list_window(blocks, first=first + 1, total=total, next_call=next_call)
    return window.result(heading=f"ROR organizations that match {query!r}:")


def _result_block(number: int, item: dict[str, Any]) -> str:
    lines = [
        f"{number}. {_display_name(item) or '(no display name)'} | id: {_string(item.get('id'))}"
    ]
    places = "; ".join(_place(location, coordinates=False) for location in _locations(item))
    facts = [
        f"locations: {places or '?'}",
        f"types: {_list_text(item.get('types')) or '?'}",
        f"status: {_string(item.get('status')) or '?'}",
    ]
    if established := _string(item.get("established")):
        facts.append(f"established: {established}")
    lines.append("   " + " | ".join(facts))
    if domains := _list_text(item.get("domains")):
        lines.append(f"   domains: {domains}")
    if website := _link(item, "website"):
        lines.append(f"   website: {website}")
    return "\n".join(lines)


def _record_text(item: dict[str, Any]) -> str:
    """The whole record, every list whole: names, locations, links, IDs and relationships."""
    places = [_place(location, coordinates=True) for location in _locations(item)]
    facts = [
        f"id: {_string(item.get('id'))}",
        f"status: {_string(item.get('status')) or '?'}",
        f"types: {_list_text(item.get('types')) or '?'}",
    ]
    if established := _string(item.get("established")):
        facts.append(f"established: {established}")
    lines = [_display_name(item) or "(no display name)", " | ".join(facts)]
    if places:
        lines.append(f"Locations ({len(places)}): {'; '.join(places)}")
    lines.append(_labelled("Names", _names(item)))
    lines.append(_labelled("Domains", [_string(d) for d in item.get("domains") or []]))
    lines.append(_labelled("Links", _links(item)))
    lines.append(_labelled("External IDs", _external_ids(item)))
    relationships = _relationships(item)
    if relationships:
        lines += [f"Relationships ({len(relationships)}):", *(f"- {r}" for r in relationships)]
    lines.append(_admin(item))
    return "\n".join(line for line in lines if line) + "\n"


def _labelled(label: str, items: list[str]) -> str:
    shown = [item for item in items if item]
    return f"{label}: {'; '.join(shown)}" if shown else ""


def _display_name(item: dict[str, Any]) -> str:
    names = [name for name in item.get("names") or [] if isinstance(name, dict)]
    for wanted in ("ror_display", "label"):
        for name in names:
            if wanted in (name.get("types") or []):
                return _string(name.get("value"))
    return ""


def _names(item: dict[str, Any]) -> list[str]:
    """Every name, with its types (ror_display, label, alias, acronym) and its language."""
    found: list[str] = []
    for name in item.get("names") or []:
        if not isinstance(name, dict) or not (value := _string(name.get("value"))):
            continue
        kinds = _list_text(name.get("types"))
        lang = _string(name.get("lang"))
        said = "; ".join(part for part in (kinds, lang) if part)
        found.append(f"{value} ({said})" if said else value)
    return found


def _locations(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        location
        for location in item.get("locations") or []
        if isinstance(location, dict) and isinstance(location.get("geonames_details"), dict)
    ]


def _place(location: dict[str, Any], *, coordinates: bool) -> str:
    """A location: the place, its region and its country with the code; with ``coordinates``,
    the latitude, the longitude and the GeoNames ID."""
    details = location["geonames_details"]
    code = _string(details.get("country_code"))
    country = _string(details.get("country_name"))
    parts = [_string(details.get("name"))]
    if coordinates:
        parts.append(_string(details.get("country_subdivision_name")))
    parts.append(f"{country} ({code})" if country and code else country or code)
    place = ", ".join(part for part in parts if part)
    if not coordinates:
        return place
    lat, lng = details.get("lat"), details.get("lng")
    if isinstance(lat, int | float) and isinstance(lng, int | float):
        place += f", {_degrees(lat)}, {_degrees(lng)}"
    geonames = _string(location.get("geonames_id"))
    return f"{place} (GeoNames {geonames})" if geonames else place


def _degrees(value: float) -> str:
    """A coordinate in decimal degrees, without scientific notation."""
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _link(item: dict[str, Any], kind: str) -> str:
    for link in item.get("links") or []:
        if isinstance(link, dict) and _string(link.get("type")) == kind:
            return _string(link.get("value"))
    return ""


def _links(item: dict[str, Any]) -> list[str]:
    return [
        f"{_string(link.get('type'))} {value}".strip()
        for link in item.get("links") or []
        if isinstance(link, dict) and (value := _string(link.get("value")))
    ]


def _external_ids(item: dict[str, Any]) -> list[str]:
    """Each external ID (GRID, ISNI, Wikidata, FundRef), the preferred one marked."""
    found: list[str] = []
    for entry in item.get("external_ids") or []:
        if not isinstance(entry, dict):
            continue
        kind, preferred = _string(entry.get("type")), _string(entry.get("preferred"))
        for value in entry.get("all") or []:
            text = _string(value)
            mark = " (preferred)" if text and text == preferred else ""
            found.append(f"{kind} {text}{mark}".strip() if text else "")
    return found


def _relationships(item: dict[str, Any]) -> list[str]:
    return [
        f"{_string(rel.get('type'))}: {_string(rel.get('label'))} ({_string(rel.get('id'))})"
        for rel in item.get("relationships") or []
        if isinstance(rel, dict)
    ]


def _admin(item: dict[str, Any]) -> str:
    admin = item.get("admin")
    stamps = admin if isinstance(admin, dict) else {}
    dates = [
        f"{label} {date}"
        for label, key in (("created", "created"), ("last modified", "last_modified"))
        if isinstance(stamp := stamps.get(key), dict) and (date := _string(stamp.get("date")))
    ]
    return f"Record: {', '.join(dates)}" if dates else ""


def _normalize_ror_id(value: str) -> str:
    """The ID of a ROR ID or URL (``https://``, ``http://`` or no scheme); empty when it is not
    one."""
    text = value.strip().lower().removeprefix("https://").removeprefix("http://")
    text = text.removeprefix("ror.org/")
    return text if _ROR_RE.fullmatch(text) else ""


def _list_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(_string(item) for item in value if _string(item))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
