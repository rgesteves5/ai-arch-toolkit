"""ROR tools — public research organization registry lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://api.ror.org/v2/organizations"
_TIMEOUT = 20
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 20
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_ROR_RE = re.compile(r"^(?:https://ror\.org/)?0[a-z0-9]{8}$", re.IGNORECASE)


@tool
def ror_search(
    query: str,
    country: str = "",
    org_type: str = "",
    max_results: int = 10,
    page: int = 1,
) -> str:
    """Search ROR research organizations.

    Args:
        query: Organization name, acronym, domain, or alias.
        country: Optional ISO2 country filter, e.g. "PT".
        org_type: Optional ROR type filter, e.g. "education", "funder", or "healthcare".
        max_results: Number of organizations to return (1-20). Defaults to 10.
        page: One-based result page. Defaults to 1.
    """
    if not _valid_text(query):
        return "ROR search failed: invalid query."
    if country and not re.fullmatch(r"^[A-Za-z]{2}$", country.strip()):
        return "ROR search failed: invalid country. Use ISO2."
    if org_type and not re.fullmatch(r"^[A-Za-z_-]{1,40}$", org_type.strip()):
        return "ROR search failed: invalid org_type."
    if page < 1:
        return "ROR search failed: page must be greater than or equal to 1."
    params = {"query": query.strip(), "page": str(page)}
    if country.strip():
        params["filter"] = f"country.country_code:{country.strip().lower()}"
    if org_type.strip():
        existing = params.get("filter")
        type_filter = f"types:{org_type.strip().lower()}"
        params["filter"] = f"{existing},{type_filter}" if existing else type_filter
    try:
        data = _fetch_json("", params)
        items = data.get("items", [])
    except urllib.error.HTTPError as e:
        return _http_error("ROR search failed", e)
    except urllib.error.URLError as e:
        return f"ROR search failed: URL error: {e.reason}"
    except TimeoutError:
        return "ROR search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ROR search failed: could not parse API response: {e}"

    if not isinstance(items, list) or not items:
        return "No ROR organizations found."
    total = _string(data.get("number_of_results")) or "?"
    page_items = items[: _bounded(max_results)]
    lines = [
        f"ROR organizations for {query!r} "
        f"(returned {len(page_items)}, total {total}, page {page}):"
    ]
    for index, item in enumerate(page_items, start=1):
        if isinstance(item, dict):
            lines.extend(_format_org(item, index=index, details=False))
    return "\n".join(lines)


@tool
def ror_organization(ror_id: str) -> str:
    """Get ROR organization metadata.

    Args:
        ror_id: ROR ID or URL, e.g. "https://ror.org/01c27hj86".
    """
    normalized = _normalize_ror_id(ror_id)
    if not normalized:
        return f"ROR organization lookup failed: invalid ror_id: {ror_id!r}"
    try:
        data = _fetch_json(f"/{normalized}", {})
    except urllib.error.HTTPError as e:
        return _http_error("ROR organization lookup failed", e)
    except urllib.error.URLError as e:
        return f"ROR organization lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "ROR organization lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ROR organization lookup failed: could not parse API response: {e}"

    lines = [f"ROR organization {normalized}:"]
    lines.extend(_format_org(data, index=None, details=True))
    return "\n".join(lines)


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _format_org(item: dict[str, Any], *, index: int | None, details: bool) -> list[str]:
    prefix = f"{index}. " if index is not None else ""
    name = _display_name(item)
    ror_id = _string(item.get("id"))
    lines = [f"{prefix}{name or '(no display name)'} | id: {ror_id}"]
    location = _location(item)
    types = _list_text(item.get("types"))
    status = _string(item.get("status"))
    lines.append(
        f"   country: {location or '?'} | types: {types or '?'} | status: {status or '?'}"
    )
    domains = _list_text(item.get("domains"))
    if domains:
        lines.append(f"   domains: {domains}")
    links = item.get("links", [])
    website = ""
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and _string(link.get("type")) == "website":
                website = _string(link.get("value"))
                break
    if website:
        lines.append(f"   website: {website}")
    if details:
        aliases = _names(item, wanted_type="alias")
        acronyms = _names(item, wanted_type="acronym")
        if aliases:
            lines.append(f"   aliases: {aliases}")
        if acronyms:
            lines.append(f"   acronyms: {acronyms}")
        relationships = item.get("relationships", [])
        if isinstance(relationships, list) and relationships:
            rels = [
                (
                    f"{_string(rel.get('type'))}: {_string(rel.get('label'))} "
                    f"({_string(rel.get('id'))})"
                )
                for rel in relationships[:10]
                if isinstance(rel, dict)
            ]
            lines.append(f"   relationships: {'; '.join(rels)}")
    return lines


def _display_name(item: dict[str, Any]) -> str:
    names = item.get("names", [])
    if not isinstance(names, list):
        return ""
    for wanted in ("ror_display", "label"):
        for name in names:
            if isinstance(name, dict) and wanted in (name.get("types") or []):
                return _string(name.get("value"))
    return ""


def _names(item: dict[str, Any], *, wanted_type: str) -> str:
    names = item.get("names", [])
    if not isinstance(names, list):
        return ""
    values = [
        _string(name.get("value"))
        for name in names
        if isinstance(name, dict) and wanted_type in (name.get("types") or [])
    ]
    return ", ".join(value for value in values[:10] if value)


def _location(item: dict[str, Any]) -> str:
    locations = item.get("locations", [])
    if not isinstance(locations, list) or not locations:
        return ""
    details = locations[0].get("geonames_details", {}) if isinstance(locations[0], dict) else {}
    return ", ".join(
        value
        for value in (
            _string(details.get("name")),
            _string(details.get("country_name")),
            _string(details.get("country_code")),
        )
        if value
    )


def _normalize_ror_id(value: str) -> str:
    text = value.strip().lower().removeprefix("https://ror.org/")
    return text if _ROR_RE.fullmatch(text) else ""


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by ROR (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _list_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(_string(item) for item in value if _string(item))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
