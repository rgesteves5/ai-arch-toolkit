"""RxNorm and DailyMed tools — public medication normalization and labels."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from ai_arch_toolkit.core import tool

_RXNAV_URL = "https://rxnav.nlm.nih.gov/REST"
_DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
_DAILYMED_PAGE_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
_TIMEOUT = 20
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_RXCUI_RE = re.compile(r"^\d{1,12}$")
_NDC_RE = re.compile(r"^[0-9-]{4,20}$")
_SETID_RE = re.compile(r"^[A-Fa-f0-9-]{32,40}$")
_NS = {"v3": "urn:hl7-org:v3"}


@tool
def rxnorm_drug_search(name: str) -> str:
    """Search RxNorm drug concepts by name.

    Args:
        name: Drug name, brand, ingredient, or clinical drug text.
    """
    if not _valid_text(name):
        return "RxNorm drug search failed: invalid name."
    try:
        data = _fetch_rxnav("/drugs.json", {"name": name.strip()})
        groups = data.get("drugGroup", {}).get("conceptGroup", [])
    except urllib.error.HTTPError as e:
        return _http_error("RxNorm drug search failed", "RxNorm", e)
    except urllib.error.URLError as e:
        return f"RxNorm drug search failed: URL error: {e.reason}"
    except TimeoutError:
        return "RxNorm drug search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"RxNorm drug search failed: could not parse API response: {e}"

    concepts = []
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict):
                for concept in group.get("conceptProperties", []) or []:
                    if isinstance(concept, dict):
                        concepts.append((group.get("tty"), concept))
    if not concepts:
        return "No RxNorm drug concepts found."
    lines = [f"RxNorm concepts for {name!r}:"]
    for index, (tty, concept) in enumerate(concepts[:_MAX_LIMIT], start=1):
        concept_name = _string(concept.get("name"))
        rxcui = _string(concept.get("rxcui"))
        lines.append(f"{index}. {concept_name} | RxCUI: {rxcui} | TTY: {_string(tty)}")
    return "\n".join(lines)


@tool
def rxnorm_concept(rxcui: str) -> str:
    """Get RxNorm concept properties by RxCUI.

    Args:
        rxcui: RxNorm concept unique identifier.
    """
    normalized = rxcui.strip()
    if not _RXCUI_RE.fullmatch(normalized):
        return f"RxNorm concept lookup failed: invalid rxcui: {rxcui!r}"
    try:
        data = _fetch_rxnav(f"/rxcui/{normalized}/properties.json", {})
        props = data.get("properties", {})
    except urllib.error.HTTPError as e:
        return _http_error("RxNorm concept lookup failed", "RxNorm", e)
    except urllib.error.URLError as e:
        return f"RxNorm concept lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "RxNorm concept lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"RxNorm concept lookup failed: could not parse API response: {e}"

    if not isinstance(props, dict) or not props:
        return f"RxNorm concept not found: {normalized}"
    lines = [f"RxNorm concept {normalized}:"]
    lines.append(_string(props.get("name")) or "(no name)")
    lines.append(
        "   "
        + " | ".join(
            [
                f"TTY: {_string(props.get('tty')) or '?'}",
                f"language: {_string(props.get('language')) or '?'}",
                f"suppress: {_string(props.get('suppress')) or '?'}",
            ]
        )
    )
    synonym = _string(props.get("synonym"))
    if synonym:
        lines.append(f"   synonym: {synonym}")
    return "\n".join(lines)


@tool
def rxnorm_related(rxcui: str, tty: str = "", max_results: int = 20) -> str:
    """Get related RxNorm concepts.

    Args:
        rxcui: RxNorm concept unique identifier.
        tty: Optional term type filter, e.g. "IN", "BN", "SCD", or "SBD".
        max_results: Number of related concepts to return (1-25). Defaults to 20.
    """
    normalized = rxcui.strip()
    if not _RXCUI_RE.fullmatch(normalized):
        return f"RxNorm related lookup failed: invalid rxcui: {rxcui!r}"
    if tty and not re.fullmatch(r"^[A-Za-z+]{1,80}$", tty.strip()):
        return "RxNorm related lookup failed: invalid tty."
    params = {"tty": tty.strip()} if tty.strip() else {}
    try:
        data = _fetch_rxnav(f"/rxcui/{normalized}/related.json", params)
        groups = data.get("relatedGroup", {}).get("conceptGroup", [])
    except urllib.error.HTTPError as e:
        return _http_error("RxNorm related lookup failed", "RxNorm", e)
    except urllib.error.URLError as e:
        return f"RxNorm related lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "RxNorm related lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"RxNorm related lookup failed: could not parse API response: {e}"

    concepts = []
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict):
                for concept in group.get("conceptProperties", []) or []:
                    if isinstance(concept, dict):
                        concepts.append((group.get("tty"), concept))
    if not concepts:
        return f"No RxNorm related concepts found for {normalized}."
    lines = [f"RxNorm related concepts for {normalized}:"]
    for index, (group_tty, concept) in enumerate(concepts[: _bounded(max_results)], start=1):
        concept_name = _string(concept.get("name"))
        rxcui = _string(concept.get("rxcui"))
        lines.append(f"{index}. {concept_name} | RxCUI: {rxcui} | TTY: {_string(group_tty)}")
    return "\n".join(lines)


@tool
def rxnorm_ndcs(rxcui: str) -> str:
    """List NDC product codes associated with an RxNorm concept.

    Args:
        rxcui: RxNorm concept unique identifier.
    """
    normalized = rxcui.strip()
    if not _RXCUI_RE.fullmatch(normalized):
        return f"RxNorm NDC lookup failed: invalid rxcui: {rxcui!r}"
    try:
        data = _fetch_rxnav(f"/rxcui/{normalized}/ndcs.json", {})
        ndcs = data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
    except urllib.error.HTTPError as e:
        return _http_error("RxNorm NDC lookup failed", "RxNorm", e)
    except urllib.error.URLError as e:
        return f"RxNorm NDC lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "RxNorm NDC lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"RxNorm NDC lookup failed: could not parse API response: {e}"

    if not isinstance(ndcs, list) or not ndcs:
        return f"No RxNorm NDCs found for {normalized}."
    return f"RxNorm NDCs for {normalized}:\n" + "\n".join(
        f"{index}. {_string(ndc)}" for index, ndc in enumerate(ndcs[:_MAX_LIMIT], start=1)
    )


@tool
def dailymed_label_search(
    drug_name: str = "",
    ndc: str = "",
    max_results: int = 10,
    page: int = 1,
) -> str:
    """Search DailyMed SPL drug labels.

    Args:
        drug_name: Optional drug name query.
        ndc: Optional NDC code query.
        max_results: Number of labels to return (1-25). Defaults to 10.
        page: One-based result page. Defaults to 1.
    """
    if not drug_name.strip() and not ndc.strip():
        return "DailyMed label search failed: provide drug_name or ndc."
    if drug_name and not _valid_text(drug_name):
        return "DailyMed label search failed: invalid drug_name."
    if ndc and not _NDC_RE.fullmatch(ndc.strip()):
        return "DailyMed label search failed: invalid ndc."
    if page < 1:
        return "DailyMed label search failed: page must be greater than or equal to 1."
    params = {"page": str(page), "pagesize": str(_bounded(max_results))}
    if drug_name.strip():
        params["drug_name"] = drug_name.strip()
    if ndc.strip():
        params["ndc"] = ndc.strip()
    try:
        data = _fetch_dailymed_json("/spls.json", params)
        items = data.get("data", [])
    except urllib.error.HTTPError as e:
        return _http_error("DailyMed label search failed", "DailyMed", e)
    except urllib.error.URLError as e:
        return f"DailyMed label search failed: URL error: {e.reason}"
    except TimeoutError:
        return "DailyMed label search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"DailyMed label search failed: could not parse API response: {e}"

    if not isinstance(items, list) or not items:
        return "No DailyMed labels found."
    meta = data.get("metadata", {})
    total = _string(meta.get("total_elements")) if isinstance(meta, dict) else "?"
    lines = [f"DailyMed labels (returned {len(items)}, total {total or '?'}, page {page}):"]
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        lines.append(f"{index}. {_string(item.get('title'))}")
        setid = _string(item.get("setid"))
        published = _string(item.get("published_date"))
        lines.append(f"   setid: {setid} | published: {published}")
    return "\n".join(lines)


@tool
def dailymed_label(setid: str, max_sections: int = 12) -> str:
    """Get DailyMed SPL label metadata and section titles by set ID.

    Args:
        setid: DailyMed SPL set ID from dailymed_label_search.
        max_sections: Number of section titles to return (1-25). Defaults to 12.
    """
    normalized = setid.strip()
    if not _SETID_RE.fullmatch(normalized):
        return f"DailyMed label lookup failed: invalid setid: {setid!r}"
    try:
        xml_text = _fetch_dailymed_text(f"/spls/{normalized}.xml", {})
    except urllib.error.HTTPError as e:
        return _http_error("DailyMed label lookup failed", "DailyMed", e)
    except urllib.error.URLError as e:
        return f"DailyMed label lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "DailyMed label lookup failed: request timed out."
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return f"DailyMed label lookup failed: could not parse XML response: {e}"

    title = _xml_text(root.find("v3:title", _NS))
    effective = _xml_attr(root.find("v3:effectiveTime", _NS), "value")
    org = _first_text(root.findall(".//v3:representedOrganization/v3:name", _NS))
    sections = [
        _xml_text(section.find("v3:title", _NS)) for section in root.findall(".//v3:section", _NS)
    ]
    sections = [section for section in sections if section][: _bounded(max_sections)]
    lines = [f"DailyMed label {normalized}:", title or "(no title)"]
    if effective or org:
        lines.append(
            f"   effective: {_format_date(effective) or '?'} | organization: {org or '?'}"
        )
    url = f"{_DAILYMED_PAGE_URL}?setid={urllib.parse.quote(normalized)}"
    lines.append(f"   DailyMed: {url}")
    if sections:
        lines.append("   sections: " + "; ".join(sections))
    return "\n".join(lines)


def _fetch_rxnav(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_RXNAV_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _fetch_json(url)


def _fetch_dailymed_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    return json.loads(_fetch_dailymed_text(path, params))


def _fetch_dailymed_text(path: str, params: dict[str, str]) -> str:
    url = f"{_DAILYMED_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_error(prefix: str, api_name: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by {api_name} (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _xml_text(element: ET.Element[str] | None) -> str:
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def _xml_attr(element: ET.Element[str] | None, name: str) -> str:
    if element is None:
        return ""
    return _string(element.attrib.get(name))


def _first_text(elements: list[ET.Element[str]]) -> str:
    for element in elements:
        text = _xml_text(element)
        if text:
            return text
    return ""


def _format_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
