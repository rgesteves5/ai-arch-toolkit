"""RxNorm and DailyMed tools — public medication normalization and labels."""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_RXNAV = Api(
    base="https://rxnav.nlm.nih.gov/REST",
    name="RxNorm",
    timeout_s=20,
    status_messages={404: "no matching records found."},
)
_DAILYMED = Api(
    base="https://dailymed.nlm.nih.gov/dailymed/services/v2",
    name="DailyMed",
    timeout_s=20,
    status_messages={404: "no matching records found."},
)
_DAILYMED_PAGE_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_RXCUI_RE = re.compile(r"^\d{1,12}$")
_NDC_RE = re.compile(r"^[0-9-]{4,20}$")
_SETID_RE = re.compile(r"^[A-Fa-f0-9-]{32,40}$")
_NS = {"v3": "urn:hl7-org:v3"}


@tool(capability="network")
def rxnorm_drug_search(name: str) -> str:
    """Search RxNorm drug concepts by name.

    Args:
        name: Drug name, brand, ingredient, or clinical drug text.

    Raises:
        ToolFailure: validation_error when ``name`` is empty, too long or has characters the
            search does not take.
    """
    if not _valid_text(name):
        raise ToolFailure(
            "validation_error",
            f"invalid name {name!r}; give 1-180 characters of a drug, brand or ingredient name, "
            "e.g. 'ibuprofen'.",
        )
    return _RXNAV.get_json(
        "drugs.json",
        params={"name": name.strip()},
        parse=lambda data: _concepts_text(
            data.get("drugGroup", {}).get("conceptGroup", []),
            header=f"RxNorm concepts for {name!r}:",
            nothing="No RxNorm drug concepts found.",
            limit=_MAX_LIMIT,
        ),
    )


@tool(capability="network")
def rxnorm_concept(rxcui: str) -> str:
    """Get RxNorm concept properties by RxCUI.

    Args:
        rxcui: RxNorm concept unique identifier.

    Raises:
        ToolFailure: validation_error when ``rxcui`` is not 1-12 digits; not_found when RxNorm
            has no concept with it.
    """
    normalized = _rxcui(rxcui)
    return _RXNAV.get_json(
        "rxcui",
        normalized,
        "properties.json",
        parse=lambda data: _concept_text(data, normalized),
    )


@tool(capability="network")
def rxnorm_related(rxcui: str, tty: str = "", max_results: int = 20) -> str:
    """Get related RxNorm concepts.

    Args:
        rxcui: RxNorm concept unique identifier.
        tty: Optional term type filter, e.g. "IN", "BN", "SCD", or "SBD".
        max_results: Number of related concepts to return (1-25). Defaults to 20.

    Raises:
        ToolFailure: validation_error when ``rxcui`` is not 1-12 digits or ``tty`` is not a
            term type.
    """
    normalized = _rxcui(rxcui)
    if tty and not re.fullmatch(r"^[A-Za-z+]{1,80}$", tty.strip()):
        raise ToolFailure(
            "validation_error",
            f"invalid tty {tty!r}; give RxNorm term types such as 'IN', 'BN', 'SCD' or 'SBD', "
            "joined with '+'.",
        )
    params = {"tty": tty.strip()} if tty.strip() else {}
    return _RXNAV.get_json(
        "rxcui",
        normalized,
        "related.json",
        params=params,
        parse=lambda data: _concepts_text(
            data.get("relatedGroup", {}).get("conceptGroup", []),
            header=f"RxNorm related concepts for {normalized}:",
            nothing=f"No RxNorm related concepts found for {normalized}.",
            limit=_bounded(max_results),
        ),
    )


@tool(capability="network")
def rxnorm_ndcs(rxcui: str) -> str:
    """List NDC product codes associated with an RxNorm concept.

    Args:
        rxcui: RxNorm concept unique identifier.

    Raises:
        ToolFailure: validation_error when ``rxcui`` is not 1-12 digits.
    """
    normalized = _rxcui(rxcui)
    return _RXNAV.get_json(
        "rxcui", normalized, "ndcs.json", parse=lambda data: _ndcs_text(data, normalized)
    )


@tool(capability="network")
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

    Raises:
        ToolFailure: validation_error when neither ``drug_name`` nor ``ndc`` is given, one of
            them is invalid, or ``page`` is below 1.
    """
    if not drug_name.strip() and not ndc.strip():
        raise ToolFailure("validation_error", "nothing to search; provide drug_name or ndc.")
    if drug_name and not _valid_text(drug_name):
        raise ToolFailure(
            "validation_error",
            f"invalid drug_name {drug_name!r}; give 1-180 characters of a drug name.",
        )
    if ndc and not _NDC_RE.fullmatch(ndc.strip()):
        raise ToolFailure(
            "validation_error",
            f"invalid ndc {ndc!r}; an NDC is 4-20 digits and dashes, e.g. '0002-4462-30'.",
        )
    if page < 1:
        raise ToolFailure(
            "validation_error", f"page must be greater than or equal to 1 (got {page})."
        )
    params = {"page": str(page), "pagesize": str(_bounded(max_results))}
    if drug_name.strip():
        params["drug_name"] = drug_name.strip()
    if ndc.strip():
        params["ndc"] = ndc.strip()
    return _DAILYMED.get_json(
        "spls.json", params=params, parse=lambda data: _labels_text(data, page)
    )


@tool(capability="network")
def dailymed_label(setid: str, max_sections: int = 12) -> str:
    """Get DailyMed SPL label metadata and section titles by set ID.

    Args:
        setid: DailyMed SPL set ID from dailymed_label_search.
        max_sections: Number of section titles to return (1-25). Defaults to 12.

    Raises:
        ToolFailure: validation_error when ``setid`` is not a DailyMed set ID.
    """
    normalized = setid.strip()
    if not _SETID_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid setid {setid!r}; a set ID is a UUID; find one with dailymed_label_search.",
        )
    return _DAILYMED.get_text(
        "spls",
        f"{normalized}.xml",
        parse=lambda xml_text: _label_text(xml_text, normalized, max_sections),
    )


def _rxcui(rxcui: str) -> str:
    """``rxcui`` stripped, or a validation_error when it is not an RxCUI."""
    normalized = rxcui.strip()
    if not _RXCUI_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid rxcui {rxcui!r}; an RxCUI is 1-12 digits; find one with rxnorm_drug_search.",
        )
    return normalized


def _concepts_text(groups: Any, *, header: str, nothing: str, limit: int) -> str:
    concepts = _concepts(groups)
    if not concepts:
        return nothing
    lines = [header]
    for index, (tty, concept) in enumerate(concepts[:limit], start=1):
        concept_name = _string(concept.get("name"))
        rxcui = _string(concept.get("rxcui"))
        lines.append(f"{index}. {concept_name} | RxCUI: {rxcui} | TTY: {_string(tty)}")
    return "\n".join(lines)


def _concepts(groups: Any) -> list[tuple[Any, dict[str, Any]]]:
    concepts = []
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict):
                for concept in group.get("conceptProperties", []) or []:
                    if isinstance(concept, dict):
                        concepts.append((group.get("tty"), concept))
    return concepts


def _concept_text(data: dict[str, Any], rxcui: str) -> str:
    props = data.get("properties", {})
    if not isinstance(props, dict) or not props:
        raise ToolFailure(
            "not_found", f"no RxNorm concept with RxCUI {rxcui}; search with rxnorm_drug_search."
        )
    lines = [f"RxNorm concept {rxcui}:"]
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


def _ndcs_text(data: dict[str, Any], rxcui: str) -> str:
    ndcs = data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
    if not isinstance(ndcs, list) or not ndcs:
        return f"No RxNorm NDCs found for {rxcui}."
    return f"RxNorm NDCs for {rxcui}:\n" + "\n".join(
        f"{index}. {_string(ndc)}" for index, ndc in enumerate(ndcs[:_MAX_LIMIT], start=1)
    )


def _labels_text(data: dict[str, Any], page: int) -> str:
    items = data.get("data", [])
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


def _label_text(xml_text: str, setid: str, max_sections: int) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise HttpError(f"could not parse XML response: {e}") from e
    title = _xml_text(root.find("v3:title", _NS))
    effective = _xml_attr(root.find("v3:effectiveTime", _NS), "value")
    org = _first_text(root.findall(".//v3:representedOrganization/v3:name", _NS))
    sections = [
        _xml_text(section.find("v3:title", _NS)) for section in root.findall(".//v3:section", _NS)
    ]
    sections = [section for section in sections if section][: _bounded(max_sections)]
    lines = [f"DailyMed label {setid}:", title or "(no title)"]
    if effective or org:
        lines.append(
            f"   effective: {_format_date(effective) or '?'} | organization: {org or '?'}"
        )
    url = f"{_DAILYMED_PAGE_URL}?setid={urllib.parse.quote(setid)}"
    lines.append(f"   DailyMed: {url}")
    if sections:
        lines.append("   sections: " + "; ".join(sections))
    return "\n".join(lines)


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
