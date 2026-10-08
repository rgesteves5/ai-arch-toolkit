"""RxNorm and DailyMed: drug concepts, their NDCs and labels, and a label's text (T07; D39, D41).

RxNav answers a concept's lists whole, without pages
(https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html): the tools page them through the window,
with the total. DailyMed pages its label search (``page``, ``pagesize``,
``metadata.total_elements``) and serves a label as its SPL document
(https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm), whose sections
``dailymed_label`` lists and ``dailymed_label_text`` reads (``_spl``).
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._spl import SplLabel, SplSection, spl_label
from ai_arch_toolkit.toolkit.tools._window import (
    Window,
    find_window,
    list_window,
    page_window,
    text_window,
)


def _rxnav_error(reply: Reply) -> ToolFailure | None:
    """RxNav answers a call it cannot process for its parameters with HTTP 400 "Bad Request"
    (https://lhncbc.nlm.nih.gov/RxNav/news/API-Changes-202107.html), with its reason when the
    body gives one."""
    if reply.status != 400:
        return None
    said = _said(reply.body)
    return ToolFailure(
        "validation_error",
        f"RxNorm could not process the request (HTTP 400, {said or 'invalid parameters'}); check "
        "the arguments: an RxCUI from rxnorm_drug_search, term types such as IN, BN, SCD",
    )


def _said(body: object) -> str:
    """An error body's words: a JSON object's ``message`` or ``error``, or a text that is not a
    page; empty when it gives none."""
    if isinstance(body, dict):
        words = next((body[key] for key in ("message", "error") if body.get(key)), "")
        return _string(words)[:300] if isinstance(words, str) else ""
    text = _string(body) if isinstance(body, str) else ""
    return "" if text.startswith(("<", "{", "[")) else text[:300]


# A call that names one concept or one label declares it (``missing=``): a 404 there is that
# record missing. A 404 on a search is the endpoint gone, which the door reports as such. RxNav
# takes at most 20 requests a second from an address
# (https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html). DailyMed reports errors by status
# alone: 404, 415 and 5xx (https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm).
_RXNAV = Api(
    base="https://rxnav.nlm.nih.gov/REST",
    name="RxNorm",
    timeout_s=20,
    min_interval_s=0.05,
    error_reader=_rxnav_error,
)
_DAILYMED = Api(
    base="https://dailymed.nlm.nih.gov/dailymed/services/v2", name="DailyMed", timeout_s=20
)
_DAILYMED_PAGE_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
_PAGE_MAX = 25
_OUTLINE_PAGE = 50
_MAX_CHARS = 20_000
_DEFAULT_CHARS = 6_000
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_RXCUI_RE = re.compile(r"^\d{1,12}$")
_NDC_RE = re.compile(r"^[0-9-]{4,20}$")
_SETID_RE = re.compile(r"^[A-Fa-f0-9-]{32,40}$")
_TTY_RE = re.compile(r"^[A-Za-z]{1,10}([\s,+]+[A-Za-z]{1,10}){0,19}$")
# DailyMed writes ``published_date`` with English month names (``Jun 10, 2026``), whatever the
# locale of the process that reads it.
_PUBLISHED_RE = re.compile(r"^([A-Za-z]{3})\w* (\d{1,2}), (\d{4})$")
_MONTHS = {
    name: number
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
# RxNorm's term types (https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix5.html).
_TTY_NAMES = {
    "IN": "Ingredient",
    "PIN": "Precise Ingredient",
    "MIN": "Multiple Ingredients",
    "SCDC": "Semantic Clinical Drug Component",
    "SCDF": "Semantic Clinical Drug Form",
    "SCDFP": "Semantic Clinical Drug Form Precise",
    "SCDG": "Semantic Clinical Drug Group",
    "SCDGP": "Semantic Clinical Drug Form Group Precise",
    "SCD": "Semantic Clinical Drug",
    "GPCK": "Generic Pack",
    "BN": "Brand Name",
    "SBDC": "Semantic Branded Drug Component",
    "SBDF": "Semantic Branded Drug Form",
    "SBDFP": "Semantic Branded Drug Form Precise",
    "SBDG": "Semantic Branded Drug Group",
    "SBD": "Semantic Branded Drug",
    "BPCK": "Brand Name Pack",
    "DF": "Dose Form",
    "DFG": "Dose Form Group",
    "PSN": "Prescribable Name",
    "SY": "Synonym",
    "TMSY": "Tall Man Lettering Synonym",
}


@tool(capability="network")
def rxnorm_drug_search(
    name: str,
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = _PAGE_MAX,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search RxNorm drug concepts by name: ingredients, brands, clinical and branded drugs.

    Args:
        name: A drug, brand, ingredient or clinical drug name, e.g. "ibuprofen".
        max_results: How many concepts to list.
        offset: How many concepts to skip; the footer gives the next offset.

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
    query = name.strip()
    return _RXNAV.get_json(
        "drugs.json",
        params={"name": query},
        parse=lambda data: _concepts_answer(
            _dict(data, "drugGroup"),
            heading=f"RxNorm concepts for {query!r} (details: rxnorm_concept):",
            nothing=f"No RxNorm drug concepts match {query!r}.",
            page=(offset, max_results),
        ),
    )


@tool(capability="network")
def rxnorm_concept(rxcui: str) -> str:
    """Get an RxNorm concept's name, term type and synonym by RxCUI.

    Args:
        rxcui: The concept's RxNorm identifier.

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
        missing=_no_concept(normalized),
    )


@tool(capability="network")
def rxnorm_related(
    rxcui: str,
    tty: str = "",
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = 20,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """List the RxNorm concepts related to a concept: its ingredients, brands, forms and drugs.

    Args:
        rxcui: The concept's RxNorm identifier.
        tty: Term types to keep, e.g. "IN", "BN", "SCD" or "SBD", several joined with spaces;
            none for every related concept.
        max_results: How many concepts to list.
        offset: How many concepts to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``rxcui`` is not 1-12 digits or ``tty`` is not term
            types; not_found when RxNorm answers that it has no concept with ``rxcui``.
    """
    normalized = _rxcui(rxcui)
    types = " ".join(re.split(r"[\s,+]+", tty.strip().upper())) if tty.strip() else ""
    if tty.strip() and not _TTY_RE.fullmatch(tty.strip()):
        raise ToolFailure(
            "validation_error",
            f"invalid tty {tty!r}; give RxNorm term types such as 'IN', 'BN', 'SCD' or 'SBD', "
            "joined with spaces.",
        )
    of = f" of term types {types}" if types else ""
    return _RXNAV.get_json(
        "rxcui",
        normalized,
        "related.json" if types else "allrelated.json",
        params={"tty": types} if types else {},
        parse=lambda data: _concepts_answer(
            _dict(data, "relatedGroup" if types else "allRelatedGroup"),
            heading=f"RxNorm concepts related to RxCUI {normalized}{of}:",
            nothing=f"RxNorm lists no related concepts{of} for RxCUI {normalized}.",
            page=(offset, max_results),
        ),
        missing=_no_concept(normalized),
    )


@tool(capability="network")
def rxnorm_ndcs(
    rxcui: str,
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = _PAGE_MAX,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """List the active NDC product codes of an RxNorm drug (an SCD, SBD, GPCK or BPCK concept).

    Args:
        rxcui: The drug's RxNorm identifier.
        max_results: How many NDCs to list.
        offset: How many NDCs to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``rxcui`` is not 1-12 digits; not_found when RxNorm
            answers that it has no concept with it.
    """
    normalized = _rxcui(rxcui)
    return _RXNAV.get_json(
        "rxcui",
        normalized,
        "ndcs.json",
        parse=lambda data: _ndcs_answer(data, normalized, offset, max_results),
        missing=_no_concept(normalized),
    )


@tool(capability="network")
def dailymed_label_search(
    drug_name: str = "",
    ndc: str = "",
    rxcui: str = "",
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = 10,
    page: Annotated[int, Range(1)] = 1,
) -> ToolResult:
    """Search DailyMed drug labels (SPL) by drug name, NDC or RxCUI.

    Args:
        drug_name: A generic or brand name.
        ndc: A National Drug Code, e.g. "0002-4462-30".
        rxcui: The RxCUI of a clinical or branded drug (an SCD or SBD from the rxnorm_*
            tools).
        max_results: How many labels a page lists.
        page: Which page, from 1; the footer gives the next.

    Raises:
        ToolFailure: validation_error when no filter is given or one of them is invalid.
    """
    filters = _label_filters(drug_name, ndc, rxcui)
    params = {"page": str(page), "pagesize": str(max_results), **filters}
    described = _labels_described(filters)
    return _DAILYMED.get_json(
        "spls.json",
        params=params,
        parse=lambda data: _labels_answer(data, described, page, max_results),
    )


@tool(capability="network")
def dailymed_label(setid: str, offset: Annotated[int, Range(0)] = 0) -> ToolResult:
    """List a DailyMed label's sections, numbered, with their sizes, after the label's title,
    version, effective date and labeler.

    Args:
        setid: The label's set ID, from dailymed_label_search.
        offset: How many sections to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``setid`` is not a set ID; not_found when DailyMed has
            no label with it.
    """
    return _label(setid, lambda normalized, label: _outline(normalized, label, offset))


@tool(capability="network")
def dailymed_label_text(
    setid: str,
    section: Annotated[int, Range(1)] | None = None,
    find: str = "",
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Read a DailyMed label's text: whole, one section (its subsections included), or the
    passages that mention a term.

    Lists come one item per line and tables one row per line.

    Args:
        setid: The label's set ID, from dailymed_label_search.
        section: The section's number from dailymed_label; none for the whole label.
        find: A term to look for: the answer is the passages around each match.
        offset: Where to start, in characters of the label or the section (with ``find``, where
            to search on from); the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``setid`` is not a set ID or the label has no such
            section; not_found when DailyMed has no label with that set ID.
    """
    term = find.strip()

    def read(normalized: str, label: SplLabel) -> ToolResult:
        where, text = f"DailyMed label {normalized}", label.text
        if section is not None:
            part = _section(label, section)
            where = f"{where}, section {section} ({part.title})"
            text = label.text[part.start : part.end]
        if not text.strip():
            return ToolResult.success(f"{where} has no text.")
        if term:
            window = find_window(text, term, offset=offset, limit=max_chars)
            heading = f"{where}, passages that mention {term!r}:"
        else:
            window = text_window(text, offset=offset, limit=max_chars)
            heading = f"{where}:"
        return _within_section(window, section).result(heading=heading)

    return _label(setid, read)


# --- RxNorm -------------------------------------------------------------------------------------


def _rxcui(rxcui: str) -> str:
    """``rxcui`` stripped, or a validation_error when it is not an RxCUI."""
    normalized = rxcui.strip()
    if not _RXCUI_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid rxcui {rxcui!r}; an RxCUI is 1-12 digits; find one with rxnorm_drug_search.",
        )
    return normalized


def _no_concept(rxcui: str) -> str:
    return f"no RxNorm concept with RxCUI {rxcui}; search with rxnorm_drug_search."


def _concepts_answer(
    group: dict[str, Any], *, heading: str, nothing: str, page: tuple[int, int]
) -> ToolResult:
    """A page of the concepts in ``group``'s ``conceptGroup``, numbered across all of them."""
    lines = [
        f"{number}. {_string(concept.get('name'))} | RxCUI {_string(concept.get('rxcui'))} | "
        f"TTY {_tty(concept.get('tty') or tty)}"
        for number, (tty, concept) in enumerate(_concepts(group.get("conceptGroup")), start=1)
    ]
    if not lines:
        return ToolResult.success(nothing)
    offset, limit = page
    return page_window(lines, offset=offset, limit=limit).result(heading=heading)


def _concepts(groups: object) -> list[tuple[object, dict[str, Any]]]:
    concepts: list[tuple[object, dict[str, Any]]] = []
    for group in groups if isinstance(groups, list) else []:
        if isinstance(group, dict):
            concepts += [
                (group.get("tty"), concept)
                for concept in _list(group.get("conceptProperties"))
                if isinstance(concept, dict)
            ]
    return concepts


def _tty(value: object) -> str:
    """A term type with its name: ``SCD (Semantic Clinical Drug)``."""
    code = _string(value)
    name = _TTY_NAMES.get(code.upper())
    return f"{code} ({name})" if name else code or "?"


def _concept_text(data: dict[str, Any], rxcui: str) -> str:
    props = data.get("properties")
    if not isinstance(props, dict) or not props:
        raise ToolFailure("not_found", _no_concept(rxcui))
    facts = [
        f"TTY: {_tty(props.get('tty'))}",
        f"language: {_string(props.get('language')) or '?'}",
        f"suppress: {_string(props.get('suppress')) or '?'}",
    ]
    lines = [f"RxNorm concept {rxcui}:", _string(props.get("name")) or "(no name)"]
    lines.append("   " + " | ".join(facts))
    if synonym := _string(props.get("synonym")):
        lines.append(f"   synonym: {synonym}")
    return "\n".join(lines)


def _ndcs_answer(data: dict[str, Any], rxcui: str, offset: int, limit: int) -> ToolResult:
    """The NDCs, in the CMS 11-digit form RxNav returns them
    (https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getNDCs.html)."""
    ndcs = [_string(ndc) for ndc in _list(_dict(_dict(data, "ndcGroup"), "ndcList").get("ndc"))]
    lines = [f"{number}. {ndc}" for number, ndc in enumerate(filter(None, ndcs), start=1)]
    if not lines:
        return ToolResult.success(f"RxNorm lists no active NDCs for RxCUI {rxcui}.")
    heading = (
        f"NDCs of RxCUI {rxcui}, in the CMS 11-digit form (its labels: "
        f'dailymed_label_search(rxcui="{rxcui}")):'
    )
    return page_window(lines, offset=offset, limit=limit).result(heading=heading)


# --- DailyMed -----------------------------------------------------------------------------------


def _label_filters(drug_name: str, ndc: str, rxcui: str) -> dict[str, str]:
    """The search's filters, by DailyMed's parameter names.

    Raises:
        ToolFailure: validation_error when none is given or one is invalid.
    """
    filters = {"drug_name": drug_name.strip(), "ndc": ndc.strip(), "rxcui": rxcui.strip()}
    if not any(filters.values()):
        raise ToolFailure(
            "validation_error", "nothing to search; provide drug_name, ndc or rxcui."
        )
    checks = (
        ("drug_name", _TEXT_RE, "1-180 characters of a drug name"),
        ("ndc", _NDC_RE, "4-20 digits and dashes, e.g. '0002-4462-30'"),
        ("rxcui", _RXCUI_RE, "1-12 digits, from the rxnorm_* tools"),
    )
    for key, pattern, form in checks:
        if filters[key] and not pattern.fullmatch(filters[key]):
            raise ToolFailure("validation_error", f"invalid {key} {filters[key]!r}; give {form}.")
    return {key: value for key, value in filters.items() if value}


def _labels_described(filters: dict[str, str]) -> str:
    names = {"drug_name": "drug name", "ndc": "NDC", "rxcui": "RxCUI"}
    return ", ".join(
        f"{names[key]} {value!r}" if key == "drug_name" else f"{names[key]} {value}"
        for key, value in filters.items()
    )


def _labels_answer(data: dict[str, Any], described: str, page: int, size: int) -> ToolResult:
    items = [item for item in _list(data.get("data")) if isinstance(item, dict)]
    meta = _dict(data, "metadata")
    pages, total = _int(meta.get("total_pages")), _int(meta.get("total_elements"))
    first = (page - 1) * size + 1
    if not items and total and first > total:
        last = pages or -(-total // size)
        return ToolResult.success(
            f"Page {page} is past the end: {total} DailyMed labels match {described}, on {last} "
            f"pages of {size}; the last is page={last}."
        )
    if not items:
        later = f" on page {page}" if page > 1 else ""
        return ToolResult.success(f"No DailyMed labels match {described}{later}.")
    entries = [_label_entry(number, item) for number, item in enumerate(items, start=first)]
    shown = first - 1 + len(items)
    more = (pages is not None and page < pages) or (total is not None and shown < total)
    next_call = {"page": page + 1} if more else None
    window = list_window(entries, first=first, total=total, next_call=next_call)
    return window.result(
        heading=f"DailyMed labels for {described} (read one with dailymed_label):"
    )


def _label_entry(number: int, item: dict[str, Any]) -> str:
    version = _string(item.get("spl_version"))
    facts = [
        f"setid: {_string(item.get('setid'))}",
        f"version {version}" if version else "",
        f"published {_published(_string(item.get('published_date')))}",
    ]
    return f"{number}. {_string(item.get('title'))}\n   " + " | ".join(f for f in facts if f)


def _published(value: str) -> str:
    """DailyMed's ``published_date`` (``Jun 10, 2026``) in ISO 8601."""
    match = _PUBLISHED_RE.match(value)
    month = _MONTHS.get(match[1].lower()) if match else None
    if match and month:
        try:
            return date(int(match[3]), month, int(match[2])).isoformat()
        except ValueError:
            pass
    return value or "?"


def _label[T](setid: str, read: Callable[[str, SplLabel], T]) -> T:
    """The label with ``setid``, read by ``read`` inside the door.

    Raises:
        ToolFailure: validation_error when ``setid`` is not a set ID; not_found when DailyMed has
            no label with it.
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
        parse=lambda xml_text: read(normalized, spl_label(xml_text)),
        missing=(
            f"DailyMed has no label with set ID {normalized}; find labels with "
            "dailymed_label_search"
        ),
    )


def _outline(setid: str, label: SplLabel, offset: int) -> ToolResult:
    url = f"{_DAILYMED_PAGE_URL}?setid={urllib.parse.quote(setid)}"
    facts = [
        f"version {label.version}" if label.version else "",
        f"effective {label.effective}" if label.effective else "",
        label.organization,
    ]
    known = " | ".join(fact for fact in facts if fact)
    head = [f"DailyMed label {setid}: {label.title or '(no title)'}"]
    head += [f"   {known}" if known else "", f"   DailyMed: {url}"]
    head = [line for line in head if line]
    if not label.sections:
        return ToolResult.success("\n".join([*head, "The label has no sections."]))
    lines = [
        f"{'  ' * min(part.depth, 5)}{part.number}. {part.title}"
        + (f" [LOINC {part.code}]" if part.code else "")
        + f": {part.end - part.start} chars"
        for part in label.sections
    ]
    head.append(f'Sections (read one with dailymed_label_text(setid="{setid}", section=N)):')
    window = page_window(lines, offset=offset, limit=_OUTLINE_PAGE)
    return window.result(heading="\n".join(head))


def _section(label: SplLabel, number: int) -> SplSection:
    """Section ``number`` of ``label``.

    Raises:
        ToolFailure: validation_error when the label has no such section.
    """
    if not 1 <= number <= len(label.sections):
        raise ToolFailure(
            "validation_error",
            f"the label has {len(label.sections)} sections, not {number}; list them with "
            "dailymed_label",
        )
    return label.sections[number - 1]


def _within_section(window: Window, section: int | None) -> Window:
    """A window of one section names the section in its next call: its offsets are the
    section's."""
    if section is None or window.next_call is None:
        return window
    return replace(window, next_call={**window.next_call, "section": section})


# --- Reading the JSON ---------------------------------------------------------------------------


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: object) -> int | None:
    """A count DailyMed sends as text (``"26"``) or as a number."""
    text = _string(value)
    return int(text) if text.isascii() and text.isdigit() else None


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
