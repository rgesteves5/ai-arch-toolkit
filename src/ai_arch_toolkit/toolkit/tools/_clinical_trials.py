"""ClinicalTrials.gov: search studies, and read a study's record (T07; D39, D41).

The API v2 pages a search by cursor: it counts the matches with the first page (``countTotal``)
and hands the next page's token (``nextPageToken``), which the footer gives back with the position,
so the next page is numbered on. A study's record is read whole and served through the window: by
section, by offset, or as the passages around a term, so no summary, criterion, arm, outcome,
site or reference is cut where nothing reads on (D39). The ``markup`` fields (summary,
description, eligibility criteria) come in markdown, one line per paragraph or list item, and
keep their lines (https://clinicaltrials.gov/api/oas/v2).
"""

from __future__ import annotations

import re
import typing
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import Window, find_window, list_window, text_window


def _refused(reply: Reply) -> ToolFailure | None:
    """ClinicalTrials.gov explains a request it refuses in a ``text/plain`` 400, the OpenAPI
    spec's ``errorMessage`` (https://clinicaltrials.gov/api/oas/v2): an expression or a filter
    value it does not take."""
    if reply.status != 400:
        return None
    said = " ".join(str(reply.body).split())[:300] or "Bad Request"
    return ToolFailure(
        "validation_error",
        f"ClinicalTrials.gov refused the request ({said}); check the terms' syntax and the filter "
        "values (status e.g. 'recruiting', study_type 'interventional', phase 'phase 3')",
    )


_API = Api(
    base="https://clinicaltrials.gov/api/v2", name="ClinicalTrials.gov", error_reader=_refused
)
_PAGE_MAX = 20
_MAX_CHARS = 20_000
_DEFAULT_CHARS = 6_000
_NCT_ID_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
_STUDY_URL = "https://clinicaltrials.gov/study/"
# The search terms' parameters, in Essie syntax (https://clinicaltrials.gov/api/oas/v2).
_TERM_PARAMS = {
    "query": "query.term",
    "condition": "query.cond",
    "intervention": "query.intr",
    "location": "query.locn",
}

type StudySection = Literal[
    "overview",
    "summary",
    "description",
    "eligibility",
    "arms",
    "outcomes",
    "locations",
    "references",
]
_SECTIONS: tuple[str, ...] = typing.get_args(StudySection.__value__)


@tool(capability="network")
def clinical_trials_search(
    query: str = "",
    condition: str = "",
    intervention: str = "",
    location: str = "",
    status: str = "",
    study_type: str = "",
    phase: str = "",
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = 5,
    page_token: str = "",
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search ClinicalTrials.gov studies.

    Args:
        query: General search terms, in the site's search syntax.
        condition: A condition or disease.
        intervention: An intervention or treatment.
        location: A place: a city, a state, a country or a facility.
        status: The overall status, e.g. "recruiting", "completed" or "terminated".
        study_type: "interventional" or "observational".
        phase: e.g. "phase 3", "early phase 1" or "NA".
        max_results: How many studies to list.
        page_token: The next page's token, from the footer; send it with the same search.
        offset: How many studies came before that page, from the footer.

    Raises:
        ToolFailure: validation_error when no search term is given, an offset comes without a
            page token, or ClinicalTrials.gov refuses a term or a filter.
    """
    given = {"query": query, "condition": condition, "intervention": intervention}
    terms = {name: value.strip() for name, value in {**given, "location": location}.items()}
    if not any(terms.values()):
        msg = "provide query, condition, intervention, or location, e.g. condition='asthma'."
        raise ToolFailure("validation_error", msg)
    if offset and not page_token.strip():
        msg = "offset numbers the page a page_token reads; pass both as the footer gives them"
        raise ToolFailure("validation_error", msg)
    kinds = {
        "status": _normalize_enum(status),
        "study type": _normalize_enum(study_type),
        "phase": _normalize_phase(phase),
    }
    params = {
        "format": "json",
        "pageSize": str(max_results),
        "countTotal": "true",
        "filter.overallStatus": kinds["status"],
        "filter.advanced": _advanced_filter(study_type, phase),
        "pageToken": page_token.strip(),
        **{_TERM_PARAMS[name]: value for name, value in terms.items()},
    }
    params = {key: value for key, value in params.items() if value}
    described = _described(terms, kinds)
    return _API.get_json(
        "studies", params=params, parse=lambda data: _search_answer(data, described, offset)
    )


@tool(capability="network")
def clinical_trial_study(
    nct_id: str,
    section: StudySection | None = None,
    find: str = "",
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Read a ClinicalTrials.gov study's record: whole, one section, or the passages that mention
    a term.

    The answer names the record's sections with their sizes; the overview has the status, the
    design, the sponsor, the dates and the enrollment.

    Args:
        nct_id: The study's NCT ID, e.g. "NCT04280705".
        section: One section of the record; none for the whole record.
        find: A term to look for: the answer is the passages around each match.
        offset: Where to start, in characters of the record or the section (with ``find``, where
            to search on from); the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when the NCT ID or the section is invalid; not_found when
            ClinicalTrials.gov has no study with that NCT ID.
    """
    normalized = nct_id.strip().upper()
    if not _NCT_ID_RE.fullmatch(normalized):
        msg = f"invalid NCT ID {nct_id!r}; an NCT ID is NCT and 8 digits, e.g. NCT04280705."
        raise ToolFailure("validation_error", msg)
    if section is not None and section not in _SECTIONS:
        msg = f"invalid section {section!r}; give one of {', '.join(_SECTIONS)}"
        raise ToolFailure("validation_error", msg)
    missing = (
        f"no ClinicalTrials.gov study with NCT ID {normalized}; search with "
        "clinical_trials_search."
    )
    record = _API.get_json(
        "studies", normalized, params={"format": "json"}, parse=_record, missing=missing
    )
    if record is None:
        raise ToolFailure("not_found", missing)
    return _read(record, section, find.strip(), offset, max_chars)


# --- The record ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _Part:
    """A section of a record: its name, its text (heading included), and its items, if a list."""

    name: str
    text: str
    items: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _Study:
    """A study's record as the tools read it."""

    nct_id: str
    title: str
    parts: tuple[_Part, ...]

    def text(self, section: str | None) -> str:
        """The record's text, or one section's (empty when the record has none)."""
        chosen = [part for part in self.parts if section in (None, part.name)]
        return "\n".join(part.text for part in chosen if part.text)

    def contents(self) -> str:
        """The sections the record has, with their sizes."""
        sizes = [
            f"{part.name} ({'' if part.items is None else f'{part.items}, '}"
            f"{len(part.text)} chars)"
            for part in self.parts
            if part.text
        ]
        return "Sections: " + ", ".join(sizes)


def _read(
    record: _Study, section: str | None, term: str, offset: int, max_chars: int
) -> ToolResult:
    title = f"ClinicalTrials.gov study {record.nct_id}: {record.title}"
    where = record.nct_id if section is None else f"{record.nct_id}, section {section}"
    text = record.text(section)
    if not text:
        return ToolResult.success(
            f"{title}\n{record.contents()}\n{record.nct_id} has no {section}."
        )
    if term:
        window = find_window(text, term, offset=offset, limit=max_chars)
        label = f"{where}, passages that mention {term!r}:"
    else:
        window = text_window(text, offset=offset, limit=max_chars)
        label = f"{where}:"
    return _within_section(window, section).result(
        heading=f"{title}\n{record.contents()}\n{label}"
    )


def _within_section(window: Window, section: str | None) -> Window:
    """A window of one section names the section in its next call: its offsets are the
    section's."""
    if section is None or window.next_call is None:
        return window
    return replace(window, next_call={**window.next_call, "section": section})


def _record(data: dict[str, Any]) -> _Study | None:
    protocol = data.get("protocolSection")
    if not isinstance(protocol, dict) or not _listed(data):
        return None
    description = _dict(protocol, "descriptionModule")
    arms = _dict(protocol, "armsInterventionsModule")
    outcomes = _outcome_lines(_dict(protocol, "outcomesModule"))
    locations = _dicts(_dict(protocol, "contactsLocationsModule"), "locations")
    sites = [f"- {site}" for site in map(_site, locations) if site]
    references = _reference_lines(_dict(protocol, "referencesModule"))
    parts = (
        _part("overview", _overview(data, protocol)),
        _part("summary", _markup(description.get("briefSummary"))),
        _part("description", _markup(description.get("detailedDescription"))),
        _part("eligibility", _eligibility(_dict(protocol, "eligibilityModule"))),
        _part("arms", _arm_lines(arms), len(_dicts(arms, "armGroups"))),
        _part("outcomes", outcomes, _count(outcomes)),
        _part("locations", sites, len(sites)),
        _part("references", references, len(references)),
    )
    nct_id = _string(_dict(protocol, "identificationModule").get("nctId"))
    return _Study(nct_id=nct_id, title=_title(protocol), parts=parts)


def _part(name: str, lines: Iterable[str], items: int | None = None) -> _Part:
    """A section: its heading and its lines; no text when it has no lines."""
    body = [line for line in lines if line]
    text = "\n".join([f"## {name.capitalize()}", *body]) + "\n" if body else ""
    return _Part(name=name, text=text, items=items)


def _count(lines: list[str]) -> int:
    return sum(line.startswith("- ") for line in lines)


def _overview(data: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    identification = _dict(protocol, "identificationModule")
    nct_id = _string(identification.get("nctId"))
    official = _string(identification.get("officialTitle"))
    return [
        _title(protocol),
        f"Official title: {official}" if official and official != _title(protocol) else "",
        _status_line(data, protocol),
        *_listing_lines(protocol),
        f"URL: {_STUDY_URL}{nct_id}",
    ]


def _title(protocol: dict[str, Any]) -> str:
    identification = _dict(protocol, "identificationModule")
    brief = _string(identification.get("briefTitle"))
    return brief or _string(identification.get("officialTitle")) or "(untitled)"


def _status_line(data: dict[str, Any], protocol: dict[str, Any]) -> str:
    design = _dict(protocol, "designModule")
    meta = [
        _string(_dict(protocol, "identificationModule").get("nctId")),
        _labelled("status", _string(_dict(protocol, "statusModule").get("overallStatus"))),
        _labelled("type", _string(design.get("studyType"))),
        _labelled("phase", ", ".join(_strings(design.get("phases")))),
        "results posted" if data.get("hasResults") else "no results posted",
    ]
    return " | ".join(item for item in meta if item)


def _listing_lines(protocol: dict[str, Any]) -> list[str]:
    """The lines a search lists for a study after its status: conditions to enrollment."""
    conditions = _strings(_dict(protocol, "conditionsModule").get("conditions"))
    interventions = [
        _join([_string(item.get("type")), _string(item.get("name"))], ": ")
        for item in _dicts(_dict(protocol, "armsInterventionsModule"), "interventions")
    ]
    sponsor = _string(
        _dict(_dict(protocol, "sponsorCollaboratorsModule"), "leadSponsor").get("name")
    )
    return [
        _labelled("Conditions", ", ".join(conditions)),
        _labelled("Interventions", ", ".join(item for item in interventions if item)),
        _labelled("Sponsor", sponsor),
        _labelled("Dates", _dates(_dict(protocol, "statusModule"))),
        _labelled(
            "Enrollment", _enrollment(_dict(_dict(protocol, "designModule"), "enrollmentInfo"))
        ),
    ]


def _dates(status: dict[str, Any]) -> str:
    dates = (
        ("start", "startDateStruct"),
        ("primary completion", "primaryCompletionDateStruct"),
        ("completion", "completionDateStruct"),
    )
    return " | ".join(
        f"{label} {when}"
        for label, key in dates
        if (when := _string(_dict(status, key).get("date")))
    )


def _enrollment(info: dict[str, Any]) -> str:
    count = _string(info.get("count"))
    kind = _string(info.get("type")).lower()
    if not count:
        return ""
    return f"{count} participants" + (f" ({kind})" if kind else "")


def _eligibility(module: dict[str, Any]) -> list[str]:
    ages = _join([_string(module.get("minimumAge")), _string(module.get("maximumAge"))], " to ")
    healthy = module.get("healthyVolunteers")
    facts = [
        _labelled("Sex", _string(module.get("sex"))),
        _labelled("ages", ages),
        _labelled(
            "healthy volunteers", "yes" if healthy is True else "no" if healthy is False else ""
        ),
    ]
    summary = " | ".join(fact for fact in facts if fact)
    return [summary, *_markup(module.get("eligibilityCriteria"))]


def _arm_lines(module: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for arm in _dicts(module, "armGroups"):
        given = ", ".join(_strings(arm.get("interventionNames")))
        head = _join([_string(arm.get("type")), _string(arm.get("label"))], ": ")
        text = _join([head, _string(arm.get("description"))], ": ")
        lines.append(f"- {text}" + (f" Interventions: {given}" if given else ""))
    details = [
        "- "
        + _join(
            [
                _string(item.get("type")),
                _string(item.get("name")),
                _string(item.get("description")),
            ],
            ": ",
        )
        for item in _dicts(module, "interventions")
    ]
    return lines + (["Interventions:", *details] if details else [])


def _outcome_lines(module: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for label, key in (
        ("Primary", "primaryOutcomes"),
        ("Secondary", "secondaryOutcomes"),
        ("Other", "otherOutcomes"),
    ):
        outcomes = [_outcome(item) for item in _dicts(module, key)]
        if outcomes:
            lines += [f"{label}:", *(f"- {outcome}" for outcome in outcomes)]
    return lines


def _outcome(item: dict[str, Any]) -> str:
    measure = _string(item.get("measure"))
    when = _string(item.get("timeFrame"))
    text = f"{measure} ({when})" if measure and when else measure or when
    return _join([text, _string(item.get("description"))], ": ")


def _site(item: dict[str, Any]) -> str:
    place = ", ".join(
        part
        for part in (_string(item.get(key)) for key in ("facility", "city", "state", "country"))
        if part
    )
    status = _string(item.get("status"))
    return f"{place} ({status})" if place and status else place


def _reference_lines(module: dict[str, Any]) -> list[str]:
    lines = [
        "- "
        + _join(
            [
                f"PMID {pmid}" if (pmid := _string(item.get("pmid"))) else "",
                _string(item.get("citation")),
            ],
            ": ",
        )
        for item in _dicts(module, "references")
    ]
    lines += [
        "- " + _join([_string(item.get("label")), _string(item.get("url"))], ": ")
        for item in _dicts(module, "seeAlsoLinks")
    ]
    return [line for line in lines if line != "- "]


# --- The search ---------------------------------------------------------------------------------


def _search_answer(data: dict[str, Any], described: str, offset: int) -> ToolResult:
    found = data.get("studies")
    studies = [item for item in found if _listed(item)] if isinstance(found, list) else []
    token = _string(data.get("nextPageToken"))
    total = data.get("totalCount")
    if not studies and not token:
        more = f" after the first {offset}" if offset else ""
        return ToolResult.success(f"No ClinicalTrials.gov studies match {described}{more}.")
    if not studies:
        return ToolResult.success(
            f"ClinicalTrials.gov sent an empty page for {described}; read on with "
            f"page_token={token!r}, offset={offset}."
        )
    entries = [_entry(number, item) for number, item in enumerate(studies, start=offset + 1)]
    next_call = {"page_token": token, "offset": offset + len(studies)} if token else None
    window = list_window(
        entries,
        first=offset + 1,
        total=total if isinstance(total, int) and not isinstance(total, bool) else None,
        next_call=next_call,
    )
    return window.result(
        heading=f"ClinicalTrials.gov studies for {described} (read one with clinical_trial_study):"
    )


def _listed(item: object) -> bool:
    """A study of a search page with an NCT ID, the one thing its entry needs."""
    if not isinstance(item, dict):
        return False
    protocol = _dict(item, "protocolSection")
    return bool(_string(_dict(protocol, "identificationModule").get("nctId")))


def _entry(number: int, data: dict[str, Any]) -> str:
    protocol = _dict(data, "protocolSection")
    lines = [
        f"{number}. {_title(protocol)}",
        _status_line(data, protocol),
        *_listing_lines(protocol),
    ]
    return "\n   ".join(line for line in lines if line)


def _described(terms: dict[str, str], filters: dict[str, str]) -> str:
    """The search in words: ``query 'covid', status RECRUITING``."""
    said = [f"{name} {value!r}" for name, value in terms.items() if value]
    said += [f"{name} {value}" for name, value in filters.items() if value]
    return ", ".join(said)


def _advanced_filter(study_type: str, phase: str) -> str:
    advanced_filters = []
    normalized_study_type = _normalize_enum(study_type)
    if normalized_study_type:
        advanced_filters.append(f"AREA[StudyType]{normalized_study_type}")
    normalized_phase = _normalize_phase(phase)
    if normalized_phase:
        advanced_filters.append(f"AREA[Phase]{normalized_phase}")
    return " AND ".join(advanced_filters)


def _normalize_enum(value: str) -> str:
    return value.strip().upper().replace("-", "_").replace(" ", "_")


def _normalize_phase(value: str) -> str:
    phase = _normalize_enum(value)
    if not phase:
        return ""
    if phase in {"N_A", "NOT_APPLICABLE"}:
        return "NA"
    if phase.startswith("PHASE") and len(phase) > 5 and phase[5].isdigit():
        return f"PHASE{phase[5:]}"
    if phase.startswith("PHASE_"):
        return "PHASE" + phase[6:]
    if phase == "EARLY_PHASE_1":
        return "EARLY_PHASE1"
    return phase


# --- Reading the JSON ---------------------------------------------------------------------------


def _markup(value: object) -> list[str]:
    """A ``markup`` field's lines (markdown): each paragraph or list item on a line of its own."""
    if not isinstance(value, str):
        return []
    return [line for line in (" ".join(raw.split()) for raw in value.splitlines()) if line]


def _labelled(label: str, value: str) -> str:
    return f"{label}: {value}" if value else ""


def _dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _dicts(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _string(item))]


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _join(values: list[str], separator: str) -> str:
    return separator.join(value for value in values if value)
