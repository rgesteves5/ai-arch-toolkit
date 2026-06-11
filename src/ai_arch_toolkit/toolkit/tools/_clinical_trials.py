"""ClinicalTrials.gov tools — public clinical study search and lookup."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

_BASE_URL = "https://clinicaltrials.gov/api/v2"
_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 20
_SUMMARY_MAX_CHARS = 900
_ELIGIBILITY_MAX_CHARS = 1200
_NCT_ID_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ClinicalTrial:
    """Normalized metadata for a ClinicalTrials.gov study."""

    nct_id: str
    brief_title: str
    official_title: str
    status: str
    study_type: str
    phases: tuple[str, ...]
    conditions: tuple[str, ...]
    interventions: tuple[str, ...]
    sponsor: str
    summary: str
    start_date: str
    primary_completion_date: str
    completion_date: str
    enrollment: str
    has_results: bool
    locations: tuple[str, ...]
    arms: tuple[str, ...]
    primary_outcomes: tuple[str, ...]
    secondary_outcomes: tuple[str, ...]
    eligibility: str
    references: tuple[str, ...]


@tool
def clinical_trials_search(
    query: str = "",
    condition: str = "",
    intervention: str = "",
    location: str = "",
    status: str = "",
    study_type: str = "",
    phase: str = "",
    max_results: int = 5,
    page_token: str = "",
) -> str:
    """Search ClinicalTrials.gov studies using the public API v2.

    Args:
        query: General search terms.
        condition: Condition or disease query.
        intervention: Intervention or treatment query.
        location: Location query.
        status: Optional overall status, e.g. "recruiting", "completed", or "terminated".
        study_type: Optional study type, e.g. "interventional" or "observational".
        phase: Optional phase, e.g. "phase3", "phase 2", or "NA".
        max_results: Number of studies to return (1-20). Defaults to 5.
        page_token: Optional next_page_token from a previous result page.
    """
    if not any(value.strip() for value in (query, condition, intervention, location, page_token)):
        return (
            "ClinicalTrials.gov search failed: provide query, condition, "
            "intervention, or location."
        )

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    params = {
        "format": "json",
        "pageSize": str(max_results),
    }

    if query.strip():
        params["query.term"] = query.strip()
    if condition.strip():
        params["query.cond"] = condition.strip()
    if intervention.strip():
        params["query.intr"] = intervention.strip()
    if location.strip():
        params["query.locn"] = location.strip()
    if page_token.strip():
        params["pageToken"] = page_token.strip()

    normalized_status = _normalize_enum(status)
    if normalized_status:
        params["filter.overallStatus"] = normalized_status

    advanced_filters = []
    normalized_study_type = _normalize_enum(study_type)
    if normalized_study_type:
        advanced_filters.append(f"AREA[StudyType]{normalized_study_type}")
    normalized_phase = _normalize_phase(phase)
    if normalized_phase:
        advanced_filters.append(f"AREA[Phase]{normalized_phase}")
    if advanced_filters:
        params["filter.advanced"] = " AND ".join(advanced_filters)

    try:
        data = _fetch_json("/studies", params)
        studies = data.get("studies", [])
        trials = [_parse_trial(item) for item in studies if isinstance(item, dict)]
        trials = [trial for trial in trials if trial is not None]
    except urllib.error.HTTPError as e:
        return f"ClinicalTrials.gov search failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"ClinicalTrials.gov search failed: URL error: {e.reason}"
    except TimeoutError:
        return "ClinicalTrials.gov search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ClinicalTrials.gov search failed: could not parse API response: {e}"

    if not trials:
        return "No ClinicalTrials.gov studies found."

    result = "ClinicalTrials.gov studies:\n" + _format_trials(trials, include_details=False)
    next_page_token = str(data.get("nextPageToken", "") or "").strip()
    if next_page_token:
        result += f"\n\nNext page token: {next_page_token}"
    return result


@tool
def clinical_trial_study(nct_id: str) -> str:
    """Fetch detailed metadata for a ClinicalTrials.gov study.

    Args:
        nct_id: ClinicalTrials.gov identifier, e.g. "NCT04280705".
    """
    normalized = nct_id.strip().upper()
    if not _NCT_ID_RE.fullmatch(normalized):
        return f"ClinicalTrials.gov study lookup failed: invalid NCT ID: {nct_id!r}"

    try:
        data = _fetch_json(f"/studies/{normalized}", {"format": "json"})
        trial = _parse_trial(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"ClinicalTrials.gov study not found: {normalized}"
        return f"ClinicalTrials.gov study lookup failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"ClinicalTrials.gov study lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "ClinicalTrials.gov study lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"ClinicalTrials.gov study lookup failed: could not parse API response: {e}"

    if trial is None:
        return f"ClinicalTrials.gov study not found: {normalized}"

    return f"ClinicalTrials.gov study {normalized}:\n" + _format_trials(
        [trial],
        include_index=False,
        include_details=True,
    )


def _fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _parse_trial(data: dict[str, Any]) -> _ClinicalTrial | None:
    protocol = data.get("protocolSection")
    if not isinstance(protocol, dict):
        return None

    identification = _dict(protocol, "identificationModule")
    status = _dict(protocol, "statusModule")
    sponsor = _dict(protocol, "sponsorCollaboratorsModule")
    description = _dict(protocol, "descriptionModule")
    conditions = _dict(protocol, "conditionsModule")
    design = _dict(protocol, "designModule")
    arms_interventions = _dict(protocol, "armsInterventionsModule")
    outcomes = _dict(protocol, "outcomesModule")
    eligibility = _dict(protocol, "eligibilityModule")
    contacts_locations = _dict(protocol, "contactsLocationsModule")
    references = _dict(protocol, "referencesModule")

    nct_id = _string(identification.get("nctId"))
    if not nct_id:
        return None

    enrollment = _dict(design, "enrollmentInfo")
    enrollment_text = _join_nonempty(
        [
            _string(enrollment.get("count")),
            _string(enrollment.get("type")),
        ],
        " ",
    )

    return _ClinicalTrial(
        nct_id=nct_id,
        brief_title=_string(identification.get("briefTitle")),
        official_title=_string(identification.get("officialTitle")),
        status=_string(status.get("overallStatus")),
        study_type=_string(design.get("studyType")),
        phases=_string_tuple(design.get("phases")),
        conditions=_string_tuple(conditions.get("conditions")),
        interventions=_interventions(arms_interventions),
        sponsor=_lead_sponsor(sponsor),
        summary=_string(description.get("briefSummary")),
        start_date=_date_struct(status.get("startDateStruct")),
        primary_completion_date=_date_struct(status.get("primaryCompletionDateStruct")),
        completion_date=_date_struct(status.get("completionDateStruct")),
        enrollment=enrollment_text,
        has_results=bool(data.get("hasResults")),
        locations=_locations(contacts_locations),
        arms=_arms(arms_interventions),
        primary_outcomes=_outcomes(outcomes.get("primaryOutcomes")),
        secondary_outcomes=_outcomes(outcomes.get("secondaryOutcomes")),
        eligibility=_string(eligibility.get("eligibilityCriteria")),
        references=_references(references),
    )


def _format_trials(
    trials: list[_ClinicalTrial],
    *,
    include_index: bool = True,
    include_details: bool = False,
) -> str:
    blocks: list[str] = []
    for index, trial in enumerate(trials, start=1):
        title = trial.brief_title or trial.official_title or "(untitled)"
        title = f"{index}. {title}" if include_index else title
        lines = [title]

        meta = [f"NCT ID: {trial.nct_id}"]
        if trial.status:
            meta.append(f"status: {trial.status}")
        if trial.study_type:
            meta.append(f"type: {trial.study_type}")
        if trial.phases:
            meta.append("phase: " + ", ".join(trial.phases))
        lines.append("   " + " | ".join(meta))

        if trial.conditions:
            lines.append("   Conditions: " + ", ".join(trial.conditions[:8]))
        if trial.interventions:
            lines.append("   Interventions: " + ", ".join(trial.interventions[:8]))
        if trial.sponsor:
            lines.append(f"   Sponsor: {trial.sponsor}")
        dates = _format_dates(trial)
        if dates:
            lines.append(f"   Dates: {dates}")
        if trial.enrollment:
            lines.append(f"   Enrollment: {trial.enrollment}")
        lines.append(f"   Has results: {trial.has_results}")
        if trial.locations:
            lines.append("   Locations: " + "; ".join(trial.locations[:5]))
        if trial.summary:
            lines.append(f"   Summary: {_truncate(trial.summary, _SUMMARY_MAX_CHARS)}")
        if include_details:
            if trial.official_title and trial.official_title != trial.brief_title:
                lines.append(f"   Official title: {trial.official_title}")
            if trial.arms:
                lines.append("   Arms:")
                for arm in trial.arms[:6]:
                    lines.append(f"     - {_truncate(arm, 220)}")
            if trial.primary_outcomes:
                lines.append("   Primary outcomes:")
                for outcome in trial.primary_outcomes[:6]:
                    lines.append(f"     - {_truncate(outcome, 240)}")
            if trial.secondary_outcomes:
                lines.append("   Secondary outcomes:")
                for outcome in trial.secondary_outcomes[:6]:
                    lines.append(f"     - {_truncate(outcome, 220)}")
            if trial.eligibility:
                lines.append(
                    f"   Eligibility: {_truncate(trial.eligibility, _ELIGIBILITY_MAX_CHARS)}"
                )
            if trial.references:
                lines.append("   References: " + " | ".join(trial.references[:5]))
        lines.append(f"   URL: https://clinicaltrials.gov/study/{trial.nct_id}")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_dates(trial: _ClinicalTrial) -> str:
    parts: list[str] = []
    if trial.start_date:
        parts.append(f"start {trial.start_date}")
    if trial.primary_completion_date:
        parts.append(f"primary completion {trial.primary_completion_date}")
    if trial.completion_date:
        parts.append(f"completion {trial.completion_date}")
    return " | ".join(parts)


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


def _lead_sponsor(module: dict[str, Any]) -> str:
    sponsor = module.get("leadSponsor")
    if isinstance(sponsor, dict):
        return _string(sponsor.get("name"))
    return ""


def _interventions(module: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in module.get("interventions", []) or []:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name"))
        kind = _string(item.get("type"))
        if name and kind:
            values.append(f"{kind}: {name}")
        elif name:
            values.append(name)
    return tuple(values)


def _arms(module: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in module.get("armGroups", []) or []:
        if not isinstance(item, dict):
            continue
        label = _string(item.get("label"))
        kind = _string(item.get("type"))
        description = _string(item.get("description"))
        text = _join_nonempty([kind, label, description], " - ")
        if text:
            values.append(text)
    return tuple(values)


def _outcomes(value: Any) -> tuple[str, ...]:
    outcomes: list[str] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        measure = _string(item.get("measure"))
        time_frame = _string(item.get("timeFrame"))
        description = _string(item.get("description"))
        text = measure
        if time_frame:
            text = f"{text} ({time_frame})" if text else time_frame
        if description:
            text = f"{text}: {description}" if text else description
        if text:
            outcomes.append(text)
    return tuple(outcomes)


def _locations(module: dict[str, Any]) -> tuple[str, ...]:
    locations: list[str] = []
    for item in module.get("locations", []) or []:
        if not isinstance(item, dict):
            continue
        parts = [
            _string(item.get("facility")),
            _string(item.get("city")),
            _string(item.get("state")),
            _string(item.get("country")),
        ]
        location = ", ".join(part for part in parts if part)
        status = _string(item.get("status"))
        if status and location:
            location = f"{location} ({status})"
        if location:
            locations.append(location)
    return tuple(locations)


def _references(module: dict[str, Any]) -> tuple[str, ...]:
    references: list[str] = []
    for key in ("references", "seeAlsoLinks"):
        for item in module.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            pmid = _string(item.get("pmid"))
            citation = _string(item.get("citation") or item.get("label"))
            url = _string(item.get("url"))
            text = _join_nonempty([f"PMID {pmid}" if pmid else "", citation, url], " - ")
            if text:
                references.append(text)
    return tuple(references)


def _date_struct(value: Any) -> str:
    if isinstance(value, dict):
        return _string(value.get("date"))
    return ""


def _dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_string(item) for item in value if _string(item))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _join_nonempty(values: list[str], separator: str) -> str:
    return separator.join(value for value in values if value)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
