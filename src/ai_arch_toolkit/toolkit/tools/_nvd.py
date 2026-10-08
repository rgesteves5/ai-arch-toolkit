"""NVD tools: search CVEs, and read one CVE's record (T06).

The CVE API 2.0 says how many CVEs match (``totalResults``) and pages by ``startIndex``; a
publication date range spans at most 120 consecutive days
(https://nvd.nist.gov/developers/vulnerabilities). A CVE's record gives its whole description,
every CVSS score with its version, and every weakness, CPE and reference, through the window
(D39).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._records import DEFAULT_CHARS, MAX_CHARS, record
from ai_arch_toolkit.toolkit.tools._window import list_window


def _nvd_error(reply: Reply) -> ToolFailure | str | None:
    """The reason NVD gives for a request it refuses, in its ``message`` response header; ``None``
    without one.

    "When client errors occur, users should examine the response header for a field named
    message" (https://nvd.nist.gov/developers/start-here; the 2.0 announcements,
    https://nvd.nist.gov/general/news/api-20-announcements). NVD refuses a bad parameter with a
    404 or a 400 that says why there: the request's fault. Its rate limit and a refused key
    (403, 429) keep the door's reading, in NVD's words.
    """
    if reply.status < 400:
        return None
    reason = " ".join(reply.headers.get("message", "").split())
    if not reason or reply.status not in (400, 404):
        return reason or None
    return ToolFailure(
        "validation_error", f"NVD refused the request: {reason}; correct that parameter"
    )


# Spaced for NVD's rate limit on requests without an API key.
_API = Api(
    base="https://services.nvd.nist.gov/rest/json/cves/2.0",
    name="NVD",
    timeout_s=15,
    min_interval_s=6.1,
    error_reader=_nvd_error,
)
# "The maximum allowable range when using any date range parameters is 120 consecutive days."
_MAX_RANGE_DAYS = 120
_CVE_ID_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_CVE_ID_FORM = "a CVE ID looks like CVE-2021-44228"
# The CVSS versions of ``metrics``, newest first, and the version each key holds.
_METRIC_KEYS = (
    ("cvssMetricV40", "4.0"),
    ("cvssMetricV31", "3.1"),
    ("cvssMetricV30", "3.0"),
    ("cvssMetricV2", "2.0"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Metric:
    """One CVSS score: its version, who scored it, and the vector."""

    version: str
    score: float | None
    severity: str
    vector: str
    kind: str
    source: str

    def summary(self) -> str:
        score = f"{self.score:.1f}" if self.score is not None else "?"
        return " ".join(part for part in (f"CVSS {self.version}: {score}", self.severity) if part)


@dataclass(frozen=True, slots=True, kw_only=True)
class _NvdCve:
    """A CVE, as NVD records it."""

    cve_id: str
    published: str
    last_modified: str
    status: str
    description: str
    metrics: tuple[_Metric, ...]
    weaknesses: tuple[tuple[str, str], ...]
    cpes: tuple[str, ...]
    references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Page:
    cves: list[_NvdCve]
    total: int | None


@tool(capability="network")
def nvd_cve_search(
    query: str = "",
    cve_id: str = "",
    cpe_name: str = "",
    cvss_severity: str = "",
    max_results: Annotated[int, Range(1, 20)] = 5,
    start: Annotated[int, Range(0)] = 0,
    pub_start_date: str = "",
    pub_end_date: str = "",
) -> ToolResult:
    """Search CVEs in the NVD by keyword, CVE ID, CPE name, CVSS v3 severity or publication
    date, numbered, with the total.

    Args:
        query: Keywords to look for in the descriptions.
        cve_id: An exact CVE ID.
        cpe_name: A CPE name, e.g. "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*".
        cvss_severity: A CVSS v3 severity: LOW, MEDIUM, HIGH or CRITICAL.
        max_results: How many CVEs to list.
        start: How many results to skip; the footer gives the next start.
        pub_start_date: The earliest publication date, YYYY-MM-DD; with pub_end_date, a range of
            at most 120 days (NVD's limit).
        pub_end_date: The latest publication date, YYYY-MM-DD.

    Raises:
        ToolFailure: validation_error when an argument is invalid, no filter is given, the
            date range passes 120 days, or NVD refuses a parameter.
    """
    filters = _search_filters(query, cve_id, cpe_name, cvss_severity)
    dates = _date_range(pub_start_date, pub_end_date)
    if not filters and dates is None:
        raise ToolFailure(
            "validation_error",
            "no filter given; provide query, cve_id, cpe_name, cvss_severity, or a publication "
            "date range",
        )
    params = {"resultsPerPage": str(max_results), "startIndex": str(start), **filters}
    if dates is not None:
        params["pubStartDate"] = f"{dates[0]:%Y-%m-%d}T00:00:00.000"
        params["pubEndDate"] = f"{dates[1]:%Y-%m-%d}T23:59:59.999"
    found = _API.get_json(params=params, parse=_cves)
    described = _described(filters, dates)
    if not found.cves and start == 0:
        return ToolResult.success(f"No NVD CVEs match {described}.")
    blocks = [_result_block(start + n, cve) for n, cve in enumerate(found.cves, start=1)]
    end = start + len(found.cves)
    more = found.total is not None and end < found.total
    next_call = {"start": end} if found.cves and more else None
    window = list_window(blocks, first=start + 1, total=found.total, next_call=next_call)
    return window.result(heading=f"NVD CVEs that match {described}:")


@tool(capability="network")
def nvd_cve(
    cve_id: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, MAX_CHARS)] = DEFAULT_CHARS,
) -> ToolResult:
    """Read a CVE's NVD record: its whole description, every CVSS score with its version and
    vector, and every weakness, affected CPE and reference.

    Args:
        cve_id: A CVE identifier, e.g. "CVE-2021-44228".
        offset: Where to start, in characters of the record; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``cve_id`` is not a CVE ID; not_found when NVD has no
            such CVE.
    """
    normalized = _cve_id(cve_id)
    if not normalized:
        raise ToolFailure("validation_error", f"cve_id cannot be empty; {_CVE_ID_FORM}")
    cves = _API.get_json(params={"cveId": normalized}, parse=_cves).cves
    if not cves:
        raise ToolFailure(
            "not_found",
            f"NVD has no CVE {normalized}; check the ID, or search by keyword with nvd_cve_search",
        )
    heading = f"NVD CVE {normalized}:"
    return record(_record_text(cves[0]), heading=heading, offset=offset, max_chars=max_chars)


def _cve_id(cve_id: str) -> str:
    """``cve_id`` normalized (upper case); empty for an empty one.

    Raises:
        ToolFailure: validation_error when ``cve_id`` is not a CVE ID.
    """
    normalized = cve_id.strip().upper()
    if normalized and not _CVE_ID_RE.fullmatch(normalized):
        raise ToolFailure("validation_error", f"invalid CVE ID {cve_id!r}; {_CVE_ID_FORM}")
    return normalized


def _search_filters(query: str, cve_id: str, cpe_name: str, cvss_severity: str) -> dict[str, str]:
    """The search's filter parameters.

    Raises:
        ToolFailure: validation_error for an invalid CVE ID or severity.
    """
    normalized_cve = _cve_id(cve_id)
    severity = cvss_severity.strip().upper()
    if severity and severity not in _SEVERITIES:
        raise ToolFailure(
            "validation_error",
            f"invalid cvss_severity {cvss_severity!r}; use LOW, MEDIUM, HIGH, or CRITICAL",
        )
    filters = {
        "keywordSearch": query.strip(),
        "cveId": normalized_cve,
        "cpeName": cpe_name.strip(),
        "cvssV3Severity": severity,
    }
    return {key: value for key, value in filters.items() if value}


def _described(filters: dict[str, str], dates: tuple[date, date] | None) -> str:
    """The search in words, for its heading and for zero results."""
    labels = {
        "keywordSearch": "keyword {!r}",
        "cveId": "CVE ID {}",
        "cpeName": "CPE {}",
        "cvssV3Severity": "CVSS v3 severity {}",
    }
    parts = [labels[key].format(value) for key, value in filters.items()]
    if dates is not None:
        parts.append(f"published {dates[0]:%Y-%m-%d} to {dates[1]:%Y-%m-%d}")
    return ", ".join(parts)


def _date_range(start: str, end: str) -> tuple[date, date] | None:
    """The publication date range; none without dates.

    Raises:
        ToolFailure: validation_error for a date missing, malformed or out of order, or a range
            past NVD's 120 days.
    """
    start, end = start.strip(), end.strip()
    if not start and not end:
        return None
    if not start or not end:
        raise ToolFailure(
            "validation_error", "pub_start_date and pub_end_date must be provided together"
        )
    start_date, end_date = _parse_date(start), _parse_date(end)
    if start_date is None:
        raise ToolFailure("validation_error", f"invalid pub_start_date {start!r}; use YYYY-MM-DD")
    if end_date is None:
        raise ToolFailure("validation_error", f"invalid pub_end_date {end!r}; use YYYY-MM-DD")
    if start_date > end_date:
        raise ToolFailure(
            "validation_error", "pub_start_date must be before or equal to pub_end_date"
        )
    days = (end_date - start_date).days + 1
    if days > _MAX_RANGE_DAYS:
        raise ToolFailure(
            "validation_error",
            f"NVD takes a publication range of at most {_MAX_RANGE_DAYS} consecutive days, and "
            f"{start_date} to {end_date} is {days}; split it into ranges of {_MAX_RANGE_DAYS} "
            "days or fewer",
        )
    return start_date, end_date


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _cves(data: dict[str, Any]) -> _Page:
    items = data.get("vulnerabilities", [])
    cves = [cve for item in items if isinstance(item, dict) if (cve := _parse_cve(item))]
    total = data.get("totalResults")
    return _Page(cves, total if isinstance(total, int) and not isinstance(total, bool) else None)


def _parse_cve(data: dict[str, Any]) -> _NvdCve | None:
    cve = data.get("cve")
    if not isinstance(cve, dict):
        return None
    cve_id = _string(cve.get("id"))
    if not cve_id:
        return None
    return _NvdCve(
        cve_id=cve_id,
        published=_timestamp(cve.get("published")),
        last_modified=_timestamp(cve.get("lastModified")),
        status=_string(cve.get("vulnStatus")),
        description=_description(cve.get("descriptions")),
        metrics=_metrics(cve.get("metrics")),
        weaknesses=_weaknesses(cve.get("weaknesses")),
        cpes=_cpes(cve.get("configurations")),
        references=_references(cve.get("references")),
    )


def _timestamp(value: Any) -> str:
    """NVD's timestamp to the second (``2021-12-10T10:15:09.143`` reads 2021-12-10T10:15:09)."""
    return _string(value).split(".", 1)[0]


def _description(value: Any) -> str:
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and item.get("lang") == "en":
            return _string(item.get("value"))
    return ""


def _metrics(metrics: Any) -> tuple[_Metric, ...]:
    """Every CVSS score, newest version first, each with who scored it."""
    found: list[_Metric] = []
    for key, version in _METRIC_KEYS:
        items = metrics.get(key) if isinstance(metrics, dict) else None
        for item in items if isinstance(items, list) else []:
            data = item.get("cvssData") if isinstance(item, dict) else None
            if not isinstance(data, dict):
                continue
            found.append(
                _Metric(
                    version=_string(data.get("version")) or version,
                    score=_float_or_none(data.get("baseScore")),
                    severity=_string(data.get("baseSeverity") or item.get("baseSeverity")),
                    vector=_string(data.get("vectorString")),
                    kind=_string(item.get("type")),
                    source=_string(item.get("source")),
                )
            )
    return tuple(found)


def _weaknesses(value: Any) -> tuple[tuple[str, str], ...]:
    """Each weakness (a CWE) with who named it: (CWE, "Primary, nvd@nist.gov")."""
    weaknesses: list[tuple[str, str]] = []
    for weakness in value if isinstance(value, list) else []:
        if not isinstance(weakness, dict):
            continue
        who = ", ".join(
            part
            for part in (_string(weakness.get("type")), _string(weakness.get("source")))
            if part
        )
        for desc in weakness.get("description") or []:
            if (
                isinstance(desc, dict)
                and desc.get("lang") == "en"
                and (cwe := _string(desc.get("value")))
            ):
                weaknesses.append((cwe, who))
    return tuple(dict.fromkeys(weaknesses))


def _cpes(value: Any) -> tuple[str, ...]:
    """Each CPE match, with whether it is vulnerable and the versions it covers."""
    cpes: list[str] = []
    for config in value if isinstance(value, list) else []:
        nodes = config.get("nodes") if isinstance(config, dict) else None
        for node in nodes if isinstance(nodes, list) else []:
            matches = node.get("cpeMatch") if isinstance(node, dict) else None
            for match in matches if isinstance(matches, list) else []:
                if isinstance(match, dict) and (cpe := _string(match.get("criteria"))):
                    cpes.append(_cpe_text(cpe, match))
    return tuple(dict.fromkeys(cpes))


def _cpe_text(cpe: str, match: dict[str, Any]) -> str:
    notes = ["vulnerable" if match.get("vulnerable") else "not vulnerable"]
    bounds = (
        ("versionStartIncluding", "from {} including"),
        ("versionStartExcluding", "from {} excluding"),
        ("versionEndIncluding", "to {} including"),
        ("versionEndExcluding", "to {} excluding"),
    )
    versions = [form.format(v) for key, form in bounds if (v := _string(match.get(key)))]
    if versions:
        notes.append(", ".join(versions))
    return f"{cpe} ({'; '.join(notes)})"


def _references(value: Any) -> tuple[str, ...]:
    """Each reference's URL, with its tags (Exploit, Patch, Vendor Advisory …)."""
    refs = value.get("referenceData") if isinstance(value, dict) else value
    references: list[str] = []
    for item in refs if isinstance(refs, list) else []:
        if isinstance(item, dict) and (url := _string(item.get("url"))):
            tags = item.get("tags")
            tagged = (
                ", ".join(_string(t) for t in tags if _string(t)) if isinstance(tags, list) else ""
            )
            references.append(f"{url} ({tagged})" if tagged else url)
    return tuple(dict.fromkeys(references))


def _versions(cve: _NvdCve) -> str:
    """One score per CVSS version, the primary scorer's when there is one."""
    chosen: dict[str, _Metric] = {}
    for metric in cve.metrics:
        if metric.version not in chosen or (
            metric.kind == "Primary" and chosen[metric.version].kind != "Primary"
        ):
            chosen[metric.version] = metric
    return " | ".join(metric.summary() for metric in chosen.values())


def _status_line(cve: _NvdCve, *, scores: bool) -> str:
    meta = [_versions(cve)] if scores else []
    if cve.status:
        meta.append(f"status: {cve.status}")
    if cve.published:
        meta.append(f"published: {cve.published}")
    if not scores and cve.last_modified:
        meta.append(f"last modified: {cve.last_modified}")
    return " | ".join(part for part in meta if part)


def _result_block(number: int, cve: _NvdCve) -> str:
    lines = [_status_line(cve, scores=True)]
    if cve.description:
        lines.append(f"Description: {cve.description}")
    if cve.weaknesses:
        lines.append(f"Weaknesses: {', '.join(cwe for cwe, _who in cve.weaknesses)}")
    counts = f"CPEs: {len(cve.cpes)} | References: {len(cve.references)}"
    lines.append(f"{counts} (nvd_cve({cve.cve_id!r}) lists them)")
    return "\n".join([f"{number}. {cve.cve_id}", *(f"   {line}" for line in lines if line)])


def _record_text(cve: _NvdCve) -> str:
    """The whole record: the description and the scores before the lists, which can run long."""
    lines = [cve.cve_id, _status_line(cve, scores=False)]
    if cve.description:
        lines.append(f"Description: {cve.description}")
    if cve.metrics:
        lines.append(f"CVSS ({len(cve.metrics)}):")
        lines += [f"- {_metric_text(metric)}" for metric in cve.metrics]
    if cve.weaknesses:
        named = [f"{cwe} ({who})" if who else cwe for cwe, who in cve.weaknesses]
        lines.append(f"Weaknesses: {'; '.join(named)}")
    if cve.cpes:
        lines += [f"CPEs ({len(cve.cpes)}):", *(f"- {cpe}" for cpe in cve.cpes)]
    if cve.references:
        lines += [f"References ({len(cve.references)}):", *(f"- {r}" for r in cve.references)]
    lines.append(f"URL: https://nvd.nist.gov/vuln/detail/{cve.cve_id}")
    return "\n".join(line for line in lines if line) + "\n"


def _metric_text(metric: _Metric) -> str:
    who = ", ".join(part for part in (metric.kind, metric.source) if part)
    text = metric.summary() + (f" ({who})" if who else "")
    return f"{text} {metric.vector}" if metric.vector else text


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(_string(value)) if _string(value) else None
    except ValueError:
        return None


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
