"""NVD tools — public CVE search and vulnerability lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

# Spaced for NVD's rate limit on requests without an API key.
_API = Api(
    base="https://services.nvd.nist.gov/rest/json/cves/2.0",
    name="NVD",
    timeout_s=15,
    min_interval_s=6.1,
)
_MAX_RESULTS_LIMIT = 20
_CVE_ID_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_DESCRIPTION_MAX_CHARS = 900


@dataclass(frozen=True, slots=True, kw_only=True)
class _NvdCve:
    """Normalized NVD CVE metadata."""

    cve_id: str
    published: str
    last_modified: str
    status: str
    description: str
    cvss_score: float | None
    cvss_severity: str
    weaknesses: tuple[str, ...]
    cpes: tuple[str, ...]
    references: tuple[str, ...]


@tool(capability="network")
def nvd_cve_search(
    query: str = "",
    cve_id: str = "",
    cpe_name: str = "",
    cvss_severity: str = "",
    max_results: int = 5,
    start: int = 0,
    pub_start_date: str = "",
    pub_end_date: str = "",
) -> str:
    """Search CVEs using the public NVD API.

    Args:
        query: Keyword search text.
        cve_id: Optional exact CVE ID.
        cpe_name: Optional CPE name filter.
        cvss_severity: Optional CVSS v3 severity: LOW, MEDIUM, HIGH, or CRITICAL.
        max_results: Number of CVEs to return (1-20). Defaults to 5.
        start: Zero-based result offset. Defaults to 0.
        pub_start_date: Optional publication date lower bound as YYYY-MM-DD.
        pub_end_date: Optional publication date upper bound as YYYY-MM-DD.
    """
    if start < 0:
        return "NVD CVE search failed: start must be greater than or equal to 0."
    filters = _search_filters(query, cve_id, cpe_name, cvss_severity)
    if isinstance(filters, str):
        return filters
    date_params = _date_params(pub_start_date, pub_end_date)
    if isinstance(date_params, str):
        return date_params
    if not filters:
        return "NVD CVE search failed: provide query, cve_id, cpe_name, or cvss_severity."
    params = {
        "resultsPerPage": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "startIndex": str(start),
        **filters,
        **date_params,
    }
    try:
        cves = _API.get_json(params=params, parse=_cves)
    except HttpError as e:
        return f"NVD CVE search failed: {e}"

    if not cves:
        return "No NVD CVEs found."

    return "NVD CVE results:\n" + _format_cves(cves)


@tool(capability="network")
def nvd_cve(cve_id: str) -> str:
    """Fetch NVD metadata for a specific CVE ID.

    Args:
        cve_id: CVE identifier, e.g. "CVE-2021-44228".
    """
    normalized = cve_id.strip().upper()
    if not _CVE_ID_RE.fullmatch(normalized):
        return f"NVD CVE lookup failed: invalid CVE ID: {cve_id!r}"

    try:
        cves = _API.get_json(params={"cveId": normalized}, parse=_cves)
    except HttpError as e:
        return f"NVD CVE lookup failed: {e}"

    if not cves:
        return f"NVD CVE not found: {normalized}"

    return f"NVD CVE {normalized}:\n" + _format_cves(cves, include_index=False)


def _search_filters(
    query: str, cve_id: str, cpe_name: str, cvss_severity: str
) -> dict[str, str] | str:
    """The search's filter parameters, or the error text for an invalid one."""
    normalized_cve = cve_id.strip().upper()
    if normalized_cve and not _CVE_ID_RE.fullmatch(normalized_cve):
        return f"NVD CVE search failed: invalid CVE ID: {cve_id!r}"
    severity = cvss_severity.strip().upper()
    if severity and severity not in _SEVERITIES:
        return "NVD CVE search failed: cvss_severity must be LOW, MEDIUM, HIGH, or CRITICAL."
    filters = {
        "keywordSearch": query.strip(),
        "cveId": normalized_cve,
        "cpeName": cpe_name.strip(),
        "cvssV3Severity": severity,
    }
    return {key: value for key, value in filters.items() if value}


def _cves(data: dict[str, Any]) -> list[_NvdCve]:
    items = data.get("vulnerabilities", [])
    cves = [_parse_cve(item) for item in items if isinstance(item, dict)]
    return [cve for cve in cves if cve is not None]


def _date_params(start: str, end: str) -> dict[str, str] | str:
    start = start.strip()
    end = end.strip()
    if not start and not end:
        return {}
    if not start or not end:
        return "NVD CVE search failed: pub_start_date and pub_end_date must be provided together."
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if start_date is None:
        return f"NVD CVE search failed: invalid pub_start_date {start!r}. Use YYYY-MM-DD."
    if end_date is None:
        return f"NVD CVE search failed: invalid pub_end_date {end!r}. Use YYYY-MM-DD."
    if start_date > end_date:
        return "NVD CVE search failed: pub_start_date must be before or equal to pub_end_date."
    return {
        "pubStartDate": f"{start_date:%Y-%m-%d}T00:00:00.000",
        "pubEndDate": f"{end_date:%Y-%m-%d}T23:59:59.999",
    }


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_cve(data: dict[str, Any]) -> _NvdCve | None:
    cve = data.get("cve")
    if not isinstance(cve, dict):
        return None
    cve_id = str(cve.get("id", "") or "").strip()
    if not cve_id:
        return None
    score, severity = _cvss(cve.get("metrics"))
    return _NvdCve(
        cve_id=cve_id,
        published=str(cve.get("published", "") or "").strip(),
        last_modified=str(cve.get("lastModified", "") or "").strip(),
        status=str(cve.get("vulnStatus", "") or "").strip(),
        description=_description(cve.get("descriptions")),
        cvss_score=score,
        cvss_severity=severity,
        weaknesses=_weaknesses(cve.get("weaknesses")),
        cpes=_cpes(cve.get("configurations")),
        references=_references(cve.get("references")),
    )


def _description(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    for item in value:
        if isinstance(item, dict) and item.get("lang") == "en":
            return str(item.get("value", "") or "").strip()
    return ""


def _cvss(metrics: Any) -> tuple[float | None, str]:
    if not isinstance(metrics, dict):
        return None, ""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV40", "cvssMetricV2"):
        items = metrics.get(key)
        if not isinstance(items, list) or not items:
            continue
        item = items[0]
        if not isinstance(item, dict):
            continue
        data = item.get("cvssData")
        if isinstance(data, dict):
            score = data.get("baseScore")
            severity = str(data.get("baseSeverity", "") or item.get("baseSeverity", "") or "")
            return _float_or_none(score), severity
        severity = str(item.get("baseSeverity", "") or "")
        return None, severity
    return None, ""


def _weaknesses(value: Any) -> tuple[str, ...]:
    weaknesses: list[str] = []
    for weakness in value or []:
        if not isinstance(weakness, dict):
            continue
        for desc in weakness.get("description", []) or []:
            if isinstance(desc, dict) and desc.get("lang") == "en":
                text = str(desc.get("value", "") or "").strip()
                if text:
                    weaknesses.append(text)
    return tuple(dict.fromkeys(weaknesses))


def _cpes(value: Any) -> tuple[str, ...]:
    cpes: list[str] = []
    for config in value or []:
        if not isinstance(config, dict):
            continue
        for node in config.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            for match in node.get("cpeMatch", []) or []:
                if isinstance(match, dict):
                    cpe = str(match.get("criteria", "") or "").strip()
                    if cpe:
                        cpes.append(cpe)
    return tuple(dict.fromkeys(cpes))


def _references(value: Any) -> tuple[str, ...]:
    refs = value.get("referenceData") if isinstance(value, dict) else value
    references: list[str] = []
    for item in refs or []:
        if isinstance(item, dict):
            url = str(item.get("url", "") or "").strip()
            if url:
                references.append(url)
    return tuple(dict.fromkeys(references))


def _format_cves(cves: list[_NvdCve], *, include_index: bool = True) -> str:
    blocks: list[str] = []
    for index, cve in enumerate(cves, start=1):
        title = f"{index}. {cve.cve_id}" if include_index else cve.cve_id
        lines = [title]
        meta = []
        if cve.cvss_score is not None:
            meta.append(f"CVSS: {cve.cvss_score:g}")
        if cve.cvss_severity:
            meta.append(f"severity: {cve.cvss_severity}")
        if cve.status:
            meta.append(f"status: {cve.status}")
        if cve.published:
            meta.append(f"published: {cve.published}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if cve.description:
            lines.append(f"   Description: {_truncate(cve.description, _DESCRIPTION_MAX_CHARS)}")
        if cve.weaknesses:
            lines.append("   Weaknesses: " + ", ".join(cve.weaknesses[:8]))
        if cve.cpes:
            lines.append("   CPEs: " + " | ".join(cve.cpes[:5]))
        if cve.references:
            lines.append("   References: " + " | ".join(cve.references[:5]))
        lines.append(f"   URL: https://nvd.nist.gov/vuln/detail/{cve.cve_id}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value is not None and str(value).strip():
            return float(value)
    except ValueError:
        return None
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
