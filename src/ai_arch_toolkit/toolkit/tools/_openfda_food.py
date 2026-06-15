"""OpenFDA food tools — public FDA food enforcement recall search."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Any

from ai_arch_toolkit.core import tool

_API_URL = "https://api.fda.gov/food/enforcement.json"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit; research tool)"
_MAX_RESULTS_LIMIT = 20
_TEXT_RE = re.compile(r"^[\w\s,.'&()/%:+-]{1,160}$", re.UNICODE)
_RECALL_RE = re.compile(r"^[A-Z]-\d{3,5}-\d{4}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True, kw_only=True)
class _OpenFdaRecall:
    """Normalized openFDA food enforcement recall."""

    recall_number: str
    status: str
    classification: str
    product_type: str
    product_description: str
    reason: str
    firm: str
    city: str
    state: str
    country: str
    distribution: str
    code_info: str
    initiation_date: str
    report_date: str
    termination_date: str


@tool
def openfda_food_recall_search(
    query: str = "",
    product: str = "",
    reason: str = "",
    classification: str = "",
    status: str = "",
    state: str = "",
    country: str = "",
    from_date: str = "",
    to_date: str = "",
    max_results: int = 10,
    skip: int = 0,
) -> str:
    """Search FDA food enforcement recalls using openFDA.

    Args:
        query: Optional general text query across product, reason, and firm.
        product: Optional product description filter.
        reason: Optional reason-for-recall filter.
        classification: Optional recall class, e.g. "Class I", "Class II", "Class III".
        status: Optional recall status, e.g. "Ongoing" or "Terminated".
        state: Optional recalling firm state code.
        country: Optional recalling firm country.
        from_date: Optional report date lower bound as YYYY-MM-DD.
        to_date: Optional report date upper bound as YYYY-MM-DD.
        max_results: Number of recalls to return (1-20). Defaults to 10.
        skip: Zero-based result offset. Defaults to 0.
    """
    if skip < 0:
        return "openFDA food recall search failed: skip must be greater than or equal to 0."
    search = _build_search(query, product, reason, classification, status, state, country)
    if isinstance(search, str) and search.startswith("invalid"):
        return f"openFDA food recall search failed: {search}"
    date_filter = _date_filter(from_date, to_date)
    if date_filter.startswith("openFDA food recall search failed:"):
        return date_filter
    if not search and not date_filter:
        return "openFDA food recall search failed: provide query, filters, or date range."
    if date_filter:
        search = f"({search}) AND {date_filter}" if search else date_filter

    try:
        data = _fetch_json(
            {
                "search": search,
                "limit": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
                "skip": str(skip),
            }
        )
        recalls = _recalls_from_data(data)
    except urllib.error.HTTPError as e:
        return _http_error("openFDA food recall search failed", e)
    except urllib.error.URLError as e:
        return f"openFDA food recall search failed: URL error: {e.reason}"
    except TimeoutError:
        return "openFDA food recall search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"openFDA food recall search failed: could not parse API response: {e}"

    if not recalls:
        return "No openFDA food recalls found."
    total = _string(data.get("meta", {}).get("results", {}).get("total")) or "?"
    return f"openFDA food recalls (returned {len(recalls)}, total {total}):\n" + _format_recalls(
        recalls
    )


@tool
def openfda_food_recall(recall_number: str) -> str:
    """Fetch an FDA food enforcement recall by recall number.

    Args:
        recall_number: FDA recall number, e.g. "F-2473-2016".
    """
    normalized = recall_number.strip().upper()
    if not _RECALL_RE.fullmatch(normalized):
        return f"openFDA food recall lookup failed: invalid recall_number: {recall_number!r}"

    try:
        data = _fetch_json({"search": f'recall_number:"{normalized}"', "limit": "1"})
        recalls = _recalls_from_data(data)
    except urllib.error.HTTPError as e:
        return _http_error("openFDA food recall lookup failed", e)
    except urllib.error.URLError as e:
        return f"openFDA food recall lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "openFDA food recall lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"openFDA food recall lookup failed: could not parse API response: {e}"

    if not recalls:
        return f"openFDA food recall not found: {normalized}"
    return f"openFDA food recall {normalized}:\n" + _format_recalls(
        recalls,
        include_index=False,
        include_details=True,
    )


def _fetch_json(params: dict[str, str]) -> dict[str, Any]:
    url = f"{_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _build_search(
    query: str,
    product: str,
    reason: str,
    classification: str,
    status: str,
    state: str,
    country: str,
) -> str:
    clauses: list[str] = []
    if query.strip():
        if not _valid_text(query):
            return "invalid query."
        text = _escape(query.strip())
        clauses.append(
            f'(product_description:"{text}" OR '
            f'reason_for_recall:"{text}" OR '
            f'recalling_firm:"{text}")'
        )
    for field, value in (
        ("product_description", product),
        ("reason_for_recall", reason),
        ("classification.exact", classification),
        ("status.exact", status),
        ("state.exact", state),
        ("country.exact", country),
    ):
        value = value.strip()
        if not value:
            continue
        if not _valid_text(value):
            return f"invalid {field.removesuffix('.exact')}."
        clauses.append(f'{field}:"{_escape(value)}"')
    return " AND ".join(clauses)


def _date_filter(from_date: str, to_date: str) -> str:
    start = from_date.strip()
    end = to_date.strip()
    if not start and not end:
        return ""
    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None
    if start and start_date is None:
        return (
            f"openFDA food recall search failed: invalid from_date {from_date!r}. Use YYYY-MM-DD."
        )
    if end and end_date is None:
        return f"openFDA food recall search failed: invalid to_date {to_date!r}. Use YYYY-MM-DD."
    if start_date and end_date and start_date > end_date:
        return "openFDA food recall search failed: from_date must be before or equal to to_date."
    start_text = f"{start_date:%Y%m%d}" if start_date else "19000101"
    end_text = f"{end_date:%Y%m%d}" if end_date else "29991231"
    return f"report_date:[{start_text} TO {end_text}]"


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _recalls_from_data(data: dict[str, Any]) -> list[_OpenFdaRecall]:
    results = data.get("results", [])
    if not isinstance(results, list):
        return []
    return [
        recall for item in results if isinstance(item, dict) if (recall := _parse_recall(item))
    ]


def _parse_recall(data: dict[str, Any]) -> _OpenFdaRecall | None:
    recall_number = _string(data.get("recall_number"))
    if not recall_number:
        return None
    return _OpenFdaRecall(
        recall_number=recall_number,
        status=_string(data.get("status")),
        classification=_string(data.get("classification")),
        product_type=_string(data.get("product_type")),
        product_description=_string(data.get("product_description")),
        reason=_string(data.get("reason_for_recall")),
        firm=_string(data.get("recalling_firm")),
        city=_string(data.get("city")),
        state=_string(data.get("state")),
        country=_string(data.get("country")),
        distribution=_string(data.get("distribution_pattern")),
        code_info=_string(data.get("code_info")),
        initiation_date=_format_fda_date(data.get("recall_initiation_date")),
        report_date=_format_fda_date(data.get("report_date")),
        termination_date=_format_fda_date(data.get("termination_date")),
    )


def _format_recalls(
    recalls: list[_OpenFdaRecall],
    *,
    include_index: bool = True,
    include_details: bool = False,
) -> str:
    blocks: list[str] = []
    for index, recall in enumerate(recalls, start=1):
        title = (
            f"{index}. {recall.recall_number} — {recall.product_description}"
            if include_index
            else recall.product_description
        )
        lines = [title]
        meta = [f"recall: {recall.recall_number}"]
        if recall.classification:
            meta.append(f"class: {recall.classification}")
        if recall.status:
            meta.append(f"status: {recall.status}")
        if recall.report_date:
            meta.append(f"report date: {recall.report_date}")
        lines.append("   " + " | ".join(meta))
        if recall.firm:
            location = ", ".join(
                item for item in (recall.city, recall.state, recall.country) if item
            )
            firm = f"{recall.firm} ({location})" if location else recall.firm
            lines.append(f"   Firm: {firm}")
        if recall.reason:
            lines.append(f"   Reason: {recall.reason}")
        if include_details and recall.distribution:
            lines.append(f"   Distribution: {recall.distribution}")
        if include_details and recall.code_info:
            lines.append(f"   Code info: {recall.code_info}")
        if include_details and recall.initiation_date:
            lines.append(f"   Initiated: {recall.initiation_date}")
        if include_details and recall.termination_date:
            lines.append(f"   Terminated: {recall.termination_date}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by openFDA (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _escape(value: str) -> str:
    return value.replace('"', "")


def _format_fda_date(value: Any) -> str:
    text = _string(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
