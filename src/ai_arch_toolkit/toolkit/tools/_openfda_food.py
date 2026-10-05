"""OpenFDA food tools — public FDA food enforcement recall search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api

# openFDA answers a search that matches nothing with HTTP 404 and
# {"error": {"code": "NOT_FOUND", "message": "No matches found!"}} (observed; the tests replay it;
# API basics: https://open.fda.gov/apis/), so its searches declare ``empty_on_404``. Its other
# errors carry their text in ``error.message``, which the door already quotes.
_API = Api(base="https://api.fda.gov/food/enforcement.json", name="openFDA", timeout_s=15)
_MAX_RESULTS_LIMIT = 20
_TEXT_RE = re.compile(r"^[\w\s,.'&()/%:+-]{1,160}$", re.UNICODE)
_RECALL_RE = re.compile(r"^[A-Z]-\d{3,5}-\d{4}$", re.IGNORECASE)
_NO_RECALLS = "No openFDA food recalls found."


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


@tool(capability="network")
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

    Raises:
        ToolFailure: validation_error when ``skip`` is negative, a filter or a date is invalid,
            or nothing is given to search.
    """
    if skip < 0:
        raise ToolFailure(
            "validation_error", f"skip must be greater than or equal to 0, not {skip}"
        )
    search = _build_search(query, product, reason, classification, status, state, country)
    date_filter = _date_filter(from_date, to_date)
    if not search and not date_filter:
        raise ToolFailure(
            "validation_error", "nothing to search; provide query, filters, or date range"
        )
    if date_filter:
        search = f"({search}) AND {date_filter}" if search else date_filter

    params = {
        "search": search,
        "limit": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "skip": str(skip),
    }
    return _API.get_json(params=params, parse=_search_text, empty_on_404=True)


@tool(capability="network")
def openfda_food_recall(recall_number: str) -> str:
    """Fetch an FDA food enforcement recall by recall number.

    Args:
        recall_number: FDA recall number, e.g. "F-2473-2016".

    Raises:
        ToolFailure: validation_error when ``recall_number`` is not a recall number; not_found
            when openFDA has no recall with it.
    """
    normalized = recall_number.strip().upper()
    if not _RECALL_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid recall_number {recall_number!r}; a recall number looks like F-2473-2016",
        )

    missing = (
        f"openFDA has no food recall {normalized}; find recalls with openfda_food_recall_search"
    )
    recalls = _API.get_json(
        params={"search": f'recall_number:"{normalized}"', "limit": "1"},
        parse=_recalls_from_data,
        missing=missing,
    )

    if not recalls:
        raise ToolFailure("not_found", missing)
    return f"openFDA food recall {normalized}:\n" + _format_recalls(
        recalls,
        include_index=False,
        include_details=True,
    )


def _search_text(data: dict[str, Any]) -> str:
    recalls = _recalls_from_data(data)
    if not recalls:
        return _NO_RECALLS
    total = _string(data.get("meta", {}).get("results", {}).get("total")) or "?"
    return f"openFDA food recalls (returned {len(recalls)}, total {total}):\n" + _format_recalls(
        recalls
    )


def _build_search(
    query: str,
    product: str,
    reason: str,
    classification: str,
    status: str,
    state: str,
    country: str,
) -> str:
    """The ``search`` expression of the text filters; empty without any.

    Raises:
        ToolFailure: validation_error for a filter with characters it cannot take.
    """
    clauses: list[str] = []
    if query.strip():
        if not _valid_text(query):
            raise _invalid_text("query")
        text = _escape(query.strip())
        clauses.append(
            f'(product_description:"{text}" OR '
            f'reason_for_recall:"{text}" OR '
            f'recalling_firm:"{text}")'
        )
    for name, field, value in (
        ("product", "product_description", product),
        ("reason", "reason_for_recall", reason),
        ("classification", "classification.exact", classification),
        ("status", "status.exact", status),
        ("state", "state.exact", state),
        ("country", "country.exact", country),
    ):
        value = value.strip()
        if not value:
            continue
        if not _valid_text(value):
            raise _invalid_text(name)
        clauses.append(f'{field}:"{_escape(value)}"')
    return " AND ".join(clauses)


def _invalid_text(name: str) -> ToolFailure:
    return ToolFailure(
        "validation_error",
        f"invalid {name}; use 1-160 letters, digits, spaces and basic punctuation (,.'&()/%:+-)",
    )


def _date_filter(from_date: str, to_date: str) -> str:
    """The report date range's clause; empty without dates.

    Raises:
        ToolFailure: validation_error for a malformed date or a range out of order.
    """
    start = from_date.strip()
    end = to_date.strip()
    if not start and not end:
        return ""
    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None
    if start and start_date is None:
        raise ToolFailure("validation_error", f"invalid from_date {from_date!r}; use YYYY-MM-DD")
    if end and end_date is None:
        raise ToolFailure("validation_error", f"invalid to_date {to_date!r}; use YYYY-MM-DD")
    if start_date and end_date and start_date > end_date:
        raise ToolFailure("validation_error", "from_date must be before or equal to to_date")
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
