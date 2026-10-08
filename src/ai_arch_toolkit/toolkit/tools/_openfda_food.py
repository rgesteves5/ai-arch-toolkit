"""openFDA food enforcement recalls: search them, and read one by recall number (T07; D39).

openFDA pages a search by ``skip`` and ``limit`` and counts the matches (``meta.results.total``);
``limit`` goes up to 1000 and ``skip`` up to 25000 (https://open.fda.gov/apis/query-parameters/),
so past that a search is narrowed by date. Its server answers a search that matches nothing with
a 404 ``NOT_FOUND`` "No matches found!", a parameter it refuses with a 400 ``BAD_REQUEST`` and the
reason, and a failed search with a 500 ``SERVER_ERROR``, each as ``{"error": {"code", "message"}}``
(https://github.com/FDA/openfda/blob/master/api/faers/api.js).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import list_window


def _openfda_error(reply: Reply) -> ToolFailure | str | None:
    """The error an openFDA answer reports: a refused parameter is the caller's to fix
    (``validation_error``); any other code is the source's words, typed by the status."""
    error = reply.body.get("error") if isinstance(reply.body, dict) else None
    if not isinstance(error, dict):
        return None
    code, message = _string(error.get("code")), _string(error.get("message"))
    if code == "BAD_REQUEST":
        return ToolFailure(
            "validation_error",
            f"openFDA refused the search ({message or code}); check the filters and the dates",
        )
    return ": ".join(text for text in (code, message) if text)


def _no_matches(reply: Reply) -> bool:
    """openFDA's 404 for a search that matches nothing, ``NOT_FOUND`` "No matches found!": any
    other 404 (a page, another body) is the endpoint gone."""
    error = reply.body.get("error") if isinstance(reply.body, dict) else None
    return isinstance(error, dict) and error.get("code") == "NOT_FOUND"


# A search that matches nothing is openFDA's 404 ``NOT_FOUND``, which every request declares
# (``empty_on_404``): a search says so, the recall lookup is ``not_found``. The reader reads every
# other error.
_API = Api(
    base="https://api.fda.gov/food/enforcement.json",
    name="openFDA",
    timeout_s=15,
    error_reader=_openfda_error,
)
_PAGE_MAX = 20
_SKIP_MAX = 25_000
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
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = 10,
    skip: Annotated[int, Range(0, _SKIP_MAX)] = 0,
) -> ToolResult:
    """Search FDA food enforcement recalls, newest reports included, through openFDA.

    Args:
        query: Text to find in the product, the reason or the firm.
        product: Text of the product description.
        reason: Text of the reason for the recall.
        classification: The recall's class: "Class I", "Class II" or "Class III".
        status: "Ongoing", "Completed" or "Terminated".
        state: The recalling firm's state code, e.g. "MI".
        country: The recalling firm's country, e.g. "United States".
        from_date: The earliest report date, as YYYY-MM-DD.
        to_date: The latest report date, as YYYY-MM-DD.
        max_results: How many recalls to list.
        skip: How many recalls to skip; the footer gives the next.

    Raises:
        ToolFailure: validation_error when a filter or a date is invalid, nothing is given to
            search, or openFDA refuses the search.
    """
    search = _build_search(query, product, reason, classification, status, state, country)
    date_filter = _date_filter(from_date, to_date)
    if not search and not date_filter:
        raise ToolFailure(
            "validation_error", "nothing to search; provide query, filters, or date range"
        )
    if date_filter:
        search = f"({search}) AND {date_filter}" if search else date_filter
    described = _described(
        {
            "query": query,
            "product": product,
            "reason": reason,
            "class": classification,
            "status": status,
            "state": state,
            "country": country,
        },
        from_date.strip(),
        to_date.strip(),
    )
    params = {"search": search, "limit": str(max_results), "skip": str(skip)}
    return _API.get_json(
        params=params,
        parse=lambda data: _search_answer(data, described, skip),
        empty_on_404=_no_matches,
    )


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
        empty_on_404=_no_matches,
    )

    if not recalls:
        raise ToolFailure("not_found", missing)
    recall = recalls[0]
    lines = [recall.product_description, *_lines(recall, details=True)]
    return f"openFDA food recall {normalized}:\n" + "\n   ".join(lines)


def _search_answer(data: dict[str, Any], described: str, skip: int) -> ToolResult:
    recalls = _recalls_from_data(data)
    if not recalls:
        later = f" after the first {skip}" if skip else ""
        return ToolResult.success(f"No openFDA food recalls match {described}{later}.")
    total = _dict(_dict(data, "meta"), "results").get("total")
    total = total if isinstance(total, int) and not isinstance(total, bool) else None
    shown = skip + len(recalls)
    more = total is not None and shown < total
    next_call = {"skip": shown} if more and shown <= _SKIP_MAX else None
    entries = [_entry(number, recall) for number, recall in enumerate(recalls, start=skip + 1)]
    window = replace(
        list_window(entries, first=skip + 1, total=total, next_call=next_call),
        rest=(
            f"openFDA reads no further than skip={_SKIP_MAX}: narrow the search with from_date "
            "and to_date for the rest"
        ),
    )
    return window.result(heading=f"openFDA food recalls for {described}:")


def _described(filters: dict[str, str], start: str, end: str) -> str:
    """The search in words: ``query 'milk', class 'Class I', report dates 2016-01-01 to …``."""
    said = [f"{name} {value.strip()!r}" for name, value in filters.items() if value.strip()]
    if start or end:
        said.append(f"report dates {start or 'any'} to {end or 'any'}")
    return ", ".join(said)


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


def _entry(number: int, recall: _OpenFdaRecall) -> str:
    """A recall in a search's list: its number, product, class, status, report date, firm and
    reason."""
    return "\n   ".join(
        [f"{number}. {recall.recall_number} — {recall.product_description}", *_lines(recall)]
    )


def _lines(recall: _OpenFdaRecall, *, details: bool = False) -> list[str]:
    meta = [
        f"recall: {recall.recall_number}",
        _labelled("class", recall.classification),
        _labelled("status", recall.status),
        _labelled("report date", recall.report_date),
    ]
    place = ", ".join(item for item in (recall.city, recall.state, recall.country) if item)
    firm = f"{recall.firm} ({place})" if recall.firm and place else recall.firm
    lines = [" | ".join(item for item in meta if item), _labelled("Firm", firm)]
    lines.append(_labelled("Reason", recall.reason))
    if details:
        lines += [
            _labelled("Distribution", recall.distribution),
            _labelled("Code info", recall.code_info),
            _labelled("Initiated", recall.initiation_date),
            _labelled("Terminated", recall.termination_date),
        ]
    return [line for line in lines if line]


def _labelled(label: str, value: str) -> str:
    return f"{label}: {value}" if value else ""


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


def _dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}
