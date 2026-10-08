"""Error answers as each source documents or sends them, for the contract test (point 2).

Each entry is one answer of the source; every network tool of the module it is listed under (or
the ``tools`` it names) must turn it into a ``ToolFailure`` of ``type`` whose message contains
``says``. A tool of a module with an ``Api`` and no entry that applies to it owes the point.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ai_arch_toolkit.core import ToolFailureType
from tests.toolkit.contract_cases import Body


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorBody:
    """One error answer and the failure it must give.

    Attributes:
        source: Where the answer comes from: the documentation's URL, or the live call that
            sent it.
        status: The HTTP status.
        body: The body.
        headers: The response headers.
        type: The failure's type.
        says: Text the failure's message carries (the source's words or code).
        tools: The tools of the module it applies to; empty for all its network tools.
    """

    source: str
    status: int
    body: Body = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    type: ToolFailureType
    says: str
    tools: frozenset[str] = frozenset()


_MEDIAWIKI_ERRORS = "https://www.mediawiki.org/wiki/API:Errors_and_warnings"


def _mediawiki(*tools: str) -> tuple[ErrorBody, ...]:
    """MediaWiki's error object, sent with HTTP 200 (en.wikibooks.org, 2026-09-29)."""
    names = frozenset(tools)
    return (
        ErrorBody(
            source=_MEDIAWIKI_ERRORS,
            status=200,
            body={"error": {"code": "readonly", "info": "The wiki is in read-only mode."}},
            type="upstream",
            says="readonly: The wiki is in read-only mode.",
            tools=names,
        ),
        ErrorBody(
            source=_MEDIAWIKI_ERRORS,
            status=200,
            body={"error": {"code": "ratelimited", "info": "You've exceeded your rate limit."}},
            type="rate_limited",
            says="You've exceeded your rate limit",
            tools=names,
        ),
    )


_BRAVE_ERRORS = "https://api-dashboard.search.brave.com/api-reference/web/search/get"
_TAVILY_ERRORS = "https://docs.tavily.com/documentation/api-reference/endpoint/search"
_BRAVE = frozenset({"brave_search"})
_TAVILY = frozenset({"tavily_search"})


def _brave(status: int, code: str, detail: str) -> dict[str, object]:
    """Brave's error body: the reference's ``ErrorResponse`` schema (``type``, and an ``error``
    with ``id``, ``status``, ``code``, ``detail`` and ``meta``); the values are the test's."""
    error = {"id": "4c1b7e6e", "status": status, "code": code, "detail": detail, "meta": {}}
    return {"type": "ErrorResponse", "error": error, "time": 1759900000}


def _tavily(error: str) -> dict[str, object]:
    """Tavily's error body, as the reference shows it: ``{"detail": {"error": …}}``."""
    return {"detail": {"error": error}}


_WEB_SEARCH = (
    # Brave documents 404, 422 and 429, each with the ErrorResponse schema.
    ErrorBody(
        source=f"{_BRAVE_ERRORS} (422)",
        status=422,
        body=_brave(422, "VALIDATION", "Unable to validate request parameter(s)."),
        type="validation_error",
        says="Unable to validate request parameter(s)",
        tools=_BRAVE,
    ),
    ErrorBody(
        source=f"{_BRAVE_ERRORS} (429)",
        status=429,
        body=_brave(429, "RATE_LIMITED", "Request rate limit exceeded for plan."),
        type="rate_limited",
        says="Request rate limit exceeded for plan.",
        tools=_BRAVE,
    ),
    ErrorBody(
        source=f"{_BRAVE_ERRORS} (404)",
        status=404,
        body=_brave(404, "NOT_FOUND", "Resource not found."),
        type="upstream",
        says="endpoint not found (HTTP 404); the API may have changed: Resource not found.",
        tools=_BRAVE,
    ),
    # Tavily documents 400, 401, 422, 429, 432, 433 and 500, with these bodies.
    ErrorBody(
        source=f"{_TAVILY_ERRORS} (400)",
        status=400,
        body=_tavily("Invalid topic. Must be 'general' or 'news'."),
        type="validation_error",
        says="Invalid topic. Must be 'general' or 'news'",
        tools=_TAVILY,
    ),
    ErrorBody(
        source=f"{_TAVILY_ERRORS} (401)",
        status=401,
        body=_tavily("Unauthorized: missing or invalid API key."),
        type="upstream",
        says="Unauthorized: missing or invalid API key",
        tools=_TAVILY,
    ),
    ErrorBody(
        source=f"{_TAVILY_ERRORS} (422)",
        status=422,
        body={
            "detail": [
                {
                    "type": "string_type",
                    "loc": ["body", "query"],
                    "msg": "Input should be a valid string",
                    "input": [],
                }
            ]
        },
        type="validation_error",
        says="query: Input should be a valid string",
        tools=_TAVILY,
    ),
    ErrorBody(
        source=f"{_TAVILY_ERRORS} (429)",
        status=429,
        body=_tavily(
            "Your request has been blocked due to excessive requests. Please reduce the rate of "
            "requests."
        ),
        type="rate_limited",
        says="Your request has been blocked due to excessive requests.",
        tools=_TAVILY,
    ),
    ErrorBody(
        source=f"{_TAVILY_ERRORS} (432)",
        status=432,
        body=_tavily(
            "This request exceeds your plan's set usage limit. Please upgrade your plan or "
            "contact support@tavily.com"
        ),
        type="rate_limited",
        says="This request exceeds your plan's set usage limit",
        tools=_TAVILY,
    ),
    ErrorBody(
        source=f"{_TAVILY_ERRORS} (433)",
        status=433,
        body=_tavily(
            "This request exceeds the pay-as-you-go limit. You can increase your limit on the "
            "Tavily dashboard."
        ),
        type="rate_limited",
        says="This request exceeds the pay-as-you-go limit",
        tools=_TAVILY,
    ),
    ErrorBody(
        source=f"{_TAVILY_ERRORS} (500)",
        status=500,
        body=_tavily("Internal Server Error"),
        type="upstream",
        says="Internal Server Error",
        tools=_TAVILY,
    ),
)


ERROR_BODIES: dict[str, tuple[ErrorBody, ...]] = {
    "_web_search": _WEB_SEARCH,
    "_air_quality": (
        ErrorBody(
            source="https://open-meteo.com/en/docs/air-quality-api (Errors)",
            status=400,
            body={
                "error": True,
                "reason": "Cannot initialize WeatherVariable from invalid String",
            },
            type="validation_error",
            says="Cannot initialize WeatherVariable from invalid String",
        ),
    ),
    "_wiki": _mediawiki(),
    "_wikidata": _mediawiki("wikidata_search"),
    "_eurostat": (
        ErrorBody(
            source="live, 2026-09-30 (statistics and catalogue APIs)",
            status=413,
            body=(
                b'{ "error": [{"status": 413,"id": 413,"label": "ASYNCHRONOUS_RESPONSE. Your '
                b'request will be treated asynchronously. Please try again later."}]}'
            ),
            type="upstream",
            says="ASYNCHRONOUS_RESPONSE",
            tools=frozenset(
                {"eurostat_dataset", "eurostat_dimensions", "eurostat_series", "eurostat_compare"}
            ),
        ),
    ),
    "_uniprot": (
        ErrorBody(
            source="live, 2026-09-30 (https://rest.uniprot.org/uniprotkb/search)",
            status=400,
            body={
                "url": "http://rest.uniprot.org/uniprotkb/search",
                "messages": ["Invalid request received. Unsupported query."],
            },
            type="upstream",
            says="Unsupported query",
        ),
    ),
}
