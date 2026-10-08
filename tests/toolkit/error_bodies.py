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

_ARCHIVE_METADATA = "https://archive.org/developers/md-read.html (extended errors; the text is"
_LIBRARY_ITEMS = frozenset({"open_library_work", "open_library_isbn"})

ERROR_BODIES: dict[str, tuple[ErrorBody, ...]] = {
    "_web_search": _WEB_SEARCH,
    "_internet_archive": (
        ErrorBody(
            source="live, 2026-09-29 (https://archive.org/advancedsearch.php, HTTP 200)",
            status=200,
            body={"error": 'a group is empty (near char ")" at position 10)'},
            type="upstream",
            says='a group is empty (near char ")" at position 10)',
            tools=frozenset({"internet_archive_search"}),
        ),
        ErrorBody(
            source=f"{_ARCHIVE_METADATA} the meaning the page gives code 104)",
            status=200,
            body={"error": "Item was deleted", "errcode": 104},
            type="not_found",
            says="Item was deleted",
            tools=frozenset({"internet_archive_item"}),
        ),
        ErrorBody(
            source=f"{_ARCHIVE_METADATA} the meaning the page gives code 102)",
            status=200,
            body={
                "error": "Item is unavailable (data node(s) are offline or not responding)",
                "errcode": 102,
            },
            type="upstream",
            says="Item is unavailable",
            tools=frozenset({"internet_archive_item"}),
        ),
    ),
    "_open_library": (
        ErrorBody(
            source=(
                "https://openlibrary.org/developers/api (one request a second without an email; "
                "violations are rate limited)"
            ),
            status=429,
            type="rate_limited",
            says="HTTP 429",
        ),
        ErrorBody(
            source="live, 2026-09-30 (a deleted work, HTTP 200)",
            status=200,
            body={"key": "/works/OL1000619W", "type": {"key": "/type/delete"}, "revision": 2},
            type="not_found",
            says="Open Library deleted /works/OL1000619W",
            tools=_LIBRARY_ITEMS,
        ),
    ),
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
    # T07a, health.
    "_clinical_trials": (
        ErrorBody(
            source=(
                "https://clinicaltrials.gov/api/oas/v2 (a 400 is a text/plain errorMessage; the "
                "words are illustrative)"
            ),
            status=400,
            body="filter.overallStatus: unknown value 'RECRUITNG'",
            type="validation_error",
            says="unknown value 'RECRUITNG'",
        ),
    ),
    "_rxnorm_dailymed": (
        ErrorBody(
            source=(
                "https://lhncbc.nlm.nih.gov/RxNav/news/API-Changes-202107.html (a 400 for "
                "invalid parameters)"
            ),
            status=400,
            type="validation_error",
            says="HTTP 400",
            tools=frozenset(
                {"rxnorm_drug_search", "rxnorm_concept", "rxnorm_related", "rxnorm_ndcs"}
            ),
        ),
        ErrorBody(
            source=(
                "https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm (errors by "
                "status: 404, 415, 5xx)"
            ),
            status=415,
            type="upstream",
            says="415",
            tools=frozenset({"dailymed_label_search", "dailymed_label", "dailymed_label_text"}),
        ),
    ),
    "_openfda_food": (
        ErrorBody(
            source="https://github.com/FDA/openfda/blob/master/api/faers/api_request.js",
            status=400,
            body={"error": {"code": "BAD_REQUEST", "message": "Skip value must 25000 or less."}},
            type="validation_error",
            says="Skip value must 25000 or less.",
        ),
        ErrorBody(
            source="https://github.com/FDA/openfda/blob/master/api/faers/api.js",
            status=500,
            body={
                "error": {"code": "SERVER_ERROR", "message": "Check your request and try again"}
            },
            type="upstream",
            says="Check your request and try again",
        ),
    ),
    "_open_food_facts": (
        ErrorBody(
            source="https://openfoodfacts.github.io/openfoodfacts-server/api/#rate-limits",
            status=503,
            type="rate_limited",
            says="HTTP 503",
        ),
    ),
    "_foodon": (
        ErrorBody(
            source=(
                "https://github.com/EBISPOT/ols4/blob/dev/backend/src/main/java/uk/ac/ebi/spot/ols/"
                "controller/api/exception/GlobalExceptionHandler.java"
            ),
            status=400,
            body={
                "status": 400,
                "message": "Failed to convert value of type 'java.lang.String' to required type "
                "'java.lang.Integer'",
            },
            type="upstream",
            says="Failed to convert value of type 'java.lang.String'",
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
