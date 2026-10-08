"""Error answers as each source documents or sends them, for the contract test (point 2).

Each entry is one answer of the source; every network tool of the module it is listed under (or
the ``tools`` it names) must turn it into a ``ToolFailure`` of ``type`` whose message contains
``says``. A tool of a module with an ``Api`` and no entry that applies to it owes the point.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ai_arch_toolkit.core import ToolFailureType
from tests.toolkit import geo_answers
from tests.toolkit.contract_cases import Body
from tests.toolkit.data_bodies import WORLD_BANK_INVALID_VALUE


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
_EUROSTAT_GUIDE = (
    "https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/"
    "api-detailed-guidelines/api-statistics"
)
_EUROSTAT_DATA = frozenset({"eurostat_dataset", "eurostat_series"})
_WORLD_BANK_ERRORS = (
    "https://datahelpdesk.worldbank.org/knowledgebase/articles/898620-api-error-codes"
)


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


def _open_meteo(source: str, reason: str, *tools: str) -> ErrorBody:
    """Open-Meteo's refusal: HTTP 400 and ``{"error": true, "reason": ...}``."""
    return ErrorBody(
        source=source,
        status=400,
        body={"error": True, "reason": reason},
        type="validation_error",
        says=reason.rstrip("."),
        tools=frozenset(tools),
    )


_IPWHOIS_ERRORS = "https://ipwhois.io/documentation (Errors)"
_FORECAST_ERRORS = "https://open-meteo.com/en/docs (Errors)"
_OVERPASS_ERRORS = "https://dev.overpass-api.de/overpass-doc/en/preface/commons.html"
_FDSN_ERRORS = "https://www.fdsn.org/webservices/FDSN-WS-Specification-Commonalities-1.2.pdf"

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
    # Geo, weather and natural events (T08b).
    "_geo": (
        *_mediawiki("country_info"),
        _open_meteo(
            "https://open-meteo.com/en/docs/geocoding-api (Errors)",
            geo_answers.GEOCODING_REASON,
            "geocode",
        ),
        _open_meteo(_FORECAST_ERRORS, geo_answers.OPEN_METEO_REASON, "timezone_lookup"),
        ErrorBody(
            source=_IPWHOIS_ERRORS,
            status=200,
            body={"success": False, "message": "Reserved range"},
            type="validation_error",
            says="Reserved range",
            tools=frozenset({"ip_lookup"}),
        ),
        ErrorBody(
            source=_IPWHOIS_ERRORS,
            status=429,
            body={"success": False, "message": "Rate limit exceeded"},
            type="rate_limited",
            says="Rate limit exceeded",
            tools=frozenset({"ip_lookup"}),
        ),
    ),
    "_weather": (_open_meteo(_FORECAST_ERRORS, geo_answers.OPEN_METEO_REASON),),
    "_osm": (
        ErrorBody(
            source="Nominatim's _format_error and get_layers (github.com/osm-search/Nominatim)",
            status=400,
            body={"error": {"code": 400, "message": geo_answers.NOMINATIM_LAYER_MESSAGE}},
            type="validation_error",
            says=geo_answers.NOMINATIM_LAYER_MESSAGE,
        ),
    ),
    "_overpass": (
        ErrorBody(
            source=_OVERPASS_ERRORS,
            status=400,
            body=geo_answers.overpass_page("line 1: parse error: ';' expected - ')' found."),
            type="validation_error",
            says="line 1: parse error",
        ),
        ErrorBody(
            source="live, 2026-09-29 (overpass-api.de, a runtime error sent with HTTP 200)",
            status=200,
            body={"elements": [], "remark": geo_answers.OVERPASS_TIMEOUT},
            type="upstream",
            says="runtime error: Query timed out",
        ),
    ),
    "_eonet": (
        ErrorBody(
            source="live, 2026-09-30 (EONET's page for an unknown event ID; its documentation, "
            "https://eonet.gsfc.nasa.gov/docs/v3, describes no error answers)",
            status=500,
            body=geo_answers.EONET_ERROR_PAGE,
            type="upstream",
            says="HTTP error 500",
        ),
    ),
    "_earthquake": (
        ErrorBody(
            source=_FDSN_ERRORS,
            status=400,
            body=geo_answers.fdsn_error(400, "Bad Request", geo_answers.USGS_BAD_START),
            type="validation_error",
            says=geo_answers.USGS_BAD_START.rstrip("."),
        ),
    ),
    "_wikidata": (
        *_mediawiki("wikidata_search"),
        ErrorBody(
            source="https://www.wikidata.org/wiki/Wikidata:Data_access (a 429 asks to wait)",
            status=429,
            headers={"Retry-After": "1"},
            type="rate_limited",
            says="HTTP 429",
            tools=frozenset({"wikidata_entity", "wikidata_sparql"}),
        ),
        ErrorBody(
            source="Blazegraph's parse error, with a 400 (a malformed query)",
            status=400,
            body="SPARQL-QUERY: queryStr=SELECT ?x WHERE {\njava.util.concurrent."
            "ExecutionException: org.openrdf.query.MalformedQueryException: Encountered "
            '"<EOF>" at line 1.',
            type="validation_error",
            says='Encountered "<EOF>" at line 1.',
            tools=frozenset({"wikidata_sparql"}),
        ),
        ErrorBody(
            source="https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual"
            "#Query_limits (the 60 s deadline)",
            status=500,
            body="SPARQL-QUERY: queryStr=SELECT ?x\njava.util.concurrent.TimeoutException",
            type="upstream",
            says="TimeoutException",
            tools=frozenset({"wikidata_sparql"}),
        ),
    ),
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
        ),
        ErrorBody(
            source=_EUROSTAT_GUIDE,
            status=200,
            body={
                "warning": {
                    "status": 413,
                    "label": "ASYNCHRONOUS_RESPONSE. Your request will be treated "
                    "asynchronously. Please try again later.",
                }
            },
            type="upstream",
            says="ASYNCHRONOUS_RESPONSE",
            tools=_EUROSTAT_DATA,
        ),
        ErrorBody(
            source=_EUROSTAT_GUIDE + " (error 100 with a 400: the result is empty)",
            status=400,
            body={"error": [{"status": 400, "id": 100, "label": "No results found"}]},
            type="not_found",
            says="No results found",
            tools=_EUROSTAT_DATA,
        ),
        ErrorBody(
            source="https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/"
            "api-faq (error 150)",
            status=400,
            body={
                "error": [
                    {
                        "status": 400,
                        "id": 150,
                        "label": "INVALID_QUERY_DIMENSION_VALUE: Query is invalid as per its "
                        "structure's definition. The following values for dimension are not "
                        "allowed: GEO=EU27.",
                    }
                ]
            },
            type="validation_error",
            says="INVALID_QUERY_DIMENSION_VALUE",
            tools=_EUROSTAT_DATA,
        ),
    ),
    "_world_bank": (
        ErrorBody(
            source=_WORLD_BANK_ERRORS + " (error 120, sent with HTTP 200, 2026-09-29)",
            status=200,
            body=WORLD_BANK_INVALID_VALUE,
            type="validation_error",
            says="Invalid value: The provided parameter value is not valid",
            tools=frozenset(
                {
                    "world_bank_topics",
                    "world_bank_sources",
                    "world_bank_countries",
                    "world_bank_indicators",
                }
            ),
        ),
        ErrorBody(
            source=_WORLD_BANK_ERRORS + " (error 120, sent with HTTP 200, 2026-09-29)",
            status=200,
            body=WORLD_BANK_INVALID_VALUE,
            type="not_found",
            says="Invalid value: The provided parameter value is not valid",
            tools=frozenset({"world_bank_indicator", "world_bank_series"}),
        ),
        ErrorBody(
            source=_WORLD_BANK_ERRORS + " (error 105)",
            status=200,
            body=[
                {
                    "message": [
                        {
                            "id": "105",
                            "key": "Service currently unavailable",
                            "value": "The requested service is temporarily unavailable.",
                        }
                    ]
                }
            ],
            type="upstream",
            says="The requested service is temporarily unavailable",
        ),
    ),
    "_who_gho": (
        ErrorBody(
            source="https://docs.oasis-open.org/odata/odata-json-format/v4.01/"
            "odata-json-format-v4.01.html#sec_ErrorResponse (the GHO API is OData)",
            status=400,
            body={
                "error": {
                    "code": "",
                    "message": "The query specified in the URI is not valid. Could not find a "
                    "property named 'Dim9' on type 'Default.FACT'.",
                }
            },
            type="validation_error",
            says="The query specified in the URI is not valid",
        ),
    ),
    "_gdelt": (
        ErrorBody(
            source="live, 2026-09-29 (text in place of the JSON, with HTTP 200)",
            status=200,
            body="Your query was too short or too long.\n",
            type="validation_error",
            says="Your query was too short or too long",
        ),
        ErrorBody(
            source="live, 2026-10-04 (https://github.com/cyanheads/gdelt-mcp-server/issues/44)",
            status=429,
            body="Please limit requests to one every 5 seconds.",
            type="rate_limited",
            says="Please limit requests to one every 5 seconds",
        ),
    ),
    "_news": (
        ErrorBody(
            source="https://firebase.google.com/docs/reference/rest/database"
            "#section-error-conditions",
            status=401,
            body={"error": "Permission denied"},
            type="upstream",
            says="Permission denied",
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
    # A 400 is a request "the client request needs modification": the caller's to fix
    # (https://www.uniprot.org/help/rest-api-headers).
    "_uniprot": (
        ErrorBody(
            source="live, 2026-09-30 (https://rest.uniprot.org/uniprotkb/search)",
            status=400,
            body={
                "url": "http://rest.uniprot.org/uniprotkb/search",
                "messages": ["Invalid request received. Unsupported query."],
            },
            type="validation_error",
            says="Unsupported query",
        ),
        ErrorBody(
            source="https://www.uniprot.org/help/rest-api-headers (400 Bad request)",
            status=400,
            body={
                "url": "https://rest.uniprot.org/uniprotkb/search",
                "messages": ["'query' is a required parameter"],
            },
            type="validation_error",
            says="'query' is a required parameter",
        ),
    ),
    "_pdb": (
        ErrorBody(
            source=(
                "the search's 400, as T02 recorded it (tests/toolkit/test_pdb.py); RCSB's own "
                "client reads a 400's body as the reason (https://github.com/rcsb/rcsb-mcp, "
                "src/rcsb_mcp/client.py)"
            ),
            status=400,
            body={"status": 400, "message": "JSON schema validation failed for query"},
            type="validation_error",
            says="JSON schema validation failed for query",
        ),
        # GraphQL answers 200 and puts its errors in the body
        # (https://data.rcsb.org/index.html#gql-api; https://github.com/rcsb/py-rcsb-api,
        # rcsbapi/data/data_query.py).
        ErrorBody(
            source="https://data.rcsb.org/index.html#gql-api",
            status=200,
            body={"errors": [{"message": "Field 'x' in type 'CoreEntry' is undefined"}]},
            type="upstream",
            says="Field 'x' in type 'CoreEntry' is undefined",
            tools=frozenset({"pdb_search"}),
        ),
    ),
    # A refused request answers 400 with its reason in ``error_message``
    # (https://github.com/chembl/chembl_webservices_py3, src/chembl_webservices/core/resource.py).
    "_chembl": (
        ErrorBody(
            source="chembl_webservices_py3, core/resource.py (check_user_search_query)",
            status=400,
            body={"error_message": "Search query too short"},
            type="validation_error",
            says="Search query too short",
            tools=frozenset(
                {"chembl_molecule_search", "chembl_target_search", "chembl_activity_search"}
            ),
        ),
        ErrorBody(
            source="chembl_webservices_py3, core/resource.py (resource lookup)",
            status=400,
            body={"error_message": "Invalid resource lookup data provided (mismatched type)."},
            type="validation_error",
            says="Invalid resource lookup data provided (mismatched type)",
            tools=frozenset({"chembl_molecule", "chembl_target"}),
        ),
    ),
    # A refused request answers 400 with its reason as plain text
    # (https://github.com/gbif/gbif-common-ws, IllegalArgumentExceptionMapper; the reason from
    # https://github.com/gbif/checklistbank, SpeciesResource.checkDeepPaging).
    "_gbif": (
        ErrorBody(
            source="gbif-common-ws IllegalArgumentExceptionMapper; checklistbank SpeciesResource",
            status=400,
            body="Offset is limited for this operation to 100000",
            type="validation_error",
            says="Offset is limited for this operation to 100000",
        ),
    ),
}
