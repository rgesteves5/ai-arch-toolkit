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


ERROR_BODIES: dict[str, tuple[ErrorBody, ...]] = {
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
    "_mediawiki": _mediawiki(),
    "_wikipedia": _mediawiki("wikipedia_search", "wikipedia_article", "wikipedia_related"),
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
