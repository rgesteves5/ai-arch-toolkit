"""The MediaWiki Action API that the wiki tools share: its error reader and the wikis allowed."""

from __future__ import annotations

import re
from dataclasses import replace

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply


def mediawiki_error(reply: Reply) -> ToolFailure | str | None:
    """The error a MediaWiki API answer reports; ``None`` for a result.

    MediaWiki answers errors with an ``error`` object (``code`` and ``info``) in place of the
    result, usually with HTTP 200, and names the code in the ``MediaWiki-API-Error`` header
    (https://www.mediawiki.org/wiki/API:Errors_and_warnings), so every ``Api`` on an ``api.php``
    declares this as its ``error_reader``. The codes that say what happened are typed:
    ``missingtitle`` is ``not_found``; ``invalidtitle`` and ``pagecannotexist`` (a special page)
    are ``validation_error`` (https://www.mediawiki.org/wiki/API:Parse); ``ratelimited`` and
    ``maxlag`` are ``rate_limited`` and worth a retry
    (https://www.mediawiki.org/wiki/Manual:Maxlag_parameter); any other code is the source's
    words, ``code: info``.
    """
    error = reply.body.get("error") if isinstance(reply.body, dict) else None
    if isinstance(error, dict):
        code, info = _string(error.get("code")), _string(error.get("info"))
    elif header := _string(reply.headers.get("mediawiki-api-error")):
        code, info = header, ""
    else:
        return None
    said = ": ".join(text for text in (code, info) if text) or "unknown error"
    brief = said.rstrip(".")
    if code == "missingtitle":
        return ToolFailure(
            "not_found", f"no such page ({brief}); find the exact title with the wiki's search"
        )
    if code == "invalidtitle":
        return ToolFailure(
            "validation_error",
            f"not a valid page title ({brief}); give a title as the wiki's search returns it",
        )
    if code == "pagecannotexist":
        return ToolFailure(
            "validation_error",
            f"not a page the wiki can hold ({brief}); give an article's title, not a special page",
        )
    if code in _RATE_LIMITED:
        return ToolFailure(
            "rate_limited",
            f"the wiki asked to slow down ({brief}); try again later",
            retryable=True,
        )
    return said


# The codes of a wiki that asks callers to wait: a user's action limit, or replication lag.
_RATE_LIMITED = frozenset({"ratelimited", "maxlag"})

# The wikis the tools may read: these hosts, or a subdomain of one (en.wikipedia.org).
WIKIMEDIA_DOMAINS = frozenset(
    {
        "wikipedia.org",
        "wikimedia.org",
        "wiktionary.org",
        "wikidata.org",
        "wikibooks.org",
        "wikiquote.org",
        "wikisource.org",
        "wikiversity.org",
        "wikivoyage.org",
        "wikinews.org",
        "mediawiki.org",
    }
)
_HOST_RE = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$")
_TIMEOUT_S = 15
# Wikimedia asks a client without an account for one request at a time, at most a few a second
# (https://meta.wikimedia.org/wiki/User-Agent_policy; https://www.mediawiki.org/wiki/API:Etiquette).
_INTERVAL_S = 0.25
# The largest answer a Wikimedia wiki sends (12 MiB; past it the page comes as a warning), with
# room for the JSON around it (https://www.mediawiki.org/wiki/Manual:$wgAPIMaxResultSize).
_MAX_BYTES = 13 * 2**20

WIKIPEDIA = "en.wikipedia.org"
WIKTIONARY = "en.wiktionary.org"
# The tools' own wikis: a 404 there is an endpoint that moved, not the caller's mistake.
_WIKIPEDIA_API = Api(
    base=f"https://{WIKIPEDIA}/w/api.php",
    name="Wikipedia",
    timeout_s=_TIMEOUT_S,
    max_bytes=_MAX_BYTES,
    min_interval_s=_INTERVAL_S,
    error_reader=mediawiki_error,
)
_WIKTIONARY_API = Api(
    base=f"https://{WIKTIONARY}/w/api.php",
    name="Wiktionary",
    timeout_s=_TIMEOUT_S,
    max_bytes=_MAX_BYTES,
    min_interval_s=_INTERVAL_S,
    error_reader=mediawiki_error,
)
_OWN = {WIKIPEDIA: _WIKIPEDIA_API, WIKTIONARY: _WIKTIONARY_API}


def wiki_host(wiki: str) -> str:
    """The host of the wiki ``wiki`` names: a host (``en.wikibooks.org``) or its URL.

    Raises:
        ToolFailure: validation_error unless it is a Wikimedia wiki.
    """
    host = wiki.strip().lower().removeprefix("https://").removeprefix("http://")
    host = host.split("/", 1)[0]
    allowed = any(host == domain or host.endswith(f".{domain}") for domain in WIKIMEDIA_DOMAINS)
    if not (_HOST_RE.fullmatch(host) and allowed):
        raise ToolFailure(
            "validation_error",
            f"invalid wiki {wiki[:200]!r}; name a Wikimedia wiki by its host, e.g. "
            "'en.wikipedia.org', 'pt.wikipedia.org' or 'en.wikibooks.org'",
        )
    return host


def wiki_api(wiki: str) -> Api:
    """The Action API (``https://host/w/api.php``) of the wiki ``wiki`` names.

    Raises:
        ToolFailure: validation_error unless it is a Wikimedia wiki.
    """
    host = wiki_host(wiki)
    if (own := _OWN.get(host)) is not None:
        return own
    api = Api.within(
        f"https://{host}/w/api.php",
        WIKIMEDIA_DOMAINS,
        name=host,
        timeout_s=_TIMEOUT_S,
        min_interval_s=_INTERVAL_S,
        error_reader=mediawiki_error,
    )
    return replace(api, max_bytes=_MAX_BYTES)


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
