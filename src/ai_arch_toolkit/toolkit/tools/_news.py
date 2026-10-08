"""News tools: the Hacker News top stories (free, no API key).

The official API (https://github.com/HackerNews/API) lists up to 500 top stories at
``/v0/topstories``, and each story is one request (``/v0/item/{id}``); the list reads on by
``offset``, through the window (D39). Firebase, which serves it, explains an error as
``{"error": "…"}`` with the status
(https://firebase.google.com/docs/reference/rest/database#section-error-conditions); the door
reads those words.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Annotated, Any, NoReturn

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api
from ai_arch_toolkit.toolkit.tools._window import list_window

_API = Api(base="https://hacker-news.firebaseio.com/v0", name="Hacker News")
# The length of the top stories list (the API's README).
_LISTED = 500


@tool(capability="network")
def hacker_news(
    count: Annotated[int, Range(1, 30)] = 5,
    offset: Annotated[int, Range(0, _LISTED - 1)] = 0,
) -> ToolResult:
    """Get the Hacker News top stories, in rank order, with their links, points and comments.

    Args:
        count: How many stories to show (each one is a request to the API).
        offset: How many top stories to skip; the footer gives the next offset.

    Raises:
        ToolFailure: upstream (or rate_limited) when the list of top stories cannot be fetched,
            or when none of the stories shown can; a story that fails while others load is named
            in its place instead.
    """
    story_ids = _API.get_json_list(
        "topstories.json", parse=lambda ids: [str(story_id) for story_id in ids]
    )
    page = story_ids[offset : offset + count]
    lines, failures = _stories(page, first=offset + 1)
    if failures and len(failures) == len(page):
        _raise_all_failed(failures)
    end = offset + len(page)
    next_call = {"offset": end} if page and end < len(story_ids) else None
    window = list_window(lines, first=offset + 1, total=len(story_ids), next_call=next_call)
    return window.result(heading=f"Hacker News top stories ({len(story_ids)} on the list):")


def _stories(
    story_ids: list[str], *, first: int
) -> tuple[list[str], list[tuple[str, ToolFailure]]]:
    """Each story's lines, numbered from ``first``, and the stories that failed (rank text, the
    failure): a story that fails is named in its place, so the others still answer."""
    lines: list[str] = []
    failures: list[tuple[str, ToolFailure]] = []
    for rank, story_id in enumerate(story_ids, start=first):
        try:
            lines.append(f"{rank}. {_story(story_id)}")
        except ToolFailure as failure:
            lines.append(f"{rank}. (could not load item {story_id}: {failure})")
            failures.append((f"#{rank} (item {story_id}: {failure})", failure))
    return lines, failures


def _raise_all_failed(failures: list[tuple[str, ToolFailure]]) -> NoReturn:
    """No story of the page loaded: the failure says why, typed by the worst of them."""
    errors = [failure.error for _, failure in failures]
    limited = any(error.type == "rate_limited" for error in errors)
    retryable = any(error.retryable for error in errors)
    raise ToolFailure(
        "rate_limited" if limited else "upstream",
        "could not load any of the Hacker News top stories: "
        + "; ".join(missed for missed, _ in failures)
        + ("; try again later" if retryable else ""),
        retryable=retryable,
    )


def _story(story_id: str) -> str:
    """The story's lines after its number."""
    return _API.get_json(
        "item", f"{story_id}.json", parse=lambda item: _story_text(item, story_id)
    )


def _story_text(item: dict[str, Any], story_id: str) -> str:
    # Titles are HTML (the API's README); a story without a URL is its discussion page.
    title = html.unescape(str(item.get("title") or "(untitled)"))
    link = item.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
    details = [f"{item.get('score', 0)} points by {item.get('by', 'unknown')}"]
    if item.get("type", "story") == "story":
        details.append(f"{item.get('descendants', 0)} comments")
    else:
        details.append(str(item.get("type")))
    if isinstance(item.get("time"), int):  # Unix time, in seconds
        details.append(f"posted {_iso(item['time'])}")
    return f"{title}\n   {link}\n   " + " | ".join(details)


def _iso(seconds: int) -> str:
    try:
        return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):  # no time a story could have
        return str(seconds)
