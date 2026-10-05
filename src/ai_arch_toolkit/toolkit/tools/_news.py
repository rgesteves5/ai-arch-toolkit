"""News tools — Hacker News top stories (free, no API key)."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api

_API = Api(base="https://hacker-news.firebaseio.com/v0", name="Hacker News")


@tool(capability="network")
def hacker_news(count: int = 5) -> str:
    """Get the top stories from Hacker News.

    Uses the official HN API (free, no API key).

    Args:
        count: Number of stories to return (1-30). Defaults to 5.

    Raises:
        ToolFailure: upstream (or rate_limited) when the list of top stories cannot be fetched,
            or when none of its stories can; a story that fails while others load is named in
            the answer instead.
    """
    count = max(1, min(count, 30))

    story_ids = _API.get_json_list(
        "topstories.json", parse=lambda ids: [str(story_id) for story_id in ids[:count]]
    )

    stories: list[str] = []
    missed: list[str] = []
    failures: list[ToolFailure] = []
    for rank, story_id in enumerate(story_ids, start=1):
        try:
            stories.append(f"  {rank}. {_story(story_id)}")
        except ToolFailure as e:  # the others still answer; the text names this one
            missed.append(f"#{rank} (item {story_id}: {e})")
            failures.append(e)

    if failures and not stories:
        limited = any(failure.error.type == "rate_limited" for failure in failures)
        retryable = any(failure.error.retryable for failure in failures)
        raise ToolFailure(
            "rate_limited" if limited else "upstream",
            "could not load any of the Hacker News top stories: "
            + "; ".join(missed)
            + ("; try again later" if retryable else ""),
            retryable=retryable,
        )
    text = f"Hacker News — Top {len(story_ids)} stories:\n\n" + "\n\n".join(stories)
    if missed:
        text += f"\n\nCould not load {len(missed)} of them: " + "; ".join(missed)
    return text


def _story(story_id: str) -> str:
    """The story's lines after its number."""
    return _API.get_json(
        "item", f"{story_id}.json", parse=lambda item: _story_text(item, story_id)
    )


def _story_text(item: dict[str, Any], story_id: str) -> str:
    title = item.get("title", "Untitled")
    link = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
    score = item.get("score", 0)
    by = item.get("by", "unknown")
    comments = item.get("descendants", 0)

    return f"{title}\n     {link}\n     {score} points by {by} | {comments} comments"
