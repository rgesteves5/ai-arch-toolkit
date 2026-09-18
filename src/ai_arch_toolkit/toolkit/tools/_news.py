"""News tools — Hacker News top stories (free, no API key)."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(base="https://hacker-news.firebaseio.com/v0", name="Hacker News")


@tool(capability="network")
def hacker_news(count: int = 5) -> str:
    """Get the top stories from Hacker News.

    Uses the official HN API (free, no API key).

    Args:
        count: Number of stories to return (1-30). Defaults to 5.
    """
    count = max(1, min(count, 30))

    try:
        story_ids = _API.get_json_list(
            "topstories.json", parse=lambda ids: [str(story_id) for story_id in ids[:count]]
        )
    except HttpError as e:
        return f"Failed to fetch HN top stories: {e}"

    stories: list[str] = []
    for story_id in story_ids:
        story = _story(story_id)
        if story is None:
            continue
        stories.append(f"  {len(stories) + 1}. {story}")

    if not stories:
        return "Failed to fetch any stories."

    return f"Hacker News — Top {len(stories)} stories:\n\n" + "\n\n".join(stories)


def _story(story_id: str) -> str | None:
    """The story's lines after its number; ``None`` when its item could not be fetched."""
    try:
        return _API.get_json(
            "item", f"{story_id}.json", parse=lambda item: _story_text(item, story_id)
        )
    except HttpError:
        return None


def _story_text(item: dict[str, Any], story_id: str) -> str:
    title = item.get("title", "Untitled")
    link = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
    score = item.get("score", 0)
    by = item.get("by", "unknown")
    comments = item.get("descendants", 0)

    return f"{title}\n     {link}\n     {score} points by {by} | {comments} comments"
