"""News tools — Hacker News top stories (free, no API key)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ai_arch_toolkit.core import tool

_TIMEOUT = 10


@tool
def hacker_news(count: int = 5) -> str:
    """Get the top stories from Hacker News.

    Uses the official HN API (free, no API key).

    Args:
        count: Number of stories to return (1-30). Defaults to 5.
    """
    count = max(1, min(count, 30))

    try:
        req = urllib.request.Request("https://hacker-news.firebaseio.com/v0/topstories.json")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            story_ids = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Failed to fetch HN top stories: {e}"

    stories: list[str] = []
    for story_id in story_ids[:count]:
        url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                item = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue

        title = item.get("title", "Untitled")
        link = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
        score = item.get("score", 0)
        by = item.get("by", "unknown")
        comments = item.get("descendants", 0)

        stories.append(
            f"  {len(stories) + 1}. {title}\n"
            f"     {link}\n"
            f"     {score} points by {by} | {comments} comments"
        )

    if not stories:
        return "Failed to fetch any stories."

    return f"Hacker News — Top {len(stories)} stories:\n\n" + "\n\n".join(stories)
