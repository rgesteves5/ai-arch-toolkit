"""Knowledge tools — Wikipedia and dictionary lookups (free, no API key)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ai_arch_toolkit.core import tool

_TIMEOUT = 10


@tool
def wikipedia_search(query: str, results: int = 3) -> str:
    """Search Wikipedia and return article titles with summaries.

    Args:
        query: The search query.
        results: Number of results to return (1-10). Defaults to 3.
    """
    results = max(1, min(results, 10))
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={urllib.request.quote(query)}"
        f"&srlimit={results}&format=json&utf8=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-arch-toolkit/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Wikipedia search failed: {e}"

    items = data.get("query", {}).get("search", [])
    if not items:
        return f"No Wikipedia results for: {query!r}"

    lines: list[str] = []
    for item in items:
        title = item.get("title", "")
        # Strip HTML tags from snippet
        snippet = item.get("snippet", "")
        snippet = _strip_html(snippet)
        lines.append(f"  - {title}: {snippet}")

    return f"Wikipedia results for {query!r}:\n" + "\n".join(lines)


@tool
def wikipedia_article(title: str, max_chars: int = 4000) -> str:
    """Get the summary extract of a Wikipedia article.

    Args:
        title: Exact article title, e.g. "Python (programming language)".
        max_chars: Maximum characters to return. Defaults to 4000.
    """
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={urllib.request.quote(title)}"
        f"&prop=extracts&exintro=1&explaintext=1&format=json&utf8=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-arch-toolkit/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Wikipedia API failed: {e}"

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return f"Article not found: {title!r}"
        extract = page.get("extract", "")
        if not extract:
            return f"No extract available for: {title!r}"
        if len(extract) > max_chars:
            return extract[:max_chars] + "\n\n[Truncated]"
        return f"{page.get('title', title)}:\n{extract}"

    return f"Article not found: {title!r}"


@tool
def define_word(word: str) -> str:
    """Look up a word definition using the Free Dictionary API.

    Args:
        word: The word to define.
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.request.quote(word)}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Word not found: {word!r}"
        return f"Dictionary API error: {e.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"Dictionary API failed: {e}"

    if not isinstance(data, list) or not data:
        return f"No definitions found for: {word!r}"

    entry = data[0]
    phonetic = entry.get("phonetic", "")
    lines: list[str] = []
    lines.append(f"{word}" + (f"  {phonetic}" if phonetic else ""))

    for meaning in entry.get("meanings", []):
        pos = meaning.get("partOfSpeech", "")
        lines.append(f"\n  {pos}:")
        for defn in meaning.get("definitions", [])[:3]:
            d = defn.get("definition", "")
            lines.append(f"    - {d}")
            example = defn.get("example")
            if example:
                lines.append(f"      Example: {example!r}")

    return "\n".join(lines)


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    import re

    return re.sub(r"<[^>]+>", "", text)
