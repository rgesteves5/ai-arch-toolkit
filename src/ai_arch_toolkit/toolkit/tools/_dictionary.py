"""Dictionary tools — word definitions via a free public API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ai_arch_toolkit.core import tool

_TIMEOUT = 10


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
            definition = defn.get("definition", "")
            lines.append(f"    - {definition}")
            example = defn.get("example")
            if example:
                lines.append(f"      Example: {example!r}")

    return "\n".join(lines)
