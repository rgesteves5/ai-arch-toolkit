"""Dictionary tools — word definitions via a free public API."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(base="https://api.dictionaryapi.dev/api/v2/entries/en", name="Free Dictionary API")


@tool(capability="network")
def define_word(word: str) -> str:
    """Look up a word definition using the Free Dictionary API.

    Args:
        word: The word to define.
    """
    try:
        return _API.get_json_list(word, parse=lambda data: _definition_text(data, word))
    except HttpError as e:
        if e.status == 404:
            return f"Word not found: {word!r}"
        if e.status is not None:
            return f"Dictionary API error: {e.status}"
        return f"Dictionary API failed: {e}"


def _definition_text(data: list[Any], word: str) -> str:
    if not data:
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
