"""Dictionary tools — word definitions via a free public API."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(base="https://api.dictionaryapi.dev/api/v2/entries/en", name="Free Dictionary API")


@tool(capability="network")
def define_word(word: str) -> str:
    """Look up a word definition using the Free Dictionary API.

    Args:
        word: The word to define.

    Raises:
        ToolFailure: validation_error when the word is blank; not_found when the Free
            Dictionary API has no entry for it.
    """
    if not word.strip():
        raise ToolFailure("validation_error", "word cannot be empty; pass an English word.")
    try:
        return _API.get_json_list(word, parse=lambda data: _definition_text(data, word))
    except HttpError as e:
        if e.status == 404:
            msg = (
                f"the Free Dictionary API has no entry for {word!r}; check the spelling, "
                "or look it up with wiktionary_entry."
            )
            raise ToolFailure("not_found", msg) from e
        raise


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
