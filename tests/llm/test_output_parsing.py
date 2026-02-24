"""Tests for output parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_arch_toolkit._legacy.llm._output_parsing import (
    extract_code_block,
    extract_list,
    parse_json,
    parse_json_as,
)


@dataclass(frozen=True, slots=True)
class Person:
    name: str
    age: int


def test_parse_json_from_plain_text() -> None:
    parsed = parse_json('{"name":"Ana","age":30}')
    assert parsed == {"name": "Ana", "age": 30}


def test_parse_json_from_fenced_code_block() -> None:
    text = 'Result:\n```json\n{"ok": true, "items": [1,2]}\n```'
    parsed = parse_json(text)
    assert parsed == {"ok": True, "items": [1, 2]}


def test_parse_json_extracts_embedded_object() -> None:
    text = 'The answer is: {"a": 1, "b": [2,3]} thanks.'
    parsed = parse_json(text)
    assert parsed == {"a": 1, "b": [2, 3]}


def test_parse_json_rejects_scalars() -> None:
    with pytest.raises(ValueError, match="object or array"):
        parse_json("123")


def test_parse_json_as_dataclass() -> None:
    person = parse_json_as('{"name":"Ana","age":30}', Person)
    assert person == Person(name="Ana", age=30)


def test_extract_code_block_with_language_filter() -> None:
    text = "```python\nprint('x')\n```\n```json\n{\"k\":1}\n```"
    assert extract_code_block(text, language="json") == '{"k":1}\n'
    assert extract_code_block(text, language="python") == "print('x')\n"


def test_extract_code_block_without_match() -> None:
    assert extract_code_block("no fences") is None


def test_extract_list_for_bullets_and_numbers() -> None:
    text = """
    - first
    2. second
    3) third
    plain fallback
    """
    assert extract_list(text) == ["first", "second", "third", "plain fallback"]
