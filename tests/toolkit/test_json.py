"""Tests for toolkit/tools/_json.py."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._json import csv_read, json_extract


class TestJsonExtract:
    def test_simple_key(self):
        assert json_extract('{"name": "Alice"}', "name") == "Alice"

    def test_nested_path(self):
        j = '{"user": {"address": {"city": "NYC"}}}'
        assert json_extract(j, "user.address.city") == "NYC"

    def test_array_index(self):
        j = '{"items": [10, 20, 30]}'
        assert json_extract(j, "items[1]") == "20"

    def test_nested_array(self):
        j = '{"data": [{"name": "a"}, {"name": "b"}]}'
        assert json_extract(j, "data[1].name") == "b"

    def test_returns_json_for_objects(self):
        j = '{"user": {"a": 1, "b": 2}}'
        result = json_extract(j, "user")
        assert '"a": 1' in result
        assert '"b": 2' in result

    def test_invalid_json(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract("not json", "key")
        assert caught.value.error.type == "validation_error"
        assert "invalid JSON" in caught.value.error.message

    def test_missing_key(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract('{"a": 1}', "b")
        assert caught.value.error.type == "not_found"
        assert "'b'" in caught.value.error.message

    def test_index_out_of_range(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract("[1, 2]", "[5]")
        assert caught.value.error.type == "not_found"

    def test_indexing_a_scalar(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract('{"a": 1}', "a.b")
        assert caught.value.error.type == "validation_error"
        assert "cannot index" in caught.value.error.message


class TestCsvRead:
    def test_basic_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        result = csv_read(str(f))
        assert "name" in result
        assert "Alice" in result
        assert "Bob" in result
        assert " | " in result  # table separator
        assert "---" in result  # header separator

    def test_truncation(self, tmp_path):
        f = tmp_path / "big.csv"
        lines = ["id,value"] + [f"{i},{i * 10}" for i in range(200)]
        f.write_text("\n".join(lines))
        result = csv_read(str(f), max_rows=5)
        assert "Showing 5" in result

    def test_file_not_found(self):
        with pytest.raises(ToolFailure) as caught:
            csv_read("/nonexistent.csv")
        assert caught.value.error.type == "not_found"
        assert "/nonexistent.csv" in caught.value.error.message

    def test_directory_is_not_a_file(self, tmp_path):
        with pytest.raises(ToolFailure) as caught:
            csv_read(str(tmp_path))
        assert caught.value.error.type == "validation_error"
        assert "not a file" in caught.value.error.message

    def test_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        result = csv_read(str(f))
        assert "Empty" in result


class TestBounds:
    def test_deeply_nested_json_is_a_validation_error(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract("[" * 100_000, "a")
        assert caught.value.error.type == "validation_error"

    def test_csv_rows_are_clamped_and_os_errors_are_failures(self, tmp_path, monkeypatch):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4\n")

        assert "[Showing 1 of" in csv_read(str(f), max_rows=-1)

        def fail_stat(*args, **kwargs):
            raise OSError(errno.ENAMETOOLONG, "File name too long")

        monkeypatch.setattr(Path, "stat", fail_stat)

        with pytest.raises(ToolFailure) as caught:
            csv_read("too-long")
        assert caught.value.error.type == "validation_error"
        assert "cannot read" in caught.value.error.message


def test_a_file_that_is_not_readable_csv_is_a_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "huge.csv"
    path.write_text('"' + "x" * 200_000 + '"\n')

    with pytest.raises(ToolFailure) as caught:
        csv_read(str(path))

    assert caught.value.error.type == "validation_error"
    assert "not readable CSV" in caught.value.error.message
