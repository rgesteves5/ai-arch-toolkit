"""Tests for toolkit/tools/_json.py."""

from __future__ import annotations

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
        result = json_extract("not json", "key")
        assert "Invalid JSON" in result

    def test_missing_key(self):
        result = json_extract('{"a": 1}', "b")
        assert "Path error" in result

    def test_index_out_of_range(self):
        result = json_extract("[1, 2]", "[5]")
        assert "Path error" in result


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
        result = csv_read("/nonexistent.csv")
        assert "not found" in result.lower()

    def test_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        result = csv_read(str(f))
        assert "Empty" in result
