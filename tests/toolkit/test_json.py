"""Tests for toolkit/tools/_json.py."""

from __future__ import annotations

import errno
import re
from pathlib import Path
from typing import Any

import pytest

from ai_arch_toolkit.core import ApprovalDecision, ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _json
from ai_arch_toolkit.toolkit.tools._json import csv_read, json_extract


def _text(result: ToolResult) -> str:
    assert result.ok and isinstance(result.value, str), result
    return result.value


def _window(result: ToolResult) -> dict[str, Any]:
    return result.metadata["window"]


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

    def test_a_missing_key_is_not_found_and_names_the_keys_there(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract('{"user": {"name": "a", "id": 1}}', "user.email")
        assert caught.value.error.type == "not_found"
        assert "'email'" in caught.value.error.message
        assert "keys 'name', 'id'" in caught.value.error.message

    def test_many_keys_are_named_with_how_many_more(self):
        many = "{" + ", ".join(f'"k{n}": {n}' for n in range(30)) + "}"
        with pytest.raises(ToolFailure) as caught:
            json_extract(many, "missing")
        assert "'k19' and 10 more" in caught.value.error.message

    def test_an_index_out_of_range_is_not_found_and_gives_the_length(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract("[1, 2]", "[5]")
        assert caught.value.error.type == "not_found"
        assert "2 items (indexes 0 to 1)" in caught.value.error.message

    def test_an_empty_list_or_object_says_it_is_empty(self):
        for document, path, kind in (("[]", "[0]", "list"), ('{"a": {}}', "a.b", "object")):
            with pytest.raises(ToolFailure) as caught:
                json_extract(document, path)
            assert caught.value.error.message.endswith(f"the {kind} there is empty")

    def test_indexing_a_scalar(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract('{"a": 1}', "a.b")
        assert caught.value.error.type == "validation_error"
        assert "cannot index" in caught.value.error.message


class TestCsvRead:
    def test_basic_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        result = _text(csv_read(str(f)))
        assert result == (f"{f} (2 rows):\nname  | age\n------+----\nAlice | 30 \nBob   | 25 ")

    def test_a_page_has_the_header_and_the_next_offset(self, tmp_path):
        f = tmp_path / "big.csv"
        f.write_text("id,value\n" + "".join(f"{i},{i * 10}\n" for i in range(200)))

        first = csv_read(str(f), max_rows=5)
        later = csv_read(str(f), offset=195, max_rows=5)

        assert _text(first).splitlines()[1:3] == ["id | value", "---+------"]
        assert _text(first).endswith("4  | 40   \n[results 1-5 of 200 | next: offset=5]")
        assert _text(later).splitlines()[1] == "id  | value"
        assert _text(later).endswith("199 | 1990 \n[results 196-200 of 200 | end]")

    def test_the_total_is_the_whole_files_past_the_old_read_limit(self, tmp_path):
        f = tmp_path / "long.csv"
        f.write_text("id,text\n" + "".join(f"{i},{'x' * 30}\n" for i in range(60_000)))

        assert _window(csv_read(str(f), max_rows=10))["total"] == 60_000

    def test_a_quoted_field_over_several_lines_is_one_row(self, tmp_path):
        f = tmp_path / "notes.csv"
        f.write_text('id,note\n1,"first\nsecond\nthird"\n2,plain\n')

        result = csv_read(str(f), max_rows=1)

        assert _window(result)["total"] == 2
        assert _window(result)["next_call"] == {"offset": 1}

    def test_following_the_footers_reads_every_row_once(self, tmp_path):
        f = tmp_path / "rows.csv"
        f.write_text("n\n" + "".join(f"{i}\n" for i in range(250)))

        rows, call = [], {"offset": 0}
        while call is not None:
            result = csv_read(str(f), max_rows=100, **call)
            rows += [line.strip() for line in _text(result).splitlines()[3:] if line[0] != "["]
            call = _window(result)["next_call"]

        assert rows == [str(i) for i in range(250)]

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
        assert _text(csv_read(str(f))) == "Empty CSV file."

    def test_a_header_without_rows(self, tmp_path):
        f = tmp_path / "header.csv"
        f.write_text("a,b\n")
        assert _text(csv_read(str(f))) == f"{f} (0 rows):\na | b\n--+--"


class TestCsvPageBounds:
    """A page is bounded by characters, not only by rows; a cell is padded only up to a cap; and
    the rows are counted only so far past the page."""

    @staticmethod
    def _wide(tmp_path: Path) -> Path:
        """A 9,999-row CSV whose second row has one 130,000-character field."""
        rows = [f"{n},short {n}" for n in range(9_999)]
        rows[1] = f"1,{'w' * 130_000}"
        path = tmp_path / "wide.csv"
        path.write_text("id,text\n" + "\n".join(rows) + "\n")
        return path

    def test_a_page_stops_before_a_row_that_would_pass_its_characters(self, tmp_path):
        path = self._wide(tmp_path)

        first = _text(csv_read(str(path), max_rows=10_000))

        assert len(first) - len(str(path)) < 150
        assert first.endswith("[results 1-1 of 9999 | next: offset=1]")

    def test_a_row_longer_than_a_page_comes_alone(self, tmp_path):
        alone = _text(csv_read(str(self._wide(tmp_path)), offset=1, max_rows=10_000))

        assert "w" * 130_000 in alone and len(alone) < 130_300
        assert alone.endswith("[results 2-2 of 9999 | next: offset=2]")

    def test_the_short_rows_after_it_fill_a_page_of_characters(self, tmp_path):
        later = _text(csv_read(str(self._wide(tmp_path)), offset=2, max_rows=10_000))

        assert _json._PAGE_CHARS - 100 < len(later) < _json._PAGE_CHARS + 300
        assert max(len(line) for line in later.splitlines()[1:]) < 60  # after the path

    def test_following_the_footers_reads_every_row_once(self, tmp_path):
        rows, call = [], {"offset": 0}
        while call is not None:
            result = csv_read(str(self._wide(tmp_path)), max_rows=10_000, **call)
            rows += [line.split("|")[0].strip() for line in _text(result).splitlines()[3:-1]]
            call = _window(result)["next_call"]

        assert rows == [str(n) for n in range(9_999)]

    def test_cells_are_padded_only_up_to_the_column_cap(self, tmp_path):
        path = tmp_path / "padded.csv"
        path.write_text(f"name,note\nlong,{'n' * 1000}\nshort,x\n")

        lines = _text(csv_read(str(path))).splitlines()

        assert lines[1] == "name  | " + "note".ljust(_json._PAD_CHARS)
        assert lines[-1] == "short | " + "x".ljust(_json._PAD_CHARS)
        assert lines[-2] == "long  | " + "n" * 1000

    def test_the_count_stops_past_its_cap_and_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_json, "_COUNT_CHARS", 1_000)
        path = tmp_path / "many.csv"
        path.write_text("n\n" + "".join(f"{n}\n" for n in range(10_000)))

        first = csv_read(str(path), max_rows=10)
        second = csv_read(str(path), offset=10, max_rows=10)

        heading = re.match(r"(.*) \(at least (\d+) rows, counted up to 1000 ", _text(first))
        assert heading is not None and heading[1] == str(path) and int(heading[2]) > 10
        assert _window(first)["total"] is None
        assert _text(first).endswith("[results 1-10 | next: offset=10]")
        assert _text(second).splitlines()[3].strip() == "10"

    def test_the_count_still_reaches_the_end_of_a_file_within_the_cap(self, tmp_path):
        path = tmp_path / "rows.csv"
        path.write_text("n\n" + "".join(f"{n}\n" for n in range(5_000)))

        assert _window(csv_read(str(path), max_rows=10))["total"] == 5_000


class TestBounds:
    def test_deeply_nested_json_is_a_validation_error(self):
        with pytest.raises(ToolFailure) as caught:
            json_extract("[" * 100_000, "a")
        assert caught.value.error.type == "validation_error"

    def test_the_executor_refuses_rows_past_the_limits(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        group = ToolGroup(csv_read, approval_handler=lambda _r: ApprovalDecision.approve())

        for arguments in ({"max_rows": 0}, {"max_rows": 10_001}, {"offset": -1}):
            call = ToolCall(id="c", name="csv_read", input={"path": str(f), **arguments})
            result = group.execute(call)
            assert result.error is not None and result.error.type == "validation_error"

    def test_os_errors_are_failures(self, monkeypatch):
        def fail_stat(*args, **kwargs):
            raise OSError(errno.ENAMETOOLONG, "File name too long")

        monkeypatch.setattr(Path, "stat", fail_stat)

        with pytest.raises(ToolFailure) as caught:
            csv_read("too-long")
        assert caught.value.error.type == "validation_error"
        assert "cannot read" in caught.value.error.message


def test_a_file_that_is_not_readable_csv_is_a_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "huge.csv"
    path.write_text("a\n" + '"' + "x" * 200_000 + '"\n')

    with pytest.raises(ToolFailure) as caught:
        csv_read(str(path))

    assert caught.value.error.type == "validation_error"
    assert "not readable CSV" in caught.value.error.message
    assert "row 2" in caught.value.error.message
