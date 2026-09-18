"""Tests for toolkit/tools/_filesystem.py."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core import ApprovalDecision, ApprovalRequest, ToolCall, ToolGroup
from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files


class TestReadFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = read_file(str(f))
        assert "line1" in result
        assert "line3" in result

    def test_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(500)))
        result = read_file(str(f), max_lines=10)
        assert "Truncated" in result
        assert "500 total lines" in result

    def test_file_not_found(self):
        result = read_file("/nonexistent/path/file.txt")
        assert "not found" in result.lower()

    def test_directory_path(self, tmp_path):
        result = read_file(str(tmp_path))
        assert "Not a file" in result


class TestListDirectory:
    def test_list_files(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.txt").write_text("text")
        (tmp_path / "subdir").mkdir()
        result = list_directory(str(tmp_path))
        assert "a.py" in result
        assert "b.txt" in result
        assert "[dir]" in result
        assert "3 entries" in result

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.txt").write_text("text")
        result = list_directory(str(tmp_path), pattern="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_nonexistent_dir(self):
        result = list_directory("/nonexistent/dir")
        assert "not found" in result.lower()

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = list_directory(str(f))
        assert "Not a directory" in result


class TestSearchFiles:
    def test_find_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("def world():\n    pass\n")
        result = search_files(str(tmp_path), "hello")
        assert "a.py" in result
        assert "b.py" not in result

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "test.py").write_text("Hello World\n")
        result = search_files(str(tmp_path), "hello")
        assert "test.py" in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "test.py").write_text("nothing here\n")
        result = search_files(str(tmp_path), "zzzzz")
        assert "No matches" in result

    def test_max_results(self, tmp_path):
        (tmp_path / "test.py").write_text("\n".join(f"match line {i}" for i in range(100)))
        result = search_files(str(tmp_path), "match", max_results=5)
        assert "Stopped at 5" in result

    def test_nonexistent_dir(self):
        result = search_files("/nonexistent", "pattern")
        assert "not found" in result.lower()


class TestReadFileGovernance:
    def test_denied_without_approval_handler(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP-SECRET")
        call = ToolCall(id="tc_1", name="read_file", input={"path": str(secret)})

        result = ToolGroup(read_file).execute(call)

        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"
        assert "TOP-SECRET" not in result.to_model_text()

    def test_reads_file_when_handler_approves(self, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("hello from disk")
        requests: list[ApprovalRequest] = []

        def approve(request: ApprovalRequest) -> ApprovalDecision:
            requests.append(request)
            return ApprovalDecision.approve()

        call = ToolCall(id="tc_1", name="read_file", input={"path": str(notes)})
        result = ToolGroup(read_file, approval_handler=approve).execute(call)

        assert result.ok is True
        assert result.value == "hello from disk"
        assert [(r.tool_name, r.capability, r.risk_level) for r in requests] == [
            ("read_file", "filesystem", "high")
        ]


class TestBounds:
    def test_max_lines_is_clamped(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("first\nsecond\n")

        assert read_file(str(f), max_lines=-1).startswith("first\n\n[Truncated")

    def test_a_file_on_one_long_line_is_read_only_up_to_the_limit(self, tmp_path):
        f = tmp_path / "one_line.txt"
        f.write_text("x" * 2_000_000)

        result = read_file(str(f))

        assert result.startswith("x" * 100_000 + "\n\n[Truncated")
        assert len(result) < 100_100

    def test_search_results_trim_long_lines_and_clamp_max_results(self, tmp_path):
        (tmp_path / "a.txt").write_text("var a = " + "x" * 2_000_000 + "\nvar a\nvar a\n")

        result = search_files(str(tmp_path), "var a", max_results=-5)

        assert len(result) < 1_000
        assert result.endswith("[Stopped at 1 results]")

    def test_an_os_error_is_an_error_string(self):
        name = "a" * 100_000

        assert read_file(name).startswith("Cannot read")
        assert list_directory(name).startswith("Cannot list")
        assert search_files(name, "x").startswith("Cannot search")

    @pytest.mark.parametrize("pattern", ["", "/etc/*"])
    def test_an_unusable_pattern_is_an_error_string(self, tmp_path, pattern):
        assert list_directory(str(tmp_path), pattern).startswith("Invalid pattern")


def test_search_skips_binary_files(tmp_path):
    (tmp_path / "blob.dat").write_bytes(b"\xff\xfe needle \x00")
    (tmp_path / "notes.txt").write_text("a needle here\n")

    assert search_files(str(tmp_path), "needle") == "notes.txt:1: a needle here"
