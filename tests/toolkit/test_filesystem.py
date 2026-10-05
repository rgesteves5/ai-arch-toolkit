"""Tests for toolkit/tools/_filesystem.py."""

from __future__ import annotations

import errno
import itertools
from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_arch_toolkit.core import ApprovalDecision, ApprovalRequest, ToolCall, ToolGroup
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


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
        failure = _failure(lambda: read_file("/nonexistent/path/file.txt"))
        assert failure.error.type == "not_found"
        assert "/nonexistent/path/file.txt" in str(failure)

    def test_directory_path(self, tmp_path):
        failure = _failure(lambda: read_file(str(tmp_path)))
        assert failure.error.type == "validation_error"
        assert "not a regular file" in str(failure)

    def test_permission_denied_is_upstream(self, monkeypatch):
        def deny(*args, **kwargs):
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(Path, "stat", deny)

        failure = _failure(lambda: read_file("locked.txt"))
        assert failure.error.type == "upstream"
        assert "permission denied" in str(failure)


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
        failure = _failure(lambda: list_directory("/nonexistent/dir"))
        assert failure.error.type == "not_found"

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        failure = _failure(lambda: list_directory(str(f)))
        assert failure.error.type == "validation_error"
        assert "not a directory" in str(failure)

    def test_no_entries_is_a_success(self, tmp_path):
        assert list_directory(str(tmp_path), "*.py") == f"No entries matching '*.py' in {tmp_path}"


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
        failure = _failure(lambda: search_files("/nonexistent", "pattern"))
        assert failure.error.type == "not_found"

    def test_a_file_is_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        failure = _failure(lambda: search_files(str(f), "x"))
        assert failure.error.type == "validation_error"
        assert "not a directory" in str(failure)


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

    def test_a_malformed_path_is_a_validation_error(self, monkeypatch):
        def fail_stat(*args, **kwargs):
            raise OSError(errno.ENAMETOOLONG, "File name too long")

        monkeypatch.setattr(Path, "stat", fail_stat)

        for call, action in (
            (lambda: read_file("too-long"), "cannot read"),
            (lambda: list_directory("too-long"), "cannot list"),
            (lambda: search_files("too-long", "x"), "cannot search"),
        ):
            failure = _failure(call)
            assert failure.error.type == "validation_error"
            assert str(failure).startswith(action)
            assert "File name too long" in str(failure)

    def test_another_os_error_is_upstream(self, monkeypatch):
        def fail_stat(*args, **kwargs):
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(Path, "stat", fail_stat)

        failure = _failure(lambda: read_file("disk.txt"))
        assert failure.error.type == "upstream"
        assert "Input/output error" in str(failure)

    @pytest.mark.parametrize("pattern", ["", "/etc/*"])
    def test_an_unusable_pattern_is_a_validation_error(self, tmp_path, pattern):
        failure = _failure(lambda: list_directory(str(tmp_path), pattern))
        assert failure.error.type == "validation_error"
        assert str(failure).startswith("invalid pattern")


def test_search_skips_binary_files(tmp_path):
    (tmp_path / "blob.dat").write_bytes(b"\xff\xfe needle \x00")
    (tmp_path / "notes.txt").write_text("a needle here\n")

    assert search_files(str(tmp_path), "needle") == "notes.txt:1: a needle here"


class TestLinksOutOfTheFolder:
    """A link inside the folder that points out of it is not followed (G-27)."""

    @staticmethod
    def _folders(tmp_path: Path) -> tuple[Path, Path]:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("a needle in the secret\n")
        root = tmp_path / "root"
        root.mkdir()
        (root / "notes.txt").write_text("a needle here\n")
        return root, outside

    def test_search_does_not_read_a_file_linked_from_outside(self, tmp_path):
        root, outside = self._folders(tmp_path)
        (root / "link.txt").symlink_to(outside / "secret.txt")

        assert search_files(str(root), "needle") == "notes.txt:1: a needle here"

    def test_search_does_not_walk_a_folder_linked_from_outside(self, tmp_path):
        root, outside = self._folders(tmp_path)
        (root / "linked").symlink_to(outside, target_is_directory=True)

        assert search_files(str(root), "needle") == "notes.txt:1: a needle here"

    def test_search_reads_a_link_that_stays_inside(self, tmp_path):
        root, _ = self._folders(tmp_path)
        (root / "alias.txt").symlink_to(root / "notes.txt")

        assert sorted(search_files(str(root), "needle").splitlines()) == [
            "alias.txt:1: a needle here",
            "notes.txt:1: a needle here",
        ]

    def test_search_from_a_linked_folder_reads_its_files(self, tmp_path):
        root, _ = self._folders(tmp_path)
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)

        assert search_files(str(alias), "needle") == "notes.txt:1: a needle here"

    def test_list_does_not_go_through_a_link_out_of_the_folder(self, tmp_path):
        root, outside = self._folders(tmp_path)
        (root / "linked").symlink_to(outside, target_is_directory=True)

        assert list_directory(str(root), "linked/*") == f"No entries matching 'linked/*' in {root}"

    def test_list_does_not_climb_out_with_dot_dot(self, tmp_path):
        root, _ = self._folders(tmp_path)

        assert list_directory(str(root), "../*") == f"No entries matching '../*' in {root}"

    def test_list_shows_a_link_inside_the_folder_by_its_name(self, tmp_path):
        root, outside = self._folders(tmp_path)
        (root / "linked").symlink_to(outside, target_is_directory=True)

        assert "[dir]  linked/" in list_directory(str(root))

    def test_list_goes_into_a_real_subfolder(self, tmp_path):
        root, _ = self._folders(tmp_path)
        (root / "sub").mkdir()
        (root / "sub" / "inner.txt").write_text("x")

        assert "inner.txt" in list_directory(str(root), "sub/*")

    def test_list_stops_at_its_cap_when_every_match_is_outside(self, tmp_path, monkeypatch):
        walked = 0

        def endless(self: Path, pattern: str) -> Iterator[Path]:
            nonlocal walked
            for index in itertools.count():
                walked += 1
                assert walked <= 2_000, "the walk went past the cap"
                yield tmp_path.parent / f"outside-{index}"

        monkeypatch.setattr(Path, "glob", endless)

        assert list_directory(str(tmp_path), "../*") == f"No entries matching '../*' in {tmp_path}"
        assert walked == 1_001
