"""Tests for toolkit/tools/_filesystem.py."""

from __future__ import annotations

import errno
import io
import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ai_arch_toolkit.core import ApprovalDecision, ApprovalRequest, ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _filesystem
from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _text(result: ToolResult) -> str:
    assert result.ok and isinstance(result.value, str), result
    return result.value


def _window(result: ToolResult) -> dict[str, Any]:
    return result.metadata["window"]


def _hits(result: ToolResult) -> list[str]:
    """The matching lines of a search, without its heading and footer."""
    lines = _text(result).splitlines()[1:]
    return [line for line in lines if not line.startswith("[results ")]


class _Recording:
    """A file handle that records the size of every read."""

    def __init__(self, handle: io.TextIOWrapper, asked: list[int]) -> None:
        self._handle, self._asked = handle, asked

    def __enter__(self) -> _Recording:
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()

    def read(self, size: int = -1) -> str:
        self._asked.append(size)
        return self._handle.read(size)


def _approve(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.approve()


def _refused(tool, **arguments: object) -> str:
    """The validation_error the executor gives for ``arguments`` (the schema's bounds)."""
    call = ToolCall(id="call", name=tool.__name__, input=arguments)
    result = ToolGroup(tool, approval_handler=_approve).execute(call)
    assert result.error is not None and result.error.type == "validation_error", result
    return result.error.message


class TestReadFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")

        assert _text(read_file(str(f))) == "line1\nline2\nline3\n"

    def test_a_window_holds_max_lines_and_names_the_next_offset(self, tmp_path):
        f = tmp_path / "big.txt"
        text = "".join(f"line {i}\n" for i in range(500))
        f.write_text(text)
        first_ten = "".join(f"line {i}\n" for i in range(10))

        result = read_file(str(f), max_lines=10)

        assert _text(result) == (
            f"{first_ten}[chars 0-{len(first_ten)} of {len(text)} | next: offset={len(first_ten)}]"
        )
        assert _window(result)["next_call"] == {"offset": len(first_ten)}

    def test_following_the_footers_rebuilds_the_whole_file(self, tmp_path):
        # Accents, Windows line ends, blank lines, a line longer than a window, no final break.
        text = (
            "".join(f"línea {i} — ação\r\n" for i in range(3000))
            + "\n\n"
            + "x" * 250_000
            + "\nend without a break"
        )
        f = tmp_path / "mixed.txt"
        f.write_bytes(text.encode())

        parts: list[str] = []
        call: dict[str, Any] | None = {"offset": 0}
        while call is not None:
            result = read_file(str(f), max_lines=700, **call)
            window = _window(result)
            assert window["total"] == len(text)
            parts.append(_text(result)[: window["last"] - window["first"]])
            call = window["next_call"]

        assert "".join(parts) == text
        assert len(parts) > 5

    def test_the_total_counts_the_whole_file_past_the_old_read_limit(self, tmp_path):
        f = tmp_path / "long.txt"
        text = "".join(f"{i:07d}\n" for i in range(400_000))  # 3.2 million characters
        f.write_text(text)

        window = _window(read_file(str(f), max_lines=5))

        assert window["total"] == len(text)

    def test_the_total_waits_until_the_rest_is_within_the_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_filesystem, "_COUNT_CHARS", 1_000)
        f = tmp_path / "long.txt"
        # 150,000 characters: more than a window (100,000) and the count (1,000) together.
        f.write_text("".join(f"{i:05d}\n" for i in range(25_000)))

        early = read_file(str(f), max_lines=10)
        late = read_file(str(f), offset=140_000, max_lines=10)

        assert _text(early).endswith("\n[chars 0-60 | next: offset=60]")
        assert _window(late)["total"] == 150_000

    def test_a_line_longer_than_a_window_is_cut_where_the_limit_falls(self, tmp_path):
        f = tmp_path / "one_line.txt"
        f.write_text("x" * 2_000_000)

        result = read_file(str(f))

        assert (
            _text(result) == "x" * 100_000 + "\n[chars 0-100000 of 2000000 | next: offset=100000]"
        )

    def test_an_offset_past_the_end_shows_nothing_and_says_end(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("abc\n")

        assert _text(read_file(str(f), offset=50)) == "[chars 4-4 of 4 | end]"

    def test_the_file_is_read_a_chunk_at_a_time(self, tmp_path, monkeypatch):
        f = tmp_path / "long.txt"
        f.write_text("y\n" * 1_500_000)
        asked: list[int] = []
        opened = Path.open

        def recording(self: Path, *args: Any, **kwargs: Any) -> _Recording:
            return _Recording(opened(self, *args, **kwargs), asked)

        monkeypatch.setattr(Path, "open", recording)

        read_file(str(f), offset=1_200_000)

        assert asked and all(0 < size <= _filesystem._CHUNK for size in asked)

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
        result = _text(list_directory(str(tmp_path)))
        assert "a.py" in result
        assert "b.txt" in result
        assert "[dir]" in result
        assert "3 entries" in result

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.txt").write_text("text")
        result = _text(list_directory(str(tmp_path), pattern="*.py"))
        assert "a.py" in result
        assert "b.txt" not in result

    def test_a_listing_longer_than_a_page_is_paged_with_its_total(self, tmp_path):
        names = [f"{number:04d}.txt" for number in range(1205)]
        for name in names:
            (tmp_path / name).write_text("x")

        first = list_directory(str(tmp_path))
        second = list_directory(str(tmp_path), offset=1000)

        assert _text(first).startswith(f"{tmp_path} (1205 entries):\n")
        assert _text(first).endswith("\n[results 1-1000 of 1205 | next: offset=1000]")
        assert _text(second).endswith("\n[results 1001-1205 of 1205 | end]")
        listed = [
            line.split()[-1]
            for result in (first, second)
            for line in _text(result).splitlines()[1:-1]
        ]
        assert listed == names

    def test_a_pattern_that_matches_more_than_the_walk_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_filesystem, "_MAX_WALK", 10)
        for number in range(11):
            (tmp_path / f"{number}.txt").write_text("x")

        failure = _failure(lambda: list_directory(str(tmp_path), "*.txt"))

        assert failure.error.type == "validation_error"
        assert "matches more than 10 entries" in str(failure) and "narrow" in str(failure)

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
        result = list_directory(str(tmp_path), "*.py")
        assert _text(result) == f"No entries matching '*.py' in {tmp_path}"


class TestSearchFiles:
    def test_find_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("def world():\n    pass\n")
        result = _text(search_files(str(tmp_path), "hello"))
        assert "a.py" in result
        assert "b.py" not in result

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "test.py").write_text("Hello World\n")
        result = _text(search_files(str(tmp_path), "hello"))
        assert "test.py" in result

    def test_no_matches_say_the_pattern(self, tmp_path):
        (tmp_path / "test.py").write_text("nothing here\n")
        result = search_files(str(tmp_path), "zzzzz")
        assert _text(result) == f"No matches for 'zzzzz' in {tmp_path}"

    def test_an_empty_pattern_is_a_validation_error(self, tmp_path):
        failure = _failure(lambda: search_files(str(tmp_path), ""))
        assert failure.error.type == "validation_error"

    def test_each_hit_gives_the_offset_read_file_reads_from(self, tmp_path):
        source = tmp_path / "a.py"
        source.write_text("import os\n\n    def hello():\n        pass\n")

        hits = _hits(search_files(str(tmp_path), "hello"))

        assert hits == ["a.py:3:15: def hello():"]
        assert _text(read_file(str(source), offset=15)).startswith("def hello():\n")

    def test_results_are_paged_with_the_next_offset(self, tmp_path):
        (tmp_path / "hay.txt").write_text("".join(f"needle {i}\n" for i in range(10)))

        first = search_files(str(tmp_path), "needle", max_results=3)
        last = search_files(str(tmp_path), "needle", max_results=3, offset=9)

        assert _hits(first) == [f"hay.txt:{i + 1}:{9 * i}: needle {i}" for i in range(3)]
        assert _text(first).endswith("\n[results 1-3 | next: offset=3]")
        assert _hits(last) == ["hay.txt:10:81: needle 9"]
        assert _text(last).endswith("\n[results 10-10 of 10 | end]")

    def test_files_come_in_name_order_folder_by_folder(self, tmp_path):
        for name in ("b.txt", "a.txt", "sub/c.txt", "A/d.txt"):
            (tmp_path / name).parent.mkdir(exist_ok=True)
            (tmp_path / name).write_text("needle\n")

        hits = _hits(search_files(str(tmp_path), "needle"))

        assert [hit.split(":")[0] for hit in hits] == ["a.txt", "b.txt", "A/d.txt", "sub/c.txt"]

    def test_the_whole_file_is_searched(self, tmp_path):
        filler = "".join(f"filler line {i}\n" for i in range(150_000))  # 2.5 million characters
        (tmp_path / "log.txt").write_text(filler + "the needle at the end\n")

        hits = _hits(search_files(str(tmp_path), "needle"))

        assert hits == [f"log.txt:150001:{len(filler)}: the needle at the end"]

    def test_a_long_line_shows_the_part_around_the_match_and_says_so(self, tmp_path):
        line = "x" * 5000 + "needle" + "y" * 5000
        source = tmp_path / "wide.txt"
        source.write_text("short\n" + line + "\n")

        (hit,) = _hits(search_files(str(tmp_path), "needle"))

        place, _, rest = hit.partition(": ")
        offset = int(place.split(":")[2])
        shown, _, note = rest.partition(" [")
        assert place.startswith("wide.txt:2:") and "needle" in shown and len(shown) <= 300
        assert note == f"part of a {len(line)}-char line; read_file(offset={offset}) reads on]"
        assert _text(read_file(str(source), offset=offset)).startswith(shown)

    def test_a_match_split_between_two_pieces_of_a_line_is_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_filesystem, "_PIECE", 8)
        (tmp_path / "a.txt").write_text("abcdenee" + "dle and more text after it\n")

        (hit,) = _hits(search_files(str(tmp_path), "needle"))

        assert hit.startswith("a.txt:1:")

    def test_a_nonexistent_dir(self):
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
    def test_the_limits_are_in_the_schema_and_the_executor_refuses_past_them(self, tmp_path):
        assert "from 1 to 10000" in _refused(read_file, path="x", max_lines=0)
        assert "offset" in _refused(read_file, path="x", offset=-1)
        assert "max_results" in _refused(search_files, directory=".", pattern="x", max_results=0)
        assert "offset" in _refused(list_directory, path=".", offset=-1)

    def test_long_lines_in_search_results_stay_short(self, tmp_path):
        (tmp_path / "a.txt").write_text("var a = " + "x" * 2_000_000 + "\nvar a\nvar a\n")

        result = search_files(str(tmp_path), "var a", max_results=1)

        assert len(_text(result)) < 1_000
        assert _text(result).endswith("[results 1-1 | next: offset=1]")

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

    assert _hits(search_files(str(tmp_path), "needle")) == ["notes.txt:1:0: a needle here"]


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

        assert _hits(search_files(str(root), "needle")) == ["notes.txt:1:0: a needle here"]

    def test_search_does_not_walk_a_folder_linked_from_outside(self, tmp_path):
        root, outside = self._folders(tmp_path)
        (root / "linked").symlink_to(outside, target_is_directory=True)

        assert _hits(search_files(str(root), "needle")) == ["notes.txt:1:0: a needle here"]

    def test_search_reads_a_link_that_stays_inside(self, tmp_path):
        root, _ = self._folders(tmp_path)
        (root / "alias.txt").symlink_to(root / "notes.txt")

        assert _hits(search_files(str(root), "needle")) == [
            "alias.txt:1:0: a needle here",
            "notes.txt:1:0: a needle here",
        ]

    def test_search_from_a_linked_folder_reads_its_files(self, tmp_path):
        root, _ = self._folders(tmp_path)
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)

        assert _hits(search_files(str(alias), "needle")) == ["notes.txt:1:0: a needle here"]

    def test_list_does_not_go_through_a_link_out_of_the_folder(self, tmp_path):
        root, outside = self._folders(tmp_path)
        (root / "linked").symlink_to(outside, target_is_directory=True)

        result = list_directory(str(root), "linked/*")
        assert _text(result) == f"No entries matching 'linked/*' in {root}"

    def test_list_does_not_climb_out_with_dot_dot(self, tmp_path):
        root, _ = self._folders(tmp_path)

        assert _text(list_directory(str(root), "../*")) == f"No entries matching '../*' in {root}"

    def test_list_shows_a_link_inside_the_folder_by_its_name(self, tmp_path):
        root, outside = self._folders(tmp_path)
        (root / "linked").symlink_to(outside, target_is_directory=True)

        assert "[dir]  linked/" in _text(list_directory(str(root)))

    def test_list_goes_into_a_real_subfolder(self, tmp_path):
        root, _ = self._folders(tmp_path)
        (root / "sub").mkdir()
        (root / "sub" / "inner.txt").write_text("x")

        assert "inner.txt" in _text(list_directory(str(root), "sub/*"))

    def test_list_stops_its_walk_at_the_cap_when_every_match_is_outside(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(_filesystem, "_MAX_WALK", 1_000)
        walked = 0

        def endless(self: Path, pattern: str) -> Iterator[Path]:
            nonlocal walked
            for index in itertools.count():
                walked += 1
                assert walked <= 2_000, "the walk went past the cap"
                yield tmp_path.parent / f"outside-{index}"

        monkeypatch.setattr(Path, "glob", endless)

        failure = _failure(lambda: list_directory(str(tmp_path), "../*"))

        # The walk stops where it did; past the cap the answer says so, instead of "no entries".
        assert walked == 1_001
        assert failure.error.type == "validation_error"
        assert "narrow" in str(failure)
