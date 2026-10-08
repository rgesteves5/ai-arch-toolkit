"""The tools of ``filesystem_tools(policy)``: typed writes and bound reads (C07d, C07e; D64).

Every test works in ``tmp_path``: a root the policy allows and a folder next to it, ``outside``,
that it does not, holding ``OUTSIDE-SECRET``. Whatever a test does, nothing may land outside.
"""

from __future__ import annotations

import asyncio
import errno
import os
import stat
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ai_arch_toolkit.core import (
    ApprovalDecision,
    ApprovalRequest,
    DryRunGate,
    MeterScope,
    ToolCall,
    ToolFailure,
    ToolGroup,
    ToolResult,
    execute_tool,
)
from ai_arch_toolkit.toolkit.tools import _filesystem_write, dangerous
from ai_arch_toolkit.toolkit.tools.dangerous import (
    FilesystemPolicy,
    PathScopeGate,
    filesystem_tools,
)
from tests.toolkit.test_tool_invariants import _plans

type Tool = Callable[..., Any]

_WRITES = ("write_file", "append_file", "make_directory", "move_path")
_READS = ("read_file", "list_directory", "search_files")


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path
    outside: Path
    policy: FilesystemPolicy
    tools: dict[str, Tool]

    def __getitem__(self, name: str) -> Tool:
        return self.tools[name]

    def outside_now(self) -> set[str]:
        """What ``outside`` holds, at any depth."""
        return {str(path.relative_to(self.outside)) for path in self.outside.rglob("*")}


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    base = Path(os.path.realpath(tmp_path))
    root, outside = base / "root", base / "outside"
    (root / "sub").mkdir(parents=True)
    outside.mkdir()
    (root / "notes.txt").write_text("a needle here\n")
    (outside / "secret.txt").write_text("OUTSIDE-SECRET\n")
    policy = FilesystemPolicy(read_roots=(root,), write_roots=(root,), cwd=root)
    return Layout(root, outside, policy, {fn.__name__: fn for fn in filesystem_tools(policy)})


def _failure(call: Callable[[], object]) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _text(result: object) -> str:
    if isinstance(result, ToolResult):
        assert result.ok and isinstance(result.value, str), result
        return result.value
    assert isinstance(result, str), result
    return result


def _approve(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.approve()


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=name, name=name, input=arguments)


def _open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def _leftovers(folder: Path) -> list[str]:
    """Temporary files a write left behind."""
    return [path.name for path in folder.rglob(".ai-arch-*")]


# --- The factory -------------------------------------------------------------------------------


class TestFactory:
    def test_a_policy_with_both_roots_gives_the_reads_and_the_writes(self, layout):
        assert set(layout.tools) == {*_READS, *_WRITES}

    def test_reads_only_without_write_roots_and_writes_only_without_read_roots(self, layout):
        reads = FilesystemPolicy(read_roots=(layout.root,))
        writes = FilesystemPolicy(write_roots=(layout.root,))

        assert {fn.__name__ for fn in filesystem_tools(reads)} == set(_READS)
        assert {fn.__name__ for fn in filesystem_tools(writes)} == set(_WRITES)

    def test_a_policy_that_reads_and_writes_nothing_gives_no_tools(self, layout):
        with pytest.raises(ValueError, match="no read_roots and no write_roots"):
            filesystem_tools(FilesystemPolicy(delete_roots=(layout.root,)))

    def test_on_windows_the_writes_are_refused_when_built_and_the_reads_stay(
        self, layout, monkeypatch
    ):
        monkeypatch.setattr(sys, "platform", "win32")

        with pytest.raises(NotImplementedError, match="POSIX"):
            filesystem_tools(layout.policy)
        reads = filesystem_tools(FilesystemPolicy(read_roots=(layout.root,)))
        assert {fn.__name__ for fn in reads} == set(_READS)

    @pytest.mark.parametrize("name", _WRITES)
    def test_each_write_is_dangerous_and_approved_call_by_call(self, layout, name):
        policy = layout[name].__tool_definition__.policy

        assert (policy.capability, policy.risk_level, policy.requires_approval) == (
            "filesystem",
            "high",
            True,
        )
        assert policy.approval_reason

    @pytest.mark.parametrize("name", _READS)
    def test_each_bound_read_keeps_the_name_schema_and_governance_of_its_tool(self, layout, name):
        bound = layout[name].__tool_definition__
        unbound = getattr(dangerous, name).__tool_definition__

        assert bound.schema == unbound.schema
        assert bound.policy == unbound.policy
        assert bound.fn is layout[name]

    def test_without_an_approver_every_write_is_denied(self, layout):
        result = ToolGroup(layout["write_file"]).execute(
            _call("write_file", path="new.txt", content="x")
        )

        assert result.error is not None and result.error.type == "approval_denied"
        assert not (layout.root / "new.txt").exists()


# --- write_file --------------------------------------------------------------------------------


class TestWriteFile:
    def test_the_exact_bytes_are_written_and_the_answer_names_the_canonical_path(self, layout):
        content = "line one\r\nlinha dois: ação\n\tend without a newline"

        answer = _text(layout["write_file"](path="sub/../new.md", content=content))

        written = layout.root / "new.md"
        assert written.read_bytes() == content.encode("utf-8")
        assert answer == f"Created {written} ({len(content.encode())} bytes)"

    def test_without_overwrite_an_existing_file_stays_as_it_was(self, layout):
        failure = _failure(lambda: layout["write_file"](path="notes.txt", content="other"))

        assert failure.error.type == "validation_error"
        assert "overwrite=true" in str(failure)
        assert (layout.root / "notes.txt").read_text() == "a needle here\n"
        assert _leftovers(layout.root) == []

    def test_overwrite_replaces_the_file_and_keeps_its_mode(self, layout):
        target = layout.root / "notes.txt"
        target.chmod(0o640)

        answer = _text(layout["write_file"](path="notes.txt", content="new", overwrite=True))

        assert target.read_text() == "new"
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        assert answer == f"Replaced {target} (14 → 3 bytes)"

    def test_a_new_file_takes_0666_under_the_umask(self, layout):
        old = os.umask(0o027)
        try:
            layout["write_file"](path="masked.txt", content="x")
        finally:
            os.umask(old)

        assert stat.S_IMODE((layout.root / "masked.txt").stat().st_mode) == 0o640

    def test_a_link_at_the_leaf_is_refused_and_its_target_stays(self, layout):
        link = layout.root / "link.txt"
        link.symlink_to(layout.outside / "secret.txt")

        failure = _failure(
            lambda: layout["write_file"](path="link.txt", content="pwned", overwrite=True)
        )

        assert failure.error.type == "permission_denied"
        assert (layout.outside / "secret.txt").read_text() == "OUTSIDE-SECRET\n"
        assert link.is_symlink()

    def test_a_missing_folder_is_not_found_unless_create_parents(self, layout):
        failure = _failure(lambda: layout["write_file"](path="a/b/c.txt", content="x"))

        assert failure.error.type == "not_found"
        assert "create_parents=true" in str(failure)
        layout["write_file"](path="a/b/c.txt", content="x", create_parents=True)
        assert (layout.root / "a" / "b" / "c.txt").read_text() == "x"

    @pytest.mark.parametrize("content", [["a"], 5, None, b"bytes"])
    def test_content_that_is_not_text_is_a_validation_error(self, layout, content):
        failure = _failure(lambda: layout["write_file"](path="x.txt", content=content))

        assert failure.error.type == "validation_error"
        assert "content must be text" in str(failure)
        assert not (layout.root / "x.txt").exists()

    def test_content_that_is_not_utf8_is_a_validation_error(self, layout):
        failure = _failure(lambda: layout["write_file"](path="x.txt", content="lone \ud800"))

        assert failure.error.type == "validation_error"
        assert "UTF-8" in str(failure)

    def test_content_past_the_policys_limit_is_refused(self, layout):
        policy = FilesystemPolicy(write_roots=(layout.root,), cwd=layout.root, max_write_bytes=8)
        write = {fn.__name__: fn for fn in filesystem_tools(policy)}["write_file"]

        failure = _failure(lambda: write(path="big.txt", content="ç" * 5))
        # Past the limit in characters, it is refused before it is encoded.
        at_once = _failure(lambda: write(path="big.txt", content="x" * 9))

        assert failure.error.type == "validation_error"
        assert "is 10 bytes" in str(failure) and "at most 8" in str(failure)
        assert "append_file" in str(failure)
        assert "is at least 9 bytes" in str(at_once)
        assert not (layout.root / "big.txt").exists()

    def test_the_root_itself_is_never_replaced(self, layout):
        failure = _failure(lambda: layout["write_file"](path=".", content="x", overwrite=True))

        assert failure.error.type == "permission_denied"

    def test_a_folder_at_the_path_is_a_validation_error(self, layout):
        failure = _failure(lambda: layout["write_file"](path="sub", content="x", overwrite=True))

        assert failure.error.type == "validation_error"
        assert (layout.root / "sub").is_dir()

    def test_a_failed_write_leaves_no_temporary_file_and_no_target(self, layout, monkeypatch):
        def failing_fsync(descriptor: int) -> None:
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(os, "fsync", failing_fsync)

        failure = _failure(lambda: layout["write_file"](path="new.txt", content="x"))

        assert failure.error.type == "upstream"
        assert "Input/output error" in str(failure)
        assert not (layout.root / "new.txt").exists()
        assert _leftovers(layout.root) == []

    @pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP])
    def test_a_filesystem_without_hard_links_says_so_and_writes_nothing(
        self, layout, monkeypatch, code
    ):
        def no_hard_links(*args: Any, **kwargs: Any) -> None:
            raise OSError(code, os.strerror(code))

        monkeypatch.setattr(os, "link", no_hard_links)

        failure = _failure(lambda: layout["write_file"](path="new.txt", content="x"))
        moved = _failure(lambda: layout["move_path"](source="notes.txt", destination="n.txt"))

        for refused in (failure, moved):
            assert refused.error.type == "upstream"
            assert "no hard links" in str(refused) and "overwrite=true" in str(refused)
        assert not (layout.root / "new.txt").exists() and (layout.root / "notes.txt").exists()
        assert _leftovers(layout.root) == []

    def test_a_folder_that_cannot_be_synced_is_a_note_not_a_failure(self, layout, monkeypatch):
        # Some filesystems refuse to sync a folder (EINVAL): the name is in place by then (L2).
        fsync = os.fsync

        def files_only(descriptor: int) -> None:
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError(errno.EINVAL, "Invalid argument")
            fsync(descriptor)

        monkeypatch.setattr(os, "fsync", files_only)
        root = layout.root

        answers = [
            _text(layout["write_file"](path="new.txt", content="x")),
            _text(layout["write_file"](path="new.txt", content="yz", overwrite=True)),
            _text(layout["make_directory"](path="made")),
            _text(layout["move_path"](source="new.txt", destination="made/new.txt")),
        ]

        firsts = [
            f"Created {root / 'new.txt'} (1 bytes)",
            f"Replaced {root / 'new.txt'} (1 → 2 bytes)",
            f"Created folder {root / 'made'}",
            f"Moved {root / 'new.txt'} to {root / 'made' / 'new.txt'}",
        ]
        for answer, first in zip(answers, firsts, strict=True):
            assert answer.startswith(first + "; "), answer
            assert "could not be synced to disk (Invalid argument)" in answer
        assert (root / "made" / "new.txt").read_text() == "yz"
        assert _leftovers(root) == []

    def test_no_descriptor_stays_open_after_a_write_or_a_failure(self, layout):
        before = _open_descriptors()

        layout["write_file"](path="a.txt", content="x")
        layout["write_file"](path="a.txt", content="y", overwrite=True)
        _failure(lambda: layout["write_file"](path="a.txt", content="z"))
        _failure(lambda: layout["write_file"](path="missing/a.txt", content="z"))
        (layout.root / "door").symlink_to(layout.outside, target_is_directory=True)
        _failure(lambda: layout["write_file"](path="door/a.txt", content="z"))

        assert _open_descriptors() == before

    async def test_two_parallel_writes_to_one_new_path_give_one_created_and_one_error(
        self, layout
    ):
        group = ToolGroup(*layout.tools.values(), approval_handler=_approve)
        calls = [_call("write_file", path="race.txt", content=f"writer {n}") for n in (1, 2)]

        results = await asyncio.gather(*(group.async_execute(call) for call in calls))

        created = [result for result in results if result.ok]
        refused = [result for result in results if not result.ok]
        assert len(created) == 1 and len(refused) == 1
        assert refused[0].error is not None and refused[0].error.type == "validation_error"
        assert "already exists" in refused[0].error.message
        assert (layout.root / "race.txt").read_text() in {"writer 1", "writer 2"}
        assert _leftovers(layout.root) == []

    def test_a_name_another_process_takes_before_the_link_is_never_replaced(
        self, layout, monkeypatch
    ):
        # The tools of one process take turns; another process may still take the name between
        # the write's look and its link, and os.link then refuses it (C07.5).
        link = os.link

        def taken_first(*args: Any, **kwargs: Any) -> None:
            (layout.root / "race.txt").write_text("the other process")
            link(*args, **kwargs)

        monkeypatch.setattr(os, "link", taken_first)

        failure = _failure(lambda: layout["write_file"](path="race.txt", content="mine"))

        assert failure.error.type == "validation_error"
        assert "already exists" in str(failure)
        assert (layout.root / "race.txt").read_text() == "the other process"
        assert _leftovers(layout.root) == []


# --- append_file -------------------------------------------------------------------------------


class TestAppendFile:
    def test_the_exact_bytes_are_added_at_the_end(self, layout):
        answer = _text(layout["append_file"](path="notes.txt", content="more\r\n"))

        assert (layout.root / "notes.txt").read_bytes() == b"a needle here\nmore\r\n"
        assert answer == f"Appended 6 bytes to {layout.root / 'notes.txt'} (now 20 bytes)"

    def test_a_missing_file_is_not_found_and_points_to_write_file(self, layout):
        failure = _failure(lambda: layout["append_file"](path="new.txt", content="x"))

        assert failure.error.type == "not_found"
        assert "write_file" in str(failure)

    def test_a_file_with_a_hard_link_is_refused_and_both_stay(self, layout):
        os.link(layout.outside / "secret.txt", layout.root / "hard.txt")

        failure = _failure(lambda: layout["append_file"](path="hard.txt", content="pwned"))

        assert failure.error.type == "permission_denied"
        assert "hard link" in str(failure)
        assert (layout.outside / "secret.txt").read_text() == "OUTSIDE-SECRET\n"

    def test_a_link_is_refused(self, layout):
        (layout.root / "link.txt").symlink_to(layout.outside / "secret.txt")

        failure = _failure(lambda: layout["append_file"](path="link.txt", content="pwned"))

        assert failure.error.type == "permission_denied"
        assert (layout.outside / "secret.txt").read_text() == "OUTSIDE-SECRET\n"

    def test_a_folder_is_a_validation_error(self, layout):
        failure = _failure(lambda: layout["append_file"](path="sub", content="x"))

        assert failure.error.type == "validation_error"

    @pytest.mark.timeout(10)
    def test_a_pipe_never_holds_the_call(self, layout):
        os.mkfifo(layout.root / "pipe")

        failure = _failure(lambda: layout["append_file"](path="pipe", content="x"))

        assert failure.error.type == "validation_error"
        assert "not a regular file" in str(failure)


# --- make_directory ----------------------------------------------------------------------------


class TestMakeDirectory:
    def test_a_folder_is_made(self, layout):
        answer = _text(layout["make_directory"](path="made"))

        assert (layout.root / "made").is_dir()
        assert answer == f"Created folder {layout.root / 'made'}"

    def test_missing_parents_need_parents(self, layout):
        failure = _failure(lambda: layout["make_directory"](path="a/b/c"))

        assert failure.error.type == "not_found"
        assert "parents=true" in str(failure)
        layout["make_directory"](path="a/b/c", parents=True)
        assert (layout.root / "a" / "b" / "c").is_dir()

    def test_a_folder_that_exists_is_a_success_that_says_so(self, layout):
        assert _text(layout["make_directory"](path="sub")) == (
            f"{layout.root / 'sub'} already exists"
        )

    def test_a_file_in_the_way_is_a_validation_error(self, layout):
        failure = _failure(lambda: layout["make_directory"](path="notes.txt"))

        assert failure.error.type == "validation_error"
        assert (layout.root / "notes.txt").is_file()

    def test_the_root_is_never_made_and_nothing_outside_either(self, layout):
        assert _failure(lambda: layout["make_directory"](path=".")).error.type == (
            "permission_denied"
        )
        failure = _failure(lambda: layout["make_directory"](path="../outside/made"))
        assert failure.error.type == "permission_denied"
        assert layout.outside_now() == {"secret.txt"}

    def test_a_new_folder_takes_0777_under_the_umask(self, layout):
        old = os.umask(0o027)
        try:
            layout["make_directory"](path="a/b", parents=True)
        finally:
            os.umask(old)

        for folder in (layout.root / "a", layout.root / "a" / "b"):
            assert stat.S_IMODE(folder.stat().st_mode) == 0o750


# --- move_path ---------------------------------------------------------------------------------


class TestMovePath:
    def test_a_file_is_moved(self, layout):
        answer = _text(layout["move_path"](source="notes.txt", destination="sub/moved.txt"))

        assert not (layout.root / "notes.txt").exists()
        assert (layout.root / "sub" / "moved.txt").read_text() == "a needle here\n"
        assert answer == (
            f"Moved {layout.root / 'notes.txt'} to {layout.root / 'sub' / 'moved.txt'}"
        )

    def test_a_folder_is_moved(self, layout):
        (layout.root / "sub" / "inner.txt").write_text("x")

        layout["move_path"](source="sub", destination="renamed")

        assert (layout.root / "renamed" / "inner.txt").read_text() == "x"
        assert not (layout.root / "sub").exists()

    @pytest.mark.parametrize("folder", [False, True])
    def test_without_overwrite_an_existing_destination_stays(self, layout, folder):
        source = layout.root / ("sub" if folder else "notes.txt")
        (layout.root / "taken").mkdir()
        (layout.root / "taken" / "keep.txt").write_text("kept")
        if not folder:
            (layout.root / "taken.txt").write_text("kept")
        destination = "taken" if folder else "taken.txt"

        failure = _failure(
            lambda: layout["move_path"](source=source.name, destination=destination)
        )

        assert failure.error.type == "validation_error"
        assert "overwrite=true" in str(failure)
        assert source.exists()

    def test_overwrite_replaces_a_file(self, layout):
        (layout.root / "old.txt").write_text("old")

        layout["move_path"](source="notes.txt", destination="old.txt", overwrite=True)

        assert (layout.root / "old.txt").read_text() == "a needle here\n"
        assert not (layout.root / "notes.txt").exists()

    def test_a_missing_source_is_not_found(self, layout):
        failure = _failure(lambda: layout["move_path"](source="gone.txt", destination="x.txt"))

        assert failure.error.type == "not_found"

    def test_a_root_and_a_folder_that_holds_one_are_never_moved(self, tmp_path):
        base = Path(os.path.realpath(tmp_path))
        (base / "docs" / "kept").mkdir(parents=True)
        policy = FilesystemPolicy(
            read_roots=(base / "docs" / "kept",), write_roots=(base,), cwd=base
        )
        move = {fn.__name__: fn for fn in filesystem_tools(policy)}["move_path"]

        for source in ("docs", "docs/kept"):
            failure = _failure(lambda source=source: move(source=source, destination="moved"))
            assert failure.error.type == "permission_denied"
        assert (base / "docs" / "kept").is_dir()

    def test_a_folder_never_moves_into_itself(self, layout):
        failure = _failure(lambda: layout["move_path"](source="sub", destination="sub/inner"))

        assert failure.error.type == "validation_error"
        assert (layout.root / "sub").is_dir()

    def test_a_link_is_never_moved(self, layout):
        (layout.root / "link").symlink_to(layout.outside, target_is_directory=True)

        failure = _failure(lambda: layout["move_path"](source="link", destination="moved"))

        assert failure.error.type == "permission_denied"
        assert (layout.root / "link").is_symlink()

    def test_nothing_moves_out_of_the_root(self, layout):
        failure = _failure(
            lambda: layout["move_path"](source="notes.txt", destination="../outside/notes.txt")
        )

        assert failure.error.type == "permission_denied"
        assert (layout.root / "notes.txt").exists()
        assert layout.outside_now() == {"secret.txt"}

    @pytest.mark.parametrize("folder", [False, True])
    def test_another_volume_is_refused_and_nothing_is_copied(self, layout, monkeypatch, folder):
        def other_volume(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.EXDEV, "Cross-device link")

        for name in ("link", "rename", "replace"):
            monkeypatch.setattr(os, name, other_volume)
        source = "sub" if folder else "notes.txt"

        failure = _failure(lambda: layout["move_path"](source=source, destination="moved"))

        assert failure.error.type == "validation_error"
        assert "another volume" in str(failure) and "never copies" in str(failure)
        assert (layout.root / source).exists() and not (layout.root / "moved").exists()

    def test_a_source_replaced_between_link_and_unlink_is_left_in_place(self, layout, monkeypatch):
        # Another process gives old.txt new content right after the move linked it to old.bak:
        # the move must not then remove the new file (M2).
        root = layout.root
        (root / "old.txt").write_text("old")
        (root / "new.txt").write_text("new")
        link = os.link

        def link_then_replace(*args: Any, **kwargs: Any) -> None:
            link(*args, **kwargs)
            os.replace(root / "new.txt", root / "old.txt")

        monkeypatch.setattr(os, "link", link_then_replace)

        answer = _text(layout["move_path"](source="old.txt", destination="old.bak"))

        assert (root / "old.bak").read_text() == "old"
        assert (root / "old.txt").read_text() == "new"
        assert answer == (
            f"Moved {root / 'old.txt'} to {root / 'old.bak'}; {root / 'old.txt'} no longer "
            "names the file that moved, and is left as it is"
        )

    def test_two_moves_of_one_process_never_interleave(self, layout, monkeypatch):
        # The race of the review: a move by link and unlink, and a replace of its source landing
        # between the two. The second call waits for the first (M2).
        root = layout.root
        (root / "old.txt").write_text("old")
        (root / "new.txt").write_text("new")
        link = os.link
        results: list[object] = []
        within: list[bool] = []

        def replace_old() -> None:
            results.append(
                layout["move_path"](source="new.txt", destination="old.txt", overwrite=True)
            )

        second = threading.Thread(target=replace_old)

        def link_then_race(*args: Any, **kwargs: Any) -> None:
            link(*args, **kwargs)
            second.start()
            second.join(0.5)
            within.append(not second.is_alive())

        monkeypatch.setattr(os, "link", link_then_race)

        layout["move_path"](source="old.txt", destination="old.bak")
        second.join(5)

        assert within == [False]  # the second move waited for the first to end
        assert results == [f"Moved {root / 'new.txt'} to {root / 'old.txt'}"]
        assert (root / "old.bak").read_text() == "old"
        assert (root / "old.txt").read_text() == "new"

    def test_a_write_that_waits_too_long_for_another_fails_and_writes_nothing(
        self, layout, monkeypatch
    ):
        monkeypatch.setattr(_filesystem_write, "_WAIT_S", 0.1)
        assert _filesystem_write._ONE_WRITE.acquire(timeout=1)
        try:
            failure = _failure(lambda: layout["write_file"](path="new.txt", content="x"))
        finally:
            _filesystem_write._ONE_WRITE.release()

        assert failure.error.type == "upstream" and failure.error.retryable
        assert "another write" in str(failure)
        assert not (layout.root / "new.txt").exists()

    def test_a_move_onto_another_hard_link_of_the_same_file_is_refused(self, layout):
        # POSIX rename does nothing then, and the move would say it moved (L3).
        root = layout.root
        os.link(root / "notes.txt", root / "sub" / "twin.txt")
        os.link(root / "notes.txt", root / "triplet.txt")

        for destination in ("sub/twin.txt", "triplet.txt"):
            failure = _failure(
                lambda destination=destination: layout["move_path"](
                    source="notes.txt", destination=destination, overwrite=True
                )
            )
            assert failure.error.type == "validation_error"
            assert "the same file" in str(failure)
            preview = _hook(layout, "move_path")(
                {"source": "notes.txt", "destination": destination, "overwrite": True}
            )
            assert preview.startswith("will fail (validation_error): ") and "same file" in preview
        assert (root / "notes.txt").exists() and (root / "sub" / "twin.txt").exists()

    def test_a_rename_that_changes_only_the_case_still_works(self, layout):
        root = layout.root
        if not (root / "NOTES.TXT").exists():
            pytest.skip("a case-sensitive disk: NOTES.TXT is another name")

        layout["move_path"](source="notes.txt", destination="NOTES.txt", overwrite=True)

        assert "NOTES.txt" in os.listdir(root) and "notes.txt" not in os.listdir(root)
        assert (root / "NOTES.txt").read_text() == "a needle here\n"


class TestMovesNeverWidenReads:
    """A write root that is not a read root stays unread: no move brings its files where the agent
    reads (M3)."""

    @pytest.fixture
    def home(self, tmp_path: Path) -> tuple[Path, dict[str, Tool], FilesystemPolicy]:
        home = Path(os.path.realpath(tmp_path))
        (home / "project").mkdir()
        (home / ".secret").write_text("SECRET")
        (home / "project" / "draft.txt").write_text("draft")
        policy = FilesystemPolicy(
            read_roots=(home / "project",), write_roots=(home,), cwd=home / "project"
        )
        return home, {fn.__name__: fn for fn in filesystem_tools(policy)}, policy

    def test_a_file_the_agent_cannot_read_never_moves_where_it_can(self, home):
        base, tools, _policy = home
        secret = str(base / ".secret")

        read = _failure(lambda: tools["read_file"](path=secret))
        moved = _failure(lambda: tools["move_path"](source=secret, destination="s.txt"))
        preview = tools["move_path"].__tool_definition__.preview(
            {"source": secret, "destination": "s.txt"}
        )

        assert read.error.type == "permission_denied"
        assert moved.error.type == "permission_denied"
        assert "would let the agent read it" in str(moved)
        assert preview == f"will fail (permission_denied): {moved.error.message}"
        assert (base / ".secret").read_text() == "SECRET"
        assert not (base / "project" / "s.txt").exists()

    def test_nor_over_a_file_it_reads_nor_in_another_case(self, home):
        base, tools, _policy = home
        for destination, overwrite in (("draft.txt", True), (str(base / "PROJECT" / "s"), False)):
            failure = _failure(
                lambda destination=destination, overwrite=overwrite: tools["move_path"](
                    source=str(base / ".secret"), destination=destination, overwrite=overwrite
                )
            )
            assert failure.error.type == "permission_denied", destination
        assert (base / "project" / "draft.txt").read_text() == "draft"

    def test_the_gate_refuses_it_before_the_approver(self, home):
        base, tools, policy = home
        asked: list[ApprovalRequest] = []

        def approve(request: ApprovalRequest) -> ApprovalDecision:
            asked.append(request)
            return ApprovalDecision.approve()

        group = ToolGroup(*tools.values(), gates=[PathScopeGate(policy)], approval_handler=approve)

        result = group.execute(_call("move_path", source=str(base / ".secret"), destination="s"))

        assert result.error is not None and result.error.type == "permission_denied"
        assert "would let the agent read it" in result.error.message
        assert asked == []
        assert (base / ".secret").exists()

    def test_moves_that_leave_reads_as_they_were_go_on(self, home):
        base, tools, _policy = home
        (base / "old.log").write_text("log")

        tools["move_path"](source="draft.txt", destination=str(base / "draft.txt"))
        tools["move_path"](source=str(base / "old.log"), destination=str(base / "new.log"))

        assert (base / "draft.txt").read_text() == "draft"
        assert (base / "new.log").read_text() == "log"


# --- The check before the system call (TOCTOU) --------------------------------------------------


class TestTheToolChecksAgain:
    def test_a_folder_swapped_for_a_link_during_approval_is_refused(self, layout):
        def swap_then_approve(request: ApprovalRequest) -> ApprovalDecision:
            (layout.root / "sub").rmdir()
            (layout.root / "sub").symlink_to(layout.outside, target_is_directory=True)
            return ApprovalDecision.approve()

        group = ToolGroup(
            *layout.tools.values(),
            gates=[PathScopeGate(layout.policy)],
            approval_handler=swap_then_approve,
        )

        result = group.execute(_call("write_file", path="sub/new.txt", content="pwned"))

        assert result.error is not None and result.error.type == "permission_denied"
        assert layout.outside_now() == {"secret.txt"}

    def test_modified_args_that_point_outside_are_refused(self, layout):
        outside = str(layout.outside / "new.txt")

        def redirect(request: ApprovalRequest) -> ApprovalDecision:
            return ApprovalDecision.approve(modified_args={"path": outside, "content": "pwned"})

        group = ToolGroup(
            *layout.tools.values(), gates=[PathScopeGate(layout.policy)], approval_handler=redirect
        )

        result = group.execute(_call("write_file", path="new.txt", content="fine"))

        assert result.error is not None and result.error.type == "permission_denied"
        assert layout.outside_now() == {"secret.txt"}

    def test_modified_args_that_move_a_root_are_refused(self, tmp_path):
        # Inside the write root, so only the policy's rules (never the walk) can refuse it.
        base = Path(os.path.realpath(tmp_path))
        (base / "docs").mkdir()
        (base / "draft.txt").write_text("draft")
        policy = FilesystemPolicy(read_roots=(base / "docs",), write_roots=(base,), cwd=base)

        def redirect(request: ApprovalRequest) -> ApprovalDecision:
            return ApprovalDecision.approve(
                modified_args={"source": "docs", "destination": "elsewhere"}
            )

        group = ToolGroup(
            *filesystem_tools(policy), gates=[PathScopeGate(policy)], approval_handler=redirect
        )

        result = group.execute(_call("move_path", source="draft.txt", destination="final.txt"))

        assert result.error is not None and result.error.type == "permission_denied"
        assert (base / "docs").is_dir() and not (base / "elsewhere").exists()

    def test_a_list_of_tools_without_the_gate_is_still_checked_by_the_tool(self, layout):
        result = execute_tool(
            _call("write_file", path="../outside/new.txt", content="pwned"),
            list(layout.tools.values()),
            approval_handler=_approve,
        )

        assert result.error is not None and result.error.type == "permission_denied"
        assert layout.outside_now() == {"secret.txt"}

    @pytest.mark.parametrize("name", ["write_file", "make_directory", "append_file"])
    def test_a_folder_swapped_after_the_tools_check_fails_the_walk(
        self, layout, monkeypatch, name
    ):
        (layout.root / "sub" / "notes.txt").write_text("x")
        check = FilesystemPolicy.check

        def check_then_swap(policy: FilesystemPolicy, path: Any, action: Any) -> Path:
            found = check(policy, path, action)
            if (layout.root / "sub").is_dir() and not (layout.root / "sub").is_symlink():
                (layout.root / "sub" / "notes.txt").unlink()
                (layout.root / "sub").rmdir()
                (layout.root / "sub").symlink_to(layout.outside, target_is_directory=True)
            return found

        monkeypatch.setattr(FilesystemPolicy, "check", check_then_swap)
        arguments = {"path": "sub/notes.txt"}
        if name != "make_directory":
            arguments["content"] = "pwned"

        failure = _failure(lambda: layout[name](**arguments))

        assert failure.error.type == "permission_denied"
        assert "symbolic link" in str(failure)
        assert layout.outside_now() == {"secret.txt"}
        assert (layout.outside / "secret.txt").read_text() == "OUTSIDE-SECRET\n"


class TestOneWordForOnePath:
    """The gate and the tool say the same of a path, through one classifier (L1)."""

    CASES = (
        ("read_file", {"path": "a\x00b"}),
        ("read_file", {"path": 5}),
        ("list_directory", {"path": "notes.txt/inner"}),
        ("write_file", {"path": "notes.txt/new", "content": "x"}),
        ("make_directory", {"path": "notes.txt/a/b", "parents": True}),
        ("move_path", {"source": "notes.txt", "destination": "notes.txt/inner"}),
    )

    @pytest.mark.parametrize(("name", "arguments"), CASES, ids=lambda case: str(case))
    def test_a_path_that_is_not_one_is_a_validation_error_with_or_without_the_gate(
        self, layout, name, arguments
    ):
        shown: list[str] = []

        def approve(request: ApprovalRequest) -> ApprovalDecision:
            shown.append(request.preview)
            return ApprovalDecision.approve()

        group = ToolGroup(
            *layout.tools.values(), gates=[PathScopeGate(layout.policy)], approval_handler=approve
        )

        alone = _failure(lambda: layout[name](**arguments))
        gated = group.execute(_call(name, **arguments))

        assert alone.error.type == "validation_error", alone
        assert gated.error is not None and gated.error.type == "validation_error", gated
        assert gated.error.message == alone.error.message
        if name in _WRITES:  # the approver was told it will fail
            assert shown == [f"will fail (validation_error): {alone.error.message}"]
        assert layout.outside_now() == {"secret.txt"}

    def test_a_tool_bound_to_another_policy_gets_no_unchecked_path(self, layout):
        other = FilesystemPolicy(read_roots=(layout.root,), cwd=layout.root)
        read = {fn.__name__: fn for fn in filesystem_tools(other)}["read_file"]
        group = ToolGroup(read, gates=[PathScopeGate(layout.policy)], approval_handler=_approve)

        result = group.execute(_call("read_file", path=5))

        assert result.error is not None and result.error.type == "permission_denied"

    def test_a_pattern_is_refused_in_the_gates_words_and_the_tools(self, layout):
        group = ToolGroup(
            *layout.tools.values(), gates=[PathScopeGate(layout.policy)], approval_handler=_approve
        )

        gated = group.execute(_call("list_directory", pattern="../outside/*"))
        alone = _failure(lambda: layout["list_directory"](pattern="../outside/*"))

        assert gated.error is not None and gated.error.type == alone.error.type
        assert gated.error.message.endswith(alone.error.message)


def test_a_dry_run_writes_nothing_and_records_the_canonical_path(layout):
    group = ToolGroup(
        layout["write_file"],
        gates=[PathScopeGate(layout.policy), DryRunGate()],
        approval_handler=_approve,
    )

    result = group.execute(_call("write_file", path="sub/../dry.txt", content="x"))

    assert result.ok and result.metadata["governance"]["executed"] is False
    assert result.metadata["audit"]["arguments"]["path"] == str(layout.root / "dry.txt")
    assert not (layout.root / "dry.txt").exists()


# --- The previews (C07c) -----------------------------------------------------------------------


class _Shown:
    """An approver that keeps the preview it was shown and denies, so nothing is written."""

    def __init__(self) -> None:
        self.previews: list[str] = []

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.previews.append(request.preview)
        return ApprovalDecision.deny(reason="only looking")


def _shown(layout: Layout, name: str, mode: str = "sync", **arguments: object) -> str:
    """The preview the approver of ``name`` sees, with the path gate in front."""
    shown = _Shown()
    group = ToolGroup(layout[name], gates=[PathScopeGate(layout.policy)], approval_handler=shown)
    call = _call(name, **arguments)
    result = group.execute(call) if mode == "sync" else asyncio.run(group.async_execute(call))
    assert result.error is not None and result.error.type == "approval_denied", result
    (preview,) = shown.previews
    return preview


def _hook(layout: Layout, name: str) -> Callable[[dict[str, Any]], str]:
    preview = layout[name].__tool_definition__.preview
    assert preview is not None
    return preview


def _tree(folder: Path) -> dict[str, bytes | None]:
    """Every path under ``folder`` and a file's bytes, so a test sees any change."""
    return {
        str(path.relative_to(folder)): None if path.is_dir() else path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if not path.is_symlink()
    }


class TestPreviews:
    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_a_replace_shows_the_size_change_and_the_lines_that_change(self, layout, mode):
        old = "title\nkeep one\nold line\nkeep two\n"
        new = "title\nkeep one\nnew line\nkeep two\nadded\n"
        target = layout.root / "a.md"
        target.write_text(old)

        preview = _shown(layout, "write_file", mode, path="a.md", content=new, overwrite=True)

        lines = preview.splitlines()
        assert lines[0] == f"replace {target} ({len(old)} → {len(new)} bytes)"
        assert "-old line" in lines and "+new line" in lines and "+added" in lines
        assert " keep one" in lines and " title" in lines
        assert target.read_text() == old

    def test_a_new_file_is_a_create_with_its_size_and_nothing_is_made(self, layout):
        before = _tree(layout.root)

        created = _shown(layout, "write_file", path="fresh.txt", content="héllo")
        nested = _shown(
            layout, "write_file", path="x/y/fresh.txt", content="hi", create_parents=True
        )

        assert created == f"create {layout.root / 'fresh.txt'} (6 bytes)"
        assert nested == f"create {layout.root / 'x' / 'y' / 'fresh.txt'} (2 bytes)"
        assert _tree(layout.root) == before

    def test_the_diff_is_cut_at_80_lines(self, layout):
        (layout.root / "long.txt").write_text("".join(f"old {i}\n" for i in range(200)))
        new = "".join(f"new {i}\n" for i in range(200))

        preview = _hook(layout, "write_file")(
            {"path": "long.txt", "content": new, "overwrite": True}
        )

        lines = preview.splitlines()
        assert lines[0].startswith("replace ")
        assert len(lines) == 1 + 80 + 1
        assert lines[-1] == "[the diff goes on: 403 lines in all, cut at 80 lines or 8 KB]"

    def test_the_diff_is_cut_at_8_kb(self, layout):
        (layout.root / "wide.txt").write_text("a" * 20_000 + "\n")
        new = "b" * 20_000 + "\n"

        preview = _hook(layout, "write_file")(
            {"path": "wide.txt", "content": new, "overwrite": True}
        )

        summary, *diff, note = preview.splitlines()
        assert summary.startswith("replace ")
        assert len("\n".join(diff).encode()) <= 8 * 1024
        assert diff[-1].startswith("-aaaa")  # the cut falls inside the long line
        assert note == "[the diff goes on: 5 lines in all, cut at 80 lines or 8 KB]"

    @pytest.mark.parametrize(
        ("old", "why"),
        [
            (b"\x89PNG\r\n\x1a\n\x00\x00binary", "the file is not text"),
            (b"\xff\xfe not utf-8", "the file is not text"),
            (b"valid utf-8\x00with a nul\n", "the file is not text"),
            (b"x" * (256 * 1024 + 1), "over 256 KB"),
        ],
        ids=["binary", "not-utf8", "nul", "large"],
    )
    def test_no_diff_for_a_binary_file_or_one_over_256_kb(self, layout, old, why):
        target = layout.root / "data.bin"
        target.write_bytes(old)

        preview = _hook(layout, "write_file")(
            {"path": "data.bin", "content": "x", "overwrite": True}
        )

        assert preview == f"replace {target} ({len(old)} → 1 bytes)\n(no diff: {why})"
        assert target.read_bytes() == old

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
    def test_a_file_it_cannot_read_gets_no_diff_and_still_shows_the_replace(self, layout):
        target = layout.root / "notes.txt"
        target.chmod(0o200)  # write only: replacing it needs no read
        try:
            preview = _hook(layout, "write_file")(
                {"path": "notes.txt", "content": "x", "overwrite": True}
            )
        finally:
            target.chmod(0o600)

        assert preview == f"replace {target} (14 → 1 bytes)\n(no diff: the file cannot be read)"

    def test_new_text_over_256_kb_gets_no_diff(self, layout):
        target = layout.root / "notes.txt"

        preview = _hook(layout, "write_file")(
            {"path": "notes.txt", "content": "y" * (256 * 1024 + 1), "overwrite": True}
        )

        assert preview == f"replace {target} (14 → {256 * 1024 + 1} bytes)\n(no diff: over 256 KB)"

    def test_the_same_lines_say_so(self, layout):
        preview = _hook(layout, "write_file")(
            {"path": "notes.txt", "content": "a needle here\r\n", "overwrite": True}
        )

        assert preview.splitlines()[1:] == ["(the same lines of text)"]

    def test_a_call_that_will_fail_says_why(self, layout):
        target = layout.root / "notes.txt"
        hook = _hook(layout, "write_file")

        exists = hook({"path": "notes.txt", "content": "x"})
        outside = hook({"path": "../outside/new.txt", "content": "x"})
        missing = hook({"path": "nowhere/new.txt", "content": "x"})
        not_text = hook({"path": "notes.txt", "content": 7, "overwrite": True})

        assert exists == (
            f"will fail (validation_error): {target} already exists; pass overwrite=true to "
            "replace it, or pick another name."
        )
        assert outside.startswith("will fail (permission_denied): ")
        assert missing == (
            f"will fail (not_found): the folder {layout.root / 'nowhere'} does not exist."
        )
        assert not_text.startswith("will fail (validation_error): content must be text")

    def test_a_link_at_the_leaf_is_never_read(self, layout):
        (layout.root / "leak.txt").symlink_to(layout.outside / "secret.txt")

        preview = _hook(layout, "write_file")(
            {"path": "leak.txt", "content": "x", "overwrite": True}
        )

        assert preview.startswith("will fail (permission_denied): ")
        assert "OUTSIDE-SECRET" not in preview

    def test_a_folder_swapped_for_a_link_after_the_check_is_never_read(self, layout, monkeypatch):
        (layout.root / "sub" / "secret.txt").write_text("inside\n")
        check = FilesystemPolicy.check

        def check_then_swap(policy: FilesystemPolicy, path: Any, action: Any) -> Path:
            found = check(policy, path, action)
            sub = layout.root / "sub"
            if sub.is_dir() and not sub.is_symlink():
                (sub / "secret.txt").unlink()
                sub.rmdir()
                sub.symlink_to(layout.outside, target_is_directory=True)
            return found

        monkeypatch.setattr(FilesystemPolicy, "check", check_then_swap)

        preview = _hook(layout, "write_file")(
            {"path": "sub/secret.txt", "content": "x", "overwrite": True}
        )

        assert preview.startswith("will fail (permission_denied): ")
        assert "OUTSIDE-SECRET" not in preview
        assert layout.outside_now() == {"secret.txt"}

    def test_the_old_text_is_read_without_following_a_link_swapped_in_after_the_look(
        self, layout, monkeypatch
    ):
        sub = layout.root / "sub"
        (sub / "secret.txt").write_text("inside\n")
        look = _filesystem_write._look

        def look_then_swap(*args: Any, **kwargs: Any) -> os.stat_result | None:
            found = look(*args, **kwargs)
            (sub / "secret.txt").unlink()
            sub.rmdir()
            sub.symlink_to(layout.outside, target_is_directory=True)
            return found

        monkeypatch.setattr(_filesystem_write, "_look", look_then_swap)

        preview = _hook(layout, "write_file")(
            {"path": "sub/secret.txt", "content": "x", "overwrite": True}
        )

        assert preview.startswith("will fail (permission_denied): ")
        assert "OUTSIDE-SECRET" not in preview

    def test_the_other_writes_say_what_they_will_do_in_one_line(self, layout):
        (layout.root / "sub" / "old.txt").write_text("12345")
        root = layout.root

        assert _shown(layout, "append_file", path="notes.txt", content="more!\n") == (
            f"append 6 bytes to {root / 'notes.txt'} (14 → 20 bytes)"
        )
        assert _shown(layout, "make_directory", path="new") == f"make folder {root / 'new'}"
        assert _shown(layout, "make_directory", path="sub") == (
            f"{root / 'sub'} already exists; nothing changes"
        )
        assert _shown(layout, "move_path", source="notes.txt", destination="sub/n.txt") == (
            f"move file {root / 'notes.txt'} to {root / 'sub' / 'n.txt'}"
        )
        assert _shown(layout, "move_path", source="sub", destination="moved") == (
            f"move folder {root / 'sub'} to {root / 'moved'}"
        )
        assert _shown(
            layout, "move_path", source="notes.txt", destination="sub/old.txt", overwrite=True
        ) == (
            f"move file {root / 'notes.txt'} to {root / 'sub' / 'old.txt'}, replacing the file "
            "there (5 bytes)"
        )

    def test_the_other_writes_say_why_they_will_fail(self, layout):
        root = layout.root
        os.link(root / "notes.txt", root / "sub" / "twin.txt")

        missing = _hook(layout, "append_file")({"path": "none.txt", "content": "x"})
        linked = _hook(layout, "append_file")({"path": "notes.txt", "content": "x"})
        in_the_way = _hook(layout, "make_directory")({"path": "notes.txt"})
        no_source = _hook(layout, "move_path")({"source": "none", "destination": "b"})
        taken = _hook(layout, "move_path")({"source": "sub", "destination": "notes.txt"})

        assert missing == (
            f"will fail (not_found): {root / 'none.txt'} does not exist; write_file makes it."
        )
        assert linked.startswith("will fail (permission_denied): ") and "2 hard links" in linked
        assert in_the_way == (
            f"will fail (validation_error): {root / 'notes.txt'} exists and is not a folder; "
            "pick another name."
        )
        assert no_source == (
            f"will fail (not_found): {root / 'none'} does not exist; list_directory shows what "
            "is there."
        )
        assert taken.startswith(f"will fail (validation_error): {root / 'notes.txt'} already")

    def test_no_preview_writes_anything(self, layout):
        (layout.root / "sub" / "old.txt").write_text("12345")
        before = _tree(layout.root)
        calls = {
            "write_file": {"path": "a/b/c.txt", "content": "x", "create_parents": True},
            "append_file": {"path": "notes.txt", "content": "more"},
            "make_directory": {"path": "p/q/r", "parents": True},
            "move_path": {"source": "notes.txt", "destination": "sub/old.txt", "overwrite": True},
        }

        for name, arguments in calls.items():
            assert not _hook(layout, name)(arguments).startswith("will fail"), name

        assert _tree(layout.root) == before
        assert _leftovers(layout.root) == []

    def test_a_dry_run_after_the_gate_writes_and_meters_nothing_and_records_the_preview(
        self, layout
    ):
        target = layout.root / "notes.txt"
        shown = _Shown()
        group = ToolGroup(
            layout["write_file"],
            gates=[PathScopeGate(layout.policy), DryRunGate()],
            approval_handler=shown,
        )

        with MeterScope() as scope:
            result = group.execute(
                _call("write_file", path="sub/../notes.txt", content="a pin\n", overwrite=True)
            )

        assert result.ok and result.metadata["governance"]["executed"] is False
        assert scope.snapshot().tool_calls == 0
        assert shown.previews == []  # the dry run stops the call before anyone is asked
        preview = result.metadata["audit"]["preview"].splitlines()
        assert preview[0] == f"replace {target} (14 → 6 bytes)"
        assert "-a needle here" in preview and "+a pin" in preview
        assert target.read_text() == "a needle here\n"


# --- The bound reads ---------------------------------------------------------------------------


class TestBoundReads:
    def test_a_search_never_reads_a_file_linked_from_outside(self, layout):
        (layout.root / "leaf_link.txt").symlink_to(layout.outside / "secret.txt")
        (layout.root / "door").symlink_to(layout.outside, target_is_directory=True)

        text = _text(layout["search_files"](directory=".", pattern="secret"))

        assert "OUTSIDE-SECRET" not in text
        assert text == f"No matches for 'secret' in {layout.root}"

    def test_a_search_names_each_file_by_the_path_read_file_takes(self, layout):
        text = _text(layout["search_files"](directory="sub/..", pattern="needle"))

        hit = f"{layout.root / 'notes.txt'}:1:0: a needle here"
        assert hit in text.splitlines()
        assert "a needle here" in _text(layout["read_file"](path=str(layout.root / "notes.txt")))

    def test_a_listing_refuses_a_pattern_that_climbs_out(self, layout):
        for pattern in ("../outside/*", "sub/../../*", "..\\outside\\*"):
            failure = _failure(lambda pattern=pattern: layout["list_directory"](pattern=pattern))
            assert failure.error.type == "permission_denied"
            assert "'..'" in str(failure)

    def test_a_listing_leaves_out_a_link_whose_target_is_outside(self, layout):
        (layout.root / "leak.txt").symlink_to(layout.outside / "secret.txt")
        (layout.root / "door").symlink_to(layout.outside, target_is_directory=True)
        (layout.root / "alias.txt").symlink_to(layout.root / "notes.txt")

        text = _text(layout["list_directory"]())

        assert "notes.txt" in text and "alias.txt" in text and "sub/" in text
        assert "leak.txt" not in text and "door" not in text

    def test_reading_a_link_to_outside_is_refused(self, layout):
        (layout.root / "leak.txt").symlink_to(layout.outside / "secret.txt")

        failure = _failure(lambda: layout["read_file"](path="leak.txt"))

        assert failure.error.type == "permission_denied"
        assert "OUTSIDE-SECRET" not in str(failure)

    def test_a_link_inside_is_read_where_it_points(self, layout):
        (layout.root / "alias.txt").symlink_to(layout.root / "notes.txt")

        assert "a needle here" in _text(layout["read_file"](path="alias.txt"))

    def test_a_relative_path_starts_from_the_policys_cwd(self, layout, monkeypatch):
        monkeypatch.chdir(layout.outside)

        assert "a needle here" in _text(layout["read_file"](path="notes.txt"))
        assert "notes.txt" in _text(layout["list_directory"]())

    def test_a_folder_is_not_read_as_a_file(self, layout):
        for path in (".", "sub"):
            failure = _failure(lambda path=path: layout["read_file"](path=path))
            assert failure.error.type == "validation_error"
            assert "not a regular file" in str(failure)

    def test_a_missing_file_is_not_found(self, layout):
        assert _failure(lambda: layout["read_file"](path="gone.txt")).error.type == "not_found"

    @pytest.mark.timeout(10)
    def test_a_pipe_never_holds_a_read(self, layout):
        os.mkfifo(layout.root / "pipe")

        failure = _failure(lambda: layout["read_file"](path="pipe"))

        assert failure.error.type == "validation_error"

    def test_a_file_swapped_for_a_link_after_the_check_is_not_read(self, layout, monkeypatch):
        check = FilesystemPolicy.check

        def check_then_swap(policy: FilesystemPolicy, path: Any, action: Any) -> Path:
            found = check(policy, path, action)
            target = layout.root / "notes.txt"
            if not target.is_symlink():
                target.unlink()
                target.symlink_to(layout.outside / "secret.txt")
            return found

        monkeypatch.setattr(FilesystemPolicy, "check", check_then_swap)

        failure = _failure(lambda: layout["read_file"](path="notes.txt"))

        assert failure.error.type == "permission_denied"
        assert "OUTSIDE-SECRET" not in str(failure)

    def test_a_search_never_reads_a_file_that_became_a_link_after_it_looked(
        self, layout, monkeypatch
    ):
        (layout.root / "leaf_link.txt").symlink_to(layout.outside / "secret.txt")
        # The search's look at each file sees no link, as if the file was swapped after it.
        monkeypatch.setattr(Path, "is_symlink", lambda self: False)

        text = _text(layout["search_files"](directory=".", pattern="secret"))

        assert "OUTSIDE-SECRET" not in text
        assert "cannot read" in text and "leaf_link.txt" in text

    def test_no_descriptor_stays_open_after_reads(self, layout):
        before = _open_descriptors()

        layout["read_file"](path="notes.txt")
        layout["search_files"](directory=".", pattern="needle")
        _failure(lambda: layout["read_file"](path="sub"))
        _failure(lambda: layout["read_file"](path="gone.txt"))

        assert _open_descriptors() == before

    @pytest.mark.parametrize("name", _READS)
    def test_hostile_arguments_give_text_or_a_typed_failure(self, layout, name):
        for label, arguments in _plans(name):
            try:
                answer = layout[name](**arguments)
            except ToolFailure as failure:
                assert failure.error.message, label
                continue
            assert isinstance(answer, ToolResult) and answer.ok, label
        assert layout.outside_now() == {"secret.txt"}
