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
    ToolCall,
    ToolFailure,
    ToolGroup,
    ToolResult,
    execute_tool,
)
from ai_arch_toolkit.toolkit.tools import dangerous
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
        self, layout, monkeypatch
    ):
        # Both writers hold their whole temporary file before either links it into place.
        both_ready = threading.Barrier(2, timeout=5)
        link = os.link

        def link_together(*args: Any, **kwargs: Any) -> None:
            both_ready.wait()
            link(*args, **kwargs)

        monkeypatch.setattr(os, "link", link_together)
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
