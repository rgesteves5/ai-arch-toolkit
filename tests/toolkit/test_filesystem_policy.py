"""The ``FilesystemPolicy`` and its ``PathScopeGate`` (C07a, C07b; D64).

``check`` is the one test of a path; the gate runs it before anyone is asked, and refuses with the
word the tools use, ``permission_denied``, before the approver and the meter.
"""

from __future__ import annotations

import ast
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ai_arch_toolkit.core import (
    ApprovalDecision,
    ApprovalRequest,
    MeterScope,
    ToolCall,
    ToolFailure,
    ToolGroup,
    tool,
)
from ai_arch_toolkit.toolkit.tools import _filesystem_policy
from ai_arch_toolkit.toolkit.tools._filesystem_policy import open_beneath, opened_folder
from ai_arch_toolkit.toolkit.tools._json import csv_read
from ai_arch_toolkit.toolkit.tools.dangerous import (
    FilesystemPolicy,
    FilesystemPolicyError,
    PathScopeGate,
    filesystem_tools,
    list_directory,
    read_file,
)


def _any(value: object) -> Any:
    """``value``, for an argument of the wrong type on purpose."""
    return value


@dataclass(frozen=True, slots=True)
class Layout:
    """A root the policy allows, and a folder next to it that it does not."""

    root: Path
    outside: Path
    policy: FilesystemPolicy


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    base = Path(os.path.realpath(tmp_path))
    root, outside = base / "root", base / "outside"
    (root / "sub").mkdir(parents=True)
    outside.mkdir()
    (root / "notes.txt").write_text("a note\n")
    (outside / "secret.txt").write_text("OUTSIDE-SECRET\n")
    policy = FilesystemPolicy(read_roots=(root,), write_roots=(root,), cwd=root)
    return Layout(root, outside, policy)


def _refused(policy: FilesystemPolicy, path: str | Path, action: Any) -> str:
    with pytest.raises(FilesystemPolicyError) as caught:
        policy.check(path, action)
    return str(caught.value)


class TestCheck:
    @pytest.mark.parametrize("action", ["read", "write"])
    def test_a_path_that_climbs_out_with_dot_dot_is_refused(self, layout, action):
        message = _refused(layout.policy, "../outside/secret.txt", action)

        assert "'../outside/secret.txt' is outside" in message
        assert "pick a path inside" in message
        # The refusal names the path as given, never where it leads (M1).
        assert str(layout.outside) not in message

    def test_a_link_at_the_leaf_is_never_written(self, layout):
        (layout.root / "inside.txt").symlink_to(layout.root / "notes.txt")

        assert "symbolic link" in _refused(layout.policy, "inside.txt", "write")

    def test_a_link_at_the_leaf_is_read_where_it_points_inside_only(self, layout):
        (layout.root / "inside.txt").symlink_to(layout.root / "notes.txt")
        (layout.root / "leak.txt").symlink_to(layout.outside / "secret.txt")

        assert layout.policy.check("inside.txt", "read") == layout.root / "notes.txt"
        _refused(layout.policy, "leak.txt", "read")

    @pytest.mark.parametrize("action", ["read", "write"])
    def test_a_link_in_a_middle_folder_that_leads_out_is_refused(self, layout, action):
        (layout.root / "door").symlink_to(layout.outside, target_is_directory=True)

        message = _refused(layout.policy, "door/secret.txt", action)
        assert "'door/secret.txt' is outside" in message
        assert str(layout.outside) not in message

    @pytest.mark.parametrize("action", ["read", "write"])
    @pytest.mark.parametrize("given", ["notes.txt/inner.txt", "notes.txt/a/b"])
    def test_a_path_through_a_file_inside_is_a_malformed_path(self, layout, action, given):
        # Inside the roots it is the path's own fault, as for the OS: never a refusal (L1).
        with pytest.raises(ValueError, match="a part of it is a file") as caught:
            layout.policy.check(given, action)

        assert not isinstance(caught.value, FilesystemPolicyError)

    @pytest.mark.parametrize("action", ["read", "write"])
    def test_a_loop_of_links_inside_is_a_malformed_path(self, layout, action):
        (layout.root / "one").symlink_to(layout.root / "two")
        (layout.root / "two").symlink_to(layout.root / "one")

        with pytest.raises(ValueError, match="loop") as caught:
            layout.policy.check("one/file.txt", action)

        assert not isinstance(caught.value, FilesystemPolicyError)

    def test_a_destination_that_does_not_exist_yet_is_accepted(self, layout):
        found = layout.policy.check("new/deeper/file.txt", "write")

        assert found == layout.root / "new" / "deeper" / "file.txt"

    def test_a_relative_path_starts_from_the_policys_cwd(self, layout, monkeypatch):
        monkeypatch.chdir(layout.outside)

        assert layout.policy.check("notes.txt", "read") == layout.root / "notes.txt"
        assert layout.policy.check("sub/new.txt", "write") == layout.root / "sub" / "new.txt"

    def test_the_root_itself_is_read_but_never_written(self, layout):
        assert layout.policy.check(".", "read") == layout.root
        assert layout.policy.check(str(layout.root), "read") == layout.root
        message = _refused(layout.policy, str(layout.root), "write")
        assert "never created, moved or replaced" in message

    def test_a_folder_that_holds_a_root_is_never_written(self, tmp_path):
        base = Path(os.path.realpath(tmp_path))
        (base / "docs").mkdir()
        policy = FilesystemPolicy(read_roots=(base / "docs",), write_roots=(base,), cwd=base)

        assert "never created" in _refused(policy, "docs", "write")
        # A case-insensitive disk would make DOCS the same folder: refused all the same.
        assert "never created" in _refused(policy, "DOCS", "write")

    @pytest.mark.parametrize("given", [".", "sub/..", "sub/.", ""])
    def test_a_path_that_names_no_file_is_never_written(self, layout, given):
        message = _refused(layout.policy, given, "write")
        assert "names no file" in message or "never created" in message

    def test_case_is_never_folded_into_an_acceptance(self, tmp_path):
        base = Path(os.path.realpath(tmp_path))
        (base / "Root").mkdir()
        policy = FilesystemPolicy(read_roots=(base / "Root",), cwd=base)

        # Same folder on a case-insensitive disk, another one elsewhere: refused either way.
        _refused(policy, "root/file.txt", "read")
        assert policy.check("Root/file.txt", "read") == base / "Root" / "file.txt"

    def test_an_action_without_roots_reaches_nothing(self, layout):
        policy = FilesystemPolicy(read_roots=(layout.root,), cwd=layout.root)

        assert "no folder" in _refused(policy, "notes.txt", "write")
        assert "no folder" in _refused(policy, "notes.txt", "delete")

    @pytest.mark.parametrize("action", ["read", "write"])
    def test_a_malformed_path_is_a_value_error_not_a_refusal(self, layout, action):
        with pytest.raises(ValueError, match="null") as caught:
            layout.policy.check("a\x00b", action)

        assert not isinstance(caught.value, FilesystemPolicyError)

    def test_an_unknown_action_is_a_value_error(self, layout):
        with pytest.raises(ValueError, match="unknown filesystem action"):
            layout.policy.check("notes.txt", _any("erase"))

    def test_the_root_of_a_path_is_the_deepest_that_holds_it(self, layout):
        policy = FilesystemPolicy(write_roots=(layout.root, layout.root / "sub"), cwd=layout.root)

        assert policy.root_of(layout.root / "sub" / "a.txt", "write") == layout.root / "sub"
        assert policy.root_of(layout.root / "a.txt", "write") == layout.root
        with pytest.raises(FilesystemPolicyError):
            policy.root_of(layout.outside / "a.txt", "write")


class TestNothingIsLearntOutside:
    """A refusal says the same of every path outside the roots, whatever is there (M1).

    Nothing outside is looked at before the roots are tested, and a refusal names the path as
    given, never where it leads.
    """

    @pytest.fixture
    def outside(self, layout: Layout) -> dict[str, str]:
        """Paths outside the roots, one for each kind of thing that could be there."""
        out = layout.outside
        (out / "folder").mkdir()
        (out / "alink").symlink_to(out.parent / "elsewhere" / "target.txt")
        (out / "folderlink").symlink_to(out / "folder", target_is_directory=True)
        (out / "one").symlink_to(out / "two")
        (out / "two").symlink_to(out / "one")
        paths = {
            "a file": "secret.txt",
            "nothing": "missing.txt",
            "a folder": "folder",
            "through a file": "secret.txt/x",
            "through nothing": "missing/x",
            "a link": "alink",
            "through a link": "folderlink/x",
            "a loop": "one/x",
            "a loop at the end": "one",
        }
        return {kind: str(out / path) for kind, path in paths.items()}

    @staticmethod
    def _shape(message: str, given: str) -> str:
        return message.replace(repr(given), "<path>")

    @pytest.mark.parametrize("action", ["read", "write"])
    def test_every_path_outside_gets_the_same_refusal(self, layout, outside, action):
        shapes = {
            kind: self._shape(_refused(layout.policy, given, action), given)
            for kind, given in outside.items()
        }

        assert len(set(shapes.values())) == 1, shapes
        (shape,) = set(shapes.values())
        assert shape.startswith("<path> is outside the folders this policy lets the agent")
        assert str(layout.outside.parent / "elsewhere") not in shape

    @pytest.mark.skipif(os.geteuid() == 0, reason="root searches any folder")
    def test_a_folder_outside_the_process_cannot_search_gets_the_same_refusal(
        self, layout, outside
    ):
        private = layout.outside / "private"
        private.mkdir()
        private.chmod(0o000)
        try:
            shut = _refused(layout.policy, str(private / "x"), "read")
        finally:
            private.chmod(0o700)

        given = outside["nothing"]
        missing = _refused(layout.policy, given, "read")
        assert self._shape(shut, str(private / "x")) == self._shape(missing, given)

    def test_nothing_outside_is_looked_at_before_the_roots_are_tested(
        self, layout, outside, monkeypatch
    ):
        looked: list[str] = []
        lstat = os.lstat

        def spying(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            looked.append(os.fspath(path))
            return lstat(path, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", spying)

        for given in outside.values():
            looked.clear()
            _refused(layout.policy, given, "write")
            # Resolving the folder that would hold the path is all: never the path itself.
            assert given not in looked, given

    def test_a_refusal_never_names_where_a_link_inside_leads(self, layout):
        (layout.root / "leak.txt").symlink_to(layout.outside / "secret.txt")
        (layout.root / "door").symlink_to(layout.outside, target_is_directory=True)

        for given in ("leak.txt", "door", "door/secret.txt"):
            message = _refused(layout.policy, given, "read")
            assert f"{given!r} is outside" in message
            assert str(layout.outside) not in message

    @pytest.mark.parametrize("name", ["read_file", "write_file"])
    def test_the_gate_and_the_tool_refuse_every_path_outside_alike(self, layout, outside, name):
        tool_ = {fn.__name__: fn for fn in filesystem_tools(layout.policy)}[name]
        group = _group(layout.policy, _Approver(), tool_)
        extra = {"content": "x"} if name == "write_file" else {}
        shapes: set[str] = set()

        for given in outside.values():
            gated = group.execute(_call(name, path=given, **extra))
            with pytest.raises(ToolFailure) as alone:
                tool_(path=given, **extra)
            assert gated.error is not None and gated.error.type == "permission_denied"
            assert alone.value.error.type == "permission_denied"
            assert gated.error.message.endswith(alone.value.error.message)
            assert gated.metadata["audit"]["filesystem"]["path"] == given
            shapes.add(self._shape(gated.error.message, given))

        assert len(shapes) == 1, shapes


class TestConstruction:
    def test_roots_and_cwd_are_stored_canonical(self, layout):
        alias = layout.outside / "alias"
        alias.symlink_to(layout.root, target_is_directory=True)

        policy = FilesystemPolicy(read_roots=(alias,), cwd=alias)

        assert policy.read_roots == (layout.root,)
        assert policy.cwd == layout.root

    def test_cwd_defaults_to_the_working_directory_when_built(self, layout, monkeypatch):
        monkeypatch.chdir(layout.root)

        assert FilesystemPolicy(read_roots=(Path("sub"),)).cwd == layout.root

    def test_relative_roots_come_from_the_working_directory_not_from_cwd(
        self, layout, monkeypatch
    ):
        monkeypatch.chdir(layout.root)

        policy = FilesystemPolicy(read_roots=(Path("sub"),), cwd=layout.outside)

        assert policy.read_roots == (layout.root / "sub",)
        assert policy.cwd == layout.outside

    def test_a_root_that_is_not_an_existing_folder_is_refused(self, layout):
        with pytest.raises(ValueError, match="not an existing folder"):
            FilesystemPolicy(write_roots=(layout.root / "missing",))
        with pytest.raises(ValueError, match="not a folder"):
            FilesystemPolicy(write_roots=(layout.root / "notes.txt",))

    def test_a_single_path_in_place_of_a_tuple_is_a_type_error(self, layout):
        with pytest.raises(TypeError, match="tuple of folders"):
            FilesystemPolicy(read_roots=_any(str(layout.root)))

    @pytest.mark.parametrize("size", [0, -1, True, 1.5])
    def test_max_write_bytes_must_be_a_positive_integer(self, layout, size):
        with pytest.raises(ValueError, match="max_write_bytes"):
            FilesystemPolicy(write_roots=(layout.root,), max_write_bytes=size)


class TestTheWalk:
    """The tools reach a path from its root down, one folder at a time (C07d)."""

    def test_a_link_on_the_way_fails_the_walk(self, layout):
        (layout.root / "door").symlink_to(layout.outside, target_is_directory=True)

        with (
            pytest.raises(FilesystemPolicyError, match="symbolic link"),
            opened_folder(layout.policy, layout.root / "door", "write"),
        ):
            pass

    def test_a_link_at_the_end_is_never_opened(self, layout):
        (layout.root / "leak.txt").symlink_to(layout.outside / "secret.txt")

        with pytest.raises(FilesystemPolicyError, match="symbolic link"):
            open_beneath(layout.policy, layout.root / "leak.txt", "read", os.O_RDONLY)

    @pytest.mark.parametrize("names", [("..", "outside"), ("sub", "..", "..")])
    def test_the_walk_only_goes_down(self, layout, names):
        with (
            pytest.raises(FilesystemPolicyError, match="not canonical"),
            opened_folder(layout.policy, layout.root.joinpath(*names), "write"),
        ):
            pass
        with pytest.raises(FilesystemPolicyError, match="not canonical"):
            open_beneath(layout.policy, layout.root.joinpath(*names, "x"), "read", os.O_RDONLY)

    def test_a_root_replaced_by_a_link_fails_the_walk(self, layout):
        moved = layout.root.with_name("moved")
        layout.root.rename(moved)
        layout.root.symlink_to(layout.outside, target_is_directory=True)

        with (
            pytest.raises(FilesystemPolicyError, match="symbolic link"),
            opened_folder(layout.policy, layout.root, "write"),
        ):
            pass

    def test_create_makes_the_missing_folders_and_opens_each(self, layout):
        target = layout.root / "a" / "b"

        with opened_folder(layout.policy, target, "write", create=True) as descriptor:
            assert os.path.samestat(os.fstat(descriptor), target.stat())


def test_a_path_that_does_not_exist_yet_is_resolved_only_with_allow_missing():
    # D64, C07.4: one way to resolve, and no lexical shortcut (resolve(), abspath, normpath).
    tree = ast.parse(inspect.getsource(_filesystem_policy))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {ast.unparse(node.func) for node in calls}
    stricts = {
        ast.unparse(keyword.value)
        for node in calls
        if ast.unparse(node.func) == "os.path.realpath"
        for keyword in node.keywords
        if keyword.arg == "strict"
    }
    realpaths = [node for node in calls if ast.unparse(node.func) == "os.path.realpath"]

    assert stricts == {"os.path.ALLOW_MISSING", "True"}  # True: a root must exist
    assert all(any(k.arg == "strict" for k in node.keywords) for node in realpaths)
    assert not {name for name in names if name.endswith((".resolve", "abspath", "normpath"))}


# --- The gate ------------------------------------------------------------------------------------


class _Approver:
    """Approves every call and remembers what it was asked."""

    def __init__(self) -> None:
        self.asked: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.asked.append(request)
        return ApprovalDecision.approve()


def _group(policy: FilesystemPolicy, approver: _Approver, *tools, **gate) -> ToolGroup:
    return ToolGroup(*tools, gates=[PathScopeGate(policy, **gate)], approval_handler=approver)


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="call", name=name, input=arguments)


class TestPathScopeGate:
    def test_a_path_outside_is_refused_before_the_approver_and_the_meter(self, layout):
        approver = _Approver()
        group = _group(layout.policy, approver, read_file)

        with MeterScope() as scope:
            result = group.execute(_call("read_file", path="../outside/secret.txt"))

        assert result.error is not None and result.error.type == "permission_denied"
        assert "pick a path inside" in result.error.message
        assert approver.asked == []
        assert scope.snapshot().tool_calls == 0
        audit = result.metadata["audit"]["filesystem"]
        assert audit["tool"] == "read_file" and audit["argument"] == "path"
        assert audit["action"] == "read" and audit["path"] == "../outside/secret.txt"
        assert audit["refused"] == result.error.message

    def test_an_allowed_call_reaches_the_approver_with_its_paths_canonical(self, layout):
        approver = _Approver()
        group = _group(layout.policy, approver, read_file)

        result = group.execute(_call("read_file", path="sub/../notes.txt"))

        assert result.ok, result
        assert approver.asked[0].arguments["path"] == str(layout.root / "notes.txt")
        assert result.metadata["audit"]["filesystem"] == {
            "path": {"action": "read", "path": str(layout.root / "notes.txt")}
        }

    def test_an_omitted_path_takes_its_default_from_the_policys_cwd(self, layout, monkeypatch):
        monkeypatch.chdir(layout.outside)  # the process's working directory is not the policy's
        approver = _Approver()

        result = _group(layout.policy, approver, list_directory).execute(_call("list_directory"))

        assert result.ok, result
        assert approver.asked[0].arguments == {"path": str(layout.root)}
        assert "notes.txt" in result.value

    def test_a_mapped_argument_omitted_without_a_default_is_refused(self, layout):
        @tool(capability="filesystem")
        def touch(target: str | None) -> str:
            """Touch a file.

            Args:
                target: The file.
            """
            return "touched"

        approver = _Approver()
        group = _group(layout.policy, approver, touch, paths={"touch": {"target": "write"}})

        result = group.execute(_call("touch"))

        assert result.error is not None and result.error.type == "permission_denied"
        assert "'target' must be a path" in result.error.message
        assert approver.asked == []

    def test_a_filesystem_tool_without_a_map_is_blocked(self, layout):
        (layout.root / "data.csv").write_text("a\n1\n")
        approver = _Approver()

        result = _group(layout.policy, approver, csv_read).execute(
            _call("csv_read", path=str(layout.root / "data.csv"))
        )

        assert result.error is not None and result.error.type == "permission_denied"
        assert "PathScopeGate(paths={'csv_read': {'path': 'read'}})" in result.error.message
        assert approver.asked == []

    def test_the_block_of_an_unmapped_tool_names_its_own_arguments(self, layout):
        @tool(capability="filesystem")
        def touch(target: str, mode: int = 0) -> str:
            """Touch a file.

            Args:
                target: The file.
                mode: Its permission bits.
            """
            return target

        result = _group(layout.policy, _Approver(), touch).execute(_call("touch", target="x"))

        assert result.error is not None
        assert "it takes 'target', 'mode'" in result.error.message
        assert "PathScopeGate(paths={'touch': {'target': 'read'}})" in result.error.message

    def test_a_tool_that_is_not_a_filesystem_one_passes_untouched(self, layout):
        @tool
        def echo(path: str) -> str:
            """Echo a text.

            Args:
                path: Any text.
            """
            return path

        result = ToolGroup(echo, gates=[PathScopeGate(layout.policy)]).execute(
            _call("echo", path="../../anything")
        )

        assert result.ok and result.value == "../../anything"

    def test_paths_add_a_tool_and_replace_its_map(self, layout):
        @tool(capability="filesystem")
        def peek(where: str) -> str:
            """Peek at a file.

            Args:
                where: The file.
            """
            return where

        approver = _Approver()
        gate = {"paths": {"peek": {"where": "read"}}}

        allowed = _group(layout.policy, approver, peek, **gate).execute(
            _call("peek", where="notes.txt")
        )
        refused = _group(layout.policy, approver, peek, **gate).execute(
            _call("peek", where="../outside/secret.txt")
        )

        assert allowed.ok and allowed.value == str(layout.root / "notes.txt")
        assert refused.error is not None and refused.error.type == "permission_denied"
        # The seven tools keep their maps next to the added one.
        assert (
            _group(layout.policy, approver, read_file, **gate)
            .execute(_call("read_file", path="notes.txt"))
            .ok
        )

    def test_an_unknown_action_in_paths_is_a_value_error(self, layout):
        with pytest.raises(ValueError, match="unknown actions"):
            PathScopeGate(layout.policy, paths={"peek": {"where": _any("erase")}})

    @pytest.mark.parametrize(
        "given",
        ["a" * 5000, "a\x00b", 5, "notes.txt/inner.txt"],
        ids=["long", "nul", "int", "file"],
    )
    def test_what_the_gate_cannot_check_never_reaches_a_tool_that_does_not_check(
        self, layout, given
    ):
        # The module's read_file has no check of its own: the gate fails closed (L1).
        approver = _Approver()

        result = _group(layout.policy, approver, read_file).execute(_call("read_file", path=given))

        assert result.error is not None and result.error.type == "permission_denied"
        assert result.error.message.startswith("The tool 'read_file' did not run: ")
        assert result.error.message.endswith("This run cannot check it, so the tool does not run.")
        assert approver.asked == []

    def test_a_pattern_that_climbs_out_is_refused_before_the_approver(self, layout):
        approver = _Approver()
        group = _group(layout.policy, approver, list_directory)

        for pattern in ("../outside/*", "sub/../../*", "..\\outside\\*"):
            result = group.execute(_call("list_directory", pattern=pattern))
            assert result.error is not None and result.error.type == "permission_denied"
            assert "climbs out of the folder with '..'" in result.error.message
            assert result.metadata["audit"]["filesystem"]["argument"] == "pattern"
        assert approver.asked == []
        assert group.execute(_call("list_directory", pattern="*.txt")).ok

    async def test_the_async_path_checks_the_same(self, layout):
        approver = _Approver()
        group = _group(layout.policy, approver, read_file)

        refused = await group.async_execute(_call("read_file", path="../outside/secret.txt"))
        allowed = await group.async_execute(_call("read_file", path="notes.txt"))

        assert refused.error is not None and refused.error.type == "permission_denied"
        assert allowed.ok and approver.asked[0].arguments["path"] == str(layout.root / "notes.txt")
