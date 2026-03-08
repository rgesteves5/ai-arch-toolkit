"""Tests for State and StateSnapshot."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._state import MergeConflictError, State, StateSnapshot
from ai_arch_toolkit.core._step import Result


class TestStateSnapshot:
    def test_layered_lookup(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {"a": 1},
                "operational": {"a": 2, "b": 3},
                "persistent": {"c": 4},
                "world": {"d": 5},
            }
        )
        assert snap["a"] == 1  # current wins
        assert snap["b"] == 3
        assert snap["c"] == 4
        assert snap["d"] == 5

    def test_contains(self) -> None:
        snap = StateSnapshot.from_dict({"operational": {"x": 1}})
        assert "x" in snap
        assert "y" not in snap

    def test_get_default(self) -> None:
        snap = StateSnapshot()
        assert snap.get("missing") is None
        assert snap.get("missing", 42) == 42

    def test_require_success(self) -> None:
        snap = StateSnapshot.from_dict({"current": {"key": "val"}})
        assert snap.require("key") == "val"

    def test_require_raises(self) -> None:
        snap = StateSnapshot()
        with pytest.raises(KeyError, match="not found"):
            snap.require("missing")

    def test_keys_union(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {"a": 1},
                "operational": {"b": 2},
                "persistent": {"c": 3},
                "world": {"a": 99},
            }
        )
        assert snap.keys() == frozenset({"a", "b", "c"})

    def test_roundtrip(self) -> None:
        data = {"current": {"x": 1}, "operational": {}, "persistent": {}, "world": {}}
        snap = StateSnapshot.from_dict(data)
        assert snap.to_dict() == data


class TestState:
    def test_getitem_setitem(self) -> None:
        s = State()
        s["key"] = "value"
        assert s["key"] == "value"

    def test_layered_lookup(self) -> None:
        s = State(current={"a": 1}, operational={"a": 2})
        assert s["a"] == 1  # current wins

    def test_set_layer(self) -> None:
        s = State()
        s.set("x", 10, layer="persistent")
        assert s.persistent == {"x": 10}
        assert s.operational == {}

    def test_set_invalid_layer(self) -> None:
        s = State()
        with pytest.raises(ValueError, match="Unknown layer"):
            s.set("x", 1, layer="invalid")

    def test_contains(self) -> None:
        s = State(world={"w": 1})
        assert "w" in s
        assert "z" not in s

    def test_require(self) -> None:
        s = State(operational={"key": 42})
        assert s.require("key") == 42

    def test_snapshot_immutable(self) -> None:
        s = State(operational={"x": 1})
        snap = s.snapshot()
        s["x"] = 999
        assert snap["x"] == 1  # snapshot unchanged

    def test_fork_independent(self) -> None:
        s = State(operational={"x": 1}, world={"w": "shared"})
        f = s.fork()
        f["x"] = 999
        assert s["x"] == 1  # original unchanged
        # World is shared
        assert f["w"] == "shared"

    def test_merge_last_wins(self) -> None:
        s = State()
        r1 = Result(artifacts={"a": 1})
        r2 = Result(artifacts={"a": 2, "b": 3})
        s.merge(r1, r2, strategy="last_wins")
        assert s["a"] == 2
        assert s["b"] == 3

    def test_merge_collect(self) -> None:
        s = State()
        r1 = Result(artifacts={"a": 1})
        r2 = Result(artifacts={"a": 2})
        s.merge(r1, r2, strategy="collect")
        assert s["a"] == [1, 2]

    def test_merge_raise_on_conflict(self) -> None:
        s = State(operational={"a": 0})
        r1 = Result(artifacts={"a": 1})
        r2 = Result(artifacts={"a": 2})
        with pytest.raises(MergeConflictError):
            s.merge(r1, r2, strategy="raise")

    def test_merge_raise_no_conflict(self) -> None:
        s = State()
        r1 = Result(artifacts={"a": 1})
        r2 = Result(artifacts={"b": 2})
        s.merge(r1, r2, strategy="raise")
        assert s["a"] == 1
        assert s["b"] == 2

    def test_merge_invalid_type(self) -> None:
        s = State()
        with pytest.raises(TypeError, match="Expected Result"):
            s.merge("not a result")  # type: ignore[arg-type]

    def test_roundtrip(self) -> None:
        s = State(current={"a": 1}, operational={"b": 2})
        data = s.to_dict()
        s2 = State.from_dict(data)
        assert s2["a"] == 1
        assert s2["b"] == 2

    def test_from_trace(self) -> None:
        from ai_arch_toolkit.core._trace import Trace

        trace = Trace(
            flow_name="test",
            initial_state={"current": {"x": 1}, "operational": {}, "persistent": {}, "world": {}},
        )
        s = State.from_trace(trace)
        assert s["x"] == 1

    def test_keys(self) -> None:
        s = State(current={"a": 1}, world={"b": 2})
        assert s.keys() == frozenset({"a", "b"})

    def test_layer_properties_return_copies(self) -> None:
        s = State(operational={"x": 1})
        op = s.operational
        op["x"] = 999
        assert s["x"] == 1  # original unchanged
