"""Tests for Scope and apply_scope."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.toolkit.flow._scope import Scope, apply_scope


class TestScope:
    def test_include_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="must not overlap"):
            Scope(include=frozenset({"a"}), exclude=frozenset({"a"}))

    def test_no_overlap_ok(self) -> None:
        Scope(include=frozenset({"a"}), exclude=frozenset({"b"}))


class TestApplyScope:
    def test_none_scope_passthrough(self) -> None:
        data = {"current": {"x": 1}, "operational": {}, "persistent": {}, "world": {}}
        snap = StateSnapshot.from_dict(data)
        result = apply_scope(snap, None)
        assert result["x"] == 1

    def test_include_filter(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {"a": 1, "b": 2},
                "operational": {"c": 3, "a": 10},
                "persistent": {},
                "world": {},
            }
        )
        scope = Scope(include=frozenset({"a"}))
        result = apply_scope(snap, scope)
        assert "a" in result
        assert "b" not in result
        assert "c" not in result
        # current layer's 'a' wins
        assert result["a"] == 1

    def test_exclude_filter(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {"a": 1, "b": 2},
                "operational": {},
                "persistent": {},
                "world": {},
            }
        )
        scope = Scope(exclude=frozenset({"b"}))
        result = apply_scope(snap, scope)
        assert "a" in result
        assert "b" not in result

    def test_per_layer_filtering(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {"x": 1},
                "operational": {"x": 2, "y": 3},
                "persistent": {"x": 4, "z": 5},
                "world": {},
            }
        )
        scope = Scope(include=frozenset({"x"}))
        result = apply_scope(snap, scope)
        # x exists in current, operational, persistent — all filtered independently
        assert result["x"] == 1  # current wins
        assert "y" not in result
        assert "z" not in result
        # Check that operational layer still has x
        assert "x" in result.operational

    def test_transform(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {"count": 5},
                "operational": {},
                "persistent": {},
                "world": {},
            }
        )
        scope = Scope(transform={"count": lambda v: v * 2})
        result = apply_scope(snap, scope)
        assert result["count"] == 10

    def test_transform_per_layer(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {"x": 1},
                "operational": {"x": 2},
                "persistent": {},
                "world": {},
            }
        )
        scope = Scope(transform={"x": lambda v: v + 100})
        result = apply_scope(snap, scope)
        assert result.current["x"] == 101
        assert result.operational["x"] == 102

    def test_enrich(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {},
                "operational": {"a": 1, "b": 2},
                "persistent": {},
                "world": {},
            }
        )
        scope = Scope(enrich={"total": lambda s: s.get("a", 0) + s.get("b", 0)})
        result = apply_scope(snap, scope)
        assert result["total"] == 3
        # Enriched values go into current layer
        assert "total" in result.current

    def test_combined_include_transform_enrich(self) -> None:
        snap = StateSnapshot.from_dict(
            {
                "current": {},
                "operational": {"price": 100, "tax": 10, "secret": "hidden"},
                "persistent": {},
                "world": {},
            }
        )
        scope = Scope(
            include=frozenset({"price", "tax"}),
            transform={"price": lambda v: v * 1.1},
            enrich={"label": lambda s: "processed"},
        )
        result = apply_scope(snap, scope)
        assert "secret" not in result
        assert result["price"] == pytest.approx(110.0)
        assert result["tax"] == 10
        assert result["label"] == "processed"
