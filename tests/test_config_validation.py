"""Unit tests for agent-specific config validation."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.agents._lats import LATSConfig
from ai_arch_toolkit.toolkit.agents._llm_compiler import LLMCompilerConfig
from ai_arch_toolkit.toolkit.agents._plan_execute import PlanExecuteConfig
from ai_arch_toolkit.toolkit.agents._reflexion import ReflexionConfig
from ai_arch_toolkit.toolkit.agents._tot import ToTConfig

# ---------------------------------------------------------------------------
# ReflexionConfig
# ---------------------------------------------------------------------------


class TestReflexionConfig:
    def test_negative_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            ReflexionConfig(max_retries=-1, evaluator=lambda t, a: 0.5)

    def test_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ReflexionConfig(threshold=1.5, evaluator=lambda t, a: 0.5)

    def test_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ReflexionConfig(threshold=-0.1, evaluator=lambda t, a: 0.5)


# ---------------------------------------------------------------------------
# ToTConfig
# ---------------------------------------------------------------------------


class TestToTConfig:
    def test_zero_candidates(self) -> None:
        with pytest.raises(ValueError, match="n_candidates"):
            ToTConfig(n_candidates=0)

    def test_negative_max_depth(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            ToTConfig(max_depth=-1)

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            ToTConfig(strategy="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LATSConfig
# ---------------------------------------------------------------------------


class TestLATSConfig:
    def test_zero_candidates(self) -> None:
        with pytest.raises(ValueError, match="n_candidates"):
            LATSConfig(n_candidates=0)

    def test_negative_max_rollouts(self) -> None:
        with pytest.raises(ValueError, match="max_rollouts"):
            LATSConfig(max_rollouts=-1)

    def test_negative_exploration_weight(self) -> None:
        with pytest.raises(ValueError, match="exploration_weight"):
            LATSConfig(exploration_weight=-1.0)


# ---------------------------------------------------------------------------
# PlanExecuteConfig
# ---------------------------------------------------------------------------


class TestPlanExecuteConfig:
    def test_negative_max_replans(self) -> None:
        with pytest.raises(ValueError, match="max_replans"):
            PlanExecuteConfig(max_replans=-1)


# ---------------------------------------------------------------------------
# LLMCompilerConfig
# ---------------------------------------------------------------------------


class TestLLMCompilerConfig:
    def test_negative_max_replans(self) -> None:
        with pytest.raises(ValueError, match="max_replans"):
            LLMCompilerConfig(max_replans=-1)
