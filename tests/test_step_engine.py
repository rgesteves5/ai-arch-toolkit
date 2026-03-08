"""Tests for the step execution engine."""

from __future__ import annotations

import asyncio

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._step_engine import execute_step


class TestExecuteStep:
    async def test_simple_step(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(value=snap["x"] + 1, artifacts={"y": 2})

        step = Step(name="add", fn=fn)
        data = {"current": {"x": 10}, "operational": {}, "persistent": {}, "world": {}}
        snap = StateSnapshot.from_dict(data)
        result, trace = await execute_step(step, snap)
        assert result.value == 11
        assert result.is_ok
        assert trace.name == "add"
        assert trace.attempts == 1

    async def test_step_error(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            raise ValueError("boom")

        step = Step(name="fail", fn=fn)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.is_error
        assert "boom" in result.error
        assert trace.error is not None

    async def test_retry_on_error(self) -> None:
        call_count = 0

        async def fn(snap: StateSnapshot) -> Result:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Result(error="transient")
            return Result(value="ok")

        policy = Policy(retry=RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.05))
        step = Step(name="retry_test", fn=fn, policy=policy)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.is_ok
        assert result.value == "ok"
        assert trace.attempts == 3
        assert "retry" in trace.policy_decisions

    async def test_timeout(self) -> None:
        async def slow(snap: StateSnapshot) -> Result:
            await asyncio.sleep(10)
            return Result(value="late")

        policy = Policy(timeout=0.05)
        step = Step(name="slow", fn=slow, policy=policy)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.is_error
        assert "timed out" in result.error
        assert "timeout" in trace.policy_decisions

    async def test_timeout_with_fallback(self) -> None:
        async def slow(snap: StateSnapshot) -> Result:
            await asyncio.sleep(10)
            return Result(value="late")

        async def backup(snap: StateSnapshot) -> Result:
            return Result(value="fallback_value")

        fb = Step(name="backup", fn=backup)
        policy = Policy(timeout=0.05, on_timeout="fallback")
        step = Step(name="slow", fn=slow, policy=policy, fallback=fb)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.is_ok
        assert result.value == "fallback_value"
        assert "fallback" in trace.policy_decisions

    async def test_confidence_threshold_retry(self) -> None:
        call_count = 0

        async def fn(snap: StateSnapshot) -> Result:
            nonlocal call_count
            call_count += 1
            conf = 0.5 if call_count == 1 else 0.9
            return Result(value="answer", confidence=conf)

        policy = Policy(
            confidence_threshold=0.7,
            on_low_confidence="retry",
            retry=RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05),
        )
        step = Step(name="conf", fn=fn, policy=policy)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.confidence == 0.9
        assert "low_confidence" in trace.policy_decisions

    async def test_confidence_threshold_escalate(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(value="answer", confidence=0.3)

        policy = Policy(confidence_threshold=0.7, on_low_confidence="escalate")
        step = Step(name="esc", fn=fn, policy=policy)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert "escalate" in trace.policy_decisions
        assert result.value == "answer"

    async def test_cost_exceeded(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(value="expensive", cost=5.0)

        policy = Policy(max_cost=1.0)
        step = Step(name="costly", fn=fn, policy=policy)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.is_error
        assert "Cost" in result.error
        assert "cost_exceeded" in trace.policy_decisions

    async def test_exhausted_continue(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(error="always fails")

        policy = Policy(
            retry=RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.05),
            on_exhausted="continue",
        )
        step = Step(name="cont", fn=fn, policy=policy)
        snap = StateSnapshot()
        result, _trace = await execute_step(step, snap)
        assert result.is_error
        # on_exhausted="continue" means the flow can proceed

    async def test_exhausted_fallback(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(error="always fails")

        async def backup(snap: StateSnapshot) -> Result:
            return Result(value="recovered")

        fb = Step(name="backup", fn=backup)
        policy = Policy(
            retry=RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.05),
            on_exhausted="fallback",
            fallback=fb,
        )
        step = Step(name="fails", fn=fn, policy=policy)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.is_ok
        assert result.value == "recovered"
        assert "fallback" in trace.policy_decisions

    async def test_no_policy_defaults(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(value=42)

        step = Step(name="simple", fn=fn)
        snap = StateSnapshot()
        result, trace = await execute_step(step, snap)
        assert result.value == 42
        assert trace.attempts == 1
        assert trace.policy_decisions == ()
