"""Tests for the step execution engine."""

from __future__ import annotations

import asyncio

import pytest

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


class TestAdmissionDeniedIsTerminal:
    """A budget/admission denial must propagate out of the step engine, never a retried error."""

    async def test_admission_denied_propagates(self) -> None:
        import pytest

        from ai_arch_toolkit.core._metering._admission import AdmissionDenied
        from ai_arch_toolkit.core._state import State

        async def fn(snap: StateSnapshot) -> Result:
            raise AdmissionDenied(dimension="cost")

        with pytest.raises(AdmissionDenied):
            await execute_step(Step(name="denied", fn=fn), State().snapshot())

    async def test_admission_denied_is_not_retried(self) -> None:
        import pytest

        from ai_arch_toolkit.core._metering._admission import AdmissionDenied
        from ai_arch_toolkit.core._state import State

        calls = 0

        async def fn(snap: StateSnapshot) -> Result:
            nonlocal calls
            calls += 1
            raise AdmissionDenied(dimension="cost")

        step = Step(name="denied", fn=fn, policy=Policy(retry=RetryConfig(max_retries=3)))
        with pytest.raises(AdmissionDenied):
            await execute_step(step, State().snapshot())
        assert calls == 1  # terminal — the retry policy did not apply

    async def test_a_normal_exception_still_becomes_an_error_result(self) -> None:
        from ai_arch_toolkit.core._state import State

        async def fn(snap: StateSnapshot) -> Result:
            raise ValueError("boom")

        result, _ = await execute_step(Step(name="err", fn=fn), State().snapshot())
        assert result.is_error and result.error is not None and "boom" in result.error


def test_the_step_backoff_stays_finite_for_very_late_attempts() -> None:
    from ai_arch_toolkit.core._retry import RetryConfig
    from ai_arch_toolkit.core._step_engine import _compute_backoff

    policy = Policy(retry=RetryConfig(base_delay=1.0, max_delay=3.0))

    assert _compute_backoff(5_000, policy) <= 3.0


# --- Every decision path of the attempt loop, pinned before and after its rewrite ----------------


def _scripted(*outcomes: Result | Exception | float) -> Step:
    """A step that plays one outcome a call: a Result, a raised exception, or a sleep (seconds)."""
    queue = list(outcomes)

    async def fn(snap: StateSnapshot) -> Result:
        outcome = queue.pop(0)
        if isinstance(outcome, float):
            await asyncio.sleep(outcome)
            return Result(value="slept")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return Step(name="s", fn=fn)


def _fallback_step(value: str = "fb", *, sleep: float = 0.0, fails: bool = False) -> Step:
    async def fn(snap: StateSnapshot) -> Result:
        if sleep:
            await asyncio.sleep(sleep)
        if fails:
            raise RuntimeError("fallback broke")
        return Result(value=value)

    return Step(name="fallback", fn=fn, policy=Policy(timeout=0.01) if sleep else None)


_OK = Result(value="ok")
_LOW = Result(value="low", confidence=0.1)
_BAD = Result(error="bad")
_ONE_RETRY = RetryConfig(max_retries=1, base_delay=1e-9, max_delay=1e-9)

_PATHS = {
    "ok": ((_OK,), Policy(), ("ok", None), (), 1),
    "error then ok": ((_BAD, _OK), Policy(retry=_ONE_RETRY), ("ok", None), ("retry",), 2),
    "raise then ok": (
        (RuntimeError("x"), _OK),
        Policy(retry=_ONE_RETRY),
        ("ok", None),
        ("retry",),
        2,
    ),
    "exhausted halt": (
        (_BAD, _BAD),
        Policy(retry=_ONE_RETRY),
        (None, "bad"),
        ("retry", "halt"),
        2,
    ),
    "exhausted continue": (
        (_BAD,),
        Policy(on_exhausted="continue"),
        (None, "bad"),
        (),
        1,
    ),
    "exhausted fallback": (
        (_BAD,),
        Policy(on_exhausted="fallback", fallback=_fallback_step()),
        ("fb", None),
        ("fallback",),
        1,
    ),
    "exhausted fallback missing": (
        (_BAD,),
        Policy(on_exhausted="fallback"),
        (None, "bad"),
        (),
        1,
    ),
    "timeout halt, never retried": (
        (1.0, _OK),
        Policy(timeout=0.01, retry=_ONE_RETRY),
        (None, "Step timed out"),
        ("timeout",),
        1,
    ),
    "timeout fallback": (
        (1.0,),
        Policy(timeout=0.01, on_timeout="fallback", fallback=_fallback_step()),
        ("fb", None),
        ("timeout", "fallback"),
        1,
    ),
    "timeout fallback missing": (
        (1.0,),
        Policy(timeout=0.01, on_timeout="fallback"),
        (None, "Step timed out"),
        ("timeout",),
        1,
    ),
    "low confidence retried": (
        (_LOW, _OK),
        Policy(confidence_threshold=0.5, retry=_ONE_RETRY),
        ("ok", None),
        ("low_confidence", "retry"),
        2,
    ),
    "low confidence on the last attempt": (
        (_LOW,),
        Policy(confidence_threshold=0.5),
        ("low", None),
        ("low_confidence",),
        1,
    ),
    "low confidence fallback": (
        (_LOW,),
        Policy(confidence_threshold=0.5, on_low_confidence="fallback", fallback=_fallback_step()),
        ("fb", None),
        ("low_confidence", "fallback"),
        1,
    ),
    "low confidence fallback missing": (
        (_LOW,),
        Policy(confidence_threshold=0.5, on_low_confidence="fallback"),
        ("low", None),
        ("low_confidence",),
        1,
    ),
    "low confidence escalated": (
        (_LOW,),
        Policy(confidence_threshold=0.5, on_low_confidence="escalate"),
        ("low", None),
        ("low_confidence", "escalate"),
        1,
    ),
    "escalation skips the cost check": (
        (Result(value="low", confidence=0.1, cost=5.0),),
        Policy(confidence_threshold=0.5, on_low_confidence="escalate", max_cost=1.0),
        ("low", None),
        ("low_confidence", "escalate"),
        1,
    ),
    "cost exceeded": (
        (Result(value="dear", cost=5.0),),
        Policy(max_cost=1.0),
        (None, "Cost exceeded limit 1.0: cost 5.0"),
        ("cost_exceeded",),
        1,
    ),
    "fallback times out": (
        (_BAD,),
        Policy(on_exhausted="fallback", fallback=_fallback_step(sleep=1.0)),
        (None, "Fallback timed out"),
        ("fallback",),
        1,
    ),
    "fallback raises": (
        (_BAD,),
        Policy(on_exhausted="fallback", fallback=_fallback_step(fails=True)),
        (None, "Fallback failed: fallback broke"),
        ("fallback",),
        1,
    ),
}


@pytest.mark.parametrize("path", sorted(_PATHS))
async def test_every_decision_path_of_the_attempt_loop(path: str) -> None:
    outcomes, policy, (value, error), decisions, attempts = _PATHS[path]

    result, trace = await execute_step(_scripted(*outcomes), StateSnapshot(), policy=policy)

    assert (result.value, result.error) == (value, error)
    assert trace.policy_decisions == decisions
    assert trace.attempts == attempts


async def test_a_step_without_fallback_uses_the_policys_and_its_own_wins() -> None:
    own = Step(name="s", fn=_scripted(_BAD).fn, fallback=_fallback_step("own"))
    policy = Policy(on_exhausted="fallback", fallback=_fallback_step("policy"))

    result, _ = await execute_step(own, StateSnapshot(), policy=policy)

    assert result.value == "own"
