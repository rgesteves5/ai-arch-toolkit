"""Step execution engine — single-step execution with policy enforcement."""

from __future__ import annotations

import asyncio
import logging
import random
import time

from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._trace import PolicyDecision, StepTrace

logger = logging.getLogger(__name__)


async def execute_step(step: Step, snapshot: StateSnapshot) -> tuple[Result, StepTrace]:
    """Execute a single Step against a (possibly scoped) snapshot.

    The caller is responsible for applying any Scope before calling this.
    Returns the Result and a StepTrace recording what happened.
    """
    policy = step.policy or Policy()
    t0 = time.monotonic()
    decisions: list[PolicyDecision] = []
    attempts = 0
    result: Result | None = None

    max_attempts = policy.retry.max_retries + 1

    for attempt in range(max_attempts):
        attempts = attempt + 1
        try:
            if policy.timeout is not None:
                result = await asyncio.wait_for(step.fn(snapshot), timeout=policy.timeout)
            else:
                result = await step.fn(snapshot)
        except TimeoutError:
            decisions.append("timeout")
            if policy.on_timeout == "fallback":
                fb = step.fallback or (policy.fallback if policy else None)
                if fb is not None:
                    decisions.append("fallback")
                    result = await _run_fallback(fb, snapshot, t0)
                    break
            result = Result(error="Step timed out", duration=time.monotonic() - t0)
            break
        except AdmissionDenied:
            raise  # budget/admission denial is terminal — never retried, never an error Result
        except Exception as exc:
            result = Result(error=str(exc), duration=time.monotonic() - t0)

        if result is not None and result.is_ok:
            # Check confidence threshold
            if (
                policy.confidence_threshold is not None
                and result.confidence is not None
                and result.confidence < policy.confidence_threshold
            ):
                decisions.append("low_confidence")
                if policy.on_low_confidence == "retry" and attempt < max_attempts - 1:
                    decisions.append("retry")
                    await asyncio.sleep(_compute_backoff(attempt, policy))
                    result = None
                    continue
                elif policy.on_low_confidence == "fallback":
                    fb = step.fallback or policy.fallback
                    if fb is not None:
                        decisions.append("fallback")
                        result = await _run_fallback(fb, snapshot, t0)
                        break
                elif policy.on_low_confidence == "escalate":
                    decisions.append("escalate")
                    break
            # Check cost limit
            if policy.max_cost is not None and result.cost > policy.max_cost:
                decisions.append("cost_exceeded")
                result = Result(
                    error=f"Cost {result.cost} exceeded limit {policy.max_cost}",
                    cost=result.cost,
                    duration=time.monotonic() - t0,
                )
                break
            break

        # Error path — retry or exhaust
        if result is not None and result.is_error:
            if attempt < max_attempts - 1:
                decisions.append("retry")
                await asyncio.sleep(_compute_backoff(attempt, policy))
                result = None
                continue
            # Exhausted
            if policy.on_exhausted == "fallback":
                fb = step.fallback or policy.fallback
                if fb is not None:
                    decisions.append("fallback")
                    result = await _run_fallback(fb, snapshot, t0)
                    break
            elif policy.on_exhausted == "continue":
                break
            else:
                decisions.append("halt")
                break

    if result is None:
        result = Result(error="No result produced", duration=time.monotonic() - t0)

    elapsed = time.monotonic() - t0

    trace = StepTrace(
        name=step.name,
        input_state=snapshot.to_dict(),
        output_result=result.to_dict(),
        duration=elapsed,
        cost=result.cost,
        confidence=result.confidence,
        usage=result.usage,
        attempts=attempts,
        policy_decisions=tuple(decisions),
        error=result.error,
        started_at=t0,
    )

    return result, trace


async def _run_fallback(step: Step, snapshot: StateSnapshot, t0: float) -> Result:
    """Execute a fallback step, respecting its own policy timeout."""
    try:
        timeout = step.policy.timeout if step.policy else None
        if timeout is not None:
            result = await asyncio.wait_for(step.fn(snapshot), timeout=timeout)
        else:
            result = await step.fn(snapshot)
        return Result(
            value=result.value,
            artifacts=result.artifacts,
            usage=result.usage,
            cost=result.cost,
            confidence=result.confidence,
            error=result.error,
            duration=time.monotonic() - t0,
        )
    except TimeoutError:
        return Result(error="Fallback timed out", duration=time.monotonic() - t0)
    except AdmissionDenied:
        raise  # terminal, even inside a fallback
    except Exception as exc:
        return Result(error=f"Fallback failed: {exc}", duration=time.monotonic() - t0)


def _compute_backoff(attempt: int, policy: Policy) -> float:
    """Exponential backoff with jitter."""
    config = policy.retry
    delay = config.base_delay * (2**attempt)
    jitter = random.uniform(0, delay * 0.25)
    return min(delay + jitter, config.max_delay)
