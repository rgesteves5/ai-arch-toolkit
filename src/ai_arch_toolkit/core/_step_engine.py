"""Step execution engine — single-step execution with policy enforcement."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from contextlib import nullcontext

from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._metering._scope import current_meter, open_span
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

    # A per-step ``max_cost`` cap needs this step's OWN metered spend (its LLM/tool charges), not
    # the cumulative run total. Run it in a dedicated meter span and project that span — the span
    # contextvar isolates it across concurrent DAG steps. Only opened when a cap is set, so the
    # common (uncapped) path keeps its overhead and span tree unchanged.
    meter = current_meter()
    track_cost = policy.max_cost is not None and meter is not None
    span_cm = open_span("step") if track_cost else nullcontext(None)
    with span_cm as span_id:
        result, attempts = await _run_attempts(step, snapshot, policy, t0, decisions, span_id)

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


async def _run_attempts(
    step: Step,
    snapshot: StateSnapshot,
    policy: Policy,
    t0: float,
    decisions: list[PolicyDecision],
    span_id: str | None,
) -> tuple[Result, int]:
    """The retry/timeout/confidence/cost loop. ``span_id`` scopes this step's metered cost."""
    max_attempts = policy.retry.max_retries + 1
    attempts = 0
    result: Result | None = None

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
            # Check cost limit — the step's manual annotation plus its metered span spend. Fail
            # CLOSED if that span incurred an unbounded (unknown) cost: an unpriced model or a
            # server tool can't be bounded, so a max_cost step must not pass (decision #4).
            if policy.max_cost is not None:
                span_cost, cost_unknown = _span_spend(span_id)
                effective_cost = result.cost + span_cost
                if cost_unknown or effective_cost > policy.max_cost:
                    decisions.append("cost_exceeded")
                    detail = (
                        "a call could not be priced (fail-closed)"
                        if cost_unknown
                        else f"cost {effective_cost}"
                    )
                    result = Result(
                        error=f"Cost exceeded limit {policy.max_cost}: {detail}",
                        cost=effective_cost,
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

    return result, attempts


def _span_spend(span_id: str | None) -> tuple[float, bool]:
    """This step's metered spend, projected from its span: ``(known_usd_cost, has_unknown_cost)``.

    ``(0.0, False)`` when unmetered/uncapped. ``has_unknown_cost`` flags spend the cap can't bound,
    so the caller fails closed rather than treat it as $0: either a settled-but-unpriced call
    (unknown cost), or a metered op still IN FLIGHT when the step returned (e.g. a stream opened
    but never drained) — its cost isn't committed to the span yet, so the span total is unbounded.
    """
    if span_id is None:
        return 0.0, False
    meter = current_meter()
    if meter is None:
        return 0.0, False
    snap = meter.for_span(span_id)
    unbounded = snap.unknown_cost_count > 0 or meter.has_live_ops(span_id)
    return snap.cost.to_float(), unbounded


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
