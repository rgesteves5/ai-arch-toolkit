"""Seeded state-machine sequences conserve counts, holds, usage and audit projection."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from ai_arch_toolkit.core import (
    Cost,
    MeterScope,
    MeterSnapshot,
    Money,
    OperationRequest,
    RunConfig,
    Usage,
)


def replay(events) -> MeterSnapshot:
    llm = tools = inputs = outputs = reads = writes = unknown = 0
    cost = Money.zero()
    uncertain_cost = Money.zero()
    uncertain_count = 0
    for event in events:
        if event.status != "aborted":
            llm += int(event.kind == "llm")
            tools += int(event.kind == "tool")
        inputs += event.usage.input_tokens
        outputs += event.usage.output_tokens
        reads += event.usage.cache_read_tokens
        writes += event.usage.cache_write_tokens
        if event.cost.kind == "unknown":
            if event.cost.at_most is None:
                unknown += 1
            else:
                uncertain_cost += event.cost.at_most
                uncertain_count += 1
        else:
            cost += event.cost.amount
    return MeterSnapshot(
        llm_calls=llm,
        tool_calls=tools,
        input_tokens=inputs,
        output_tokens=outputs,
        cache_read_tokens=reads,
        cache_write_tokens=writes,
        cost=cost,
        unknown_cost_count=unknown,
        uncertain_cost=uncertain_cost,
        uncertain_cost_count=uncertain_count,
    )


@pytest.mark.parametrize("seed", [7, 41, 2026, 9317])
def test_terminal_sequences_conserve_meter_and_replay(seed: int) -> None:
    rng = random.Random(seed)
    scope = MeterScope(RunConfig(retain_meter_events=True))
    with scope:
        for _ in range(200):
            kind = rng.choice(("llm", "tool", "custom"))
            op = scope.open(
                OperationRequest(
                    kind=kind, parent_span_id="run", count=0 if kind == "custom" else 1
                )
            )
            action = rng.choice(("abort", "pending", "settle", "fail", "incomplete"))
            if action == "abort":
                op.abort()
            elif action != "pending":
                op.mark_started()
                if action == "settle":
                    usage = Usage(input_tokens=rng.randrange(50), output_tokens=rng.randrange(20))
                    amount = Money.from_usd(rng.randrange(10) / 1000)
                    cost = rng.choice(
                        (
                            Cost.known(amount),
                            Cost.unknown("unpriced"),
                            Cost.unknown("bounded", at_most=amount),
                        )
                    )
                    op.settle(usage=usage, cost=cost)
                    op.settle(usage=usage, cost=cost)  # terminal replay cannot double-charge
                elif action == "fail":
                    op.fail("indeterminate")
                    op.fail("indeterminate")
    snapshot = replace(scope.snapshot(), elapsed_s=0.0)
    assert not scope.has_live_ops("run")
    assert snapshot == replay(scope.events())
    assert len(scope.events()) == 200
    assert [event.seq for event in scope.events()] == list(range(1, 201))
