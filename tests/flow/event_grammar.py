"""The grammar every flow event stream follows, checked against the run's own trace.

A stream opens with ``flow_start`` and closes with ``flow_end``. In between, every ``step_start``
is closed exactly once, by its own ``step_end``, also when the run stops early: the ends of the
steps a timeout or a budget denial cut come before the run's stop event (a ``timeout`` or
``budget_exceeded`` that names no step), and nothing comes after it. A step's ``llm_event``s and
policy decisions come between its start and its end. And the stream tells the trace's story: its
``step_end``, ``step_skipped`` and stop events match the trace entries one to one, in the same
order, and each ``step_end`` and ``step_skipped`` carries its entry itself (G-22).
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_arch_toolkit.core._trace import StepTrace, Trace
from ai_arch_toolkit.toolkit.flow import FlowEvent

_STOPS = {"timeout": "flow_timeout", "policy_decision": "budget_exceeded"}
_IN_STEP = {"retry", "timeout", "fallback", "policy_decision", "llm_event"}


def check_grammar(events: Sequence[FlowEvent], trace: Trace) -> None:
    """Fail with the first rule the stream breaks."""
    kinds = [event.type for event in events]
    assert kinds[:1] == ["flow_start"] and kinds[-1:] == ["flow_end"], kinds
    assert kinds.count("flow_start") == kinds.count("flow_end") == 1, kinds
    story, carried = steps_story(events[1:-1])
    told = [(step.name, "<skipped>" if step.skipped else step.error) for step in trace.steps]
    told = [(name, None if name in _STOPS.values() else error) for name, error in told]
    assert story == told, f"stream says {story}, trace says {told}"
    entries = [step for step in trace.steps if step.name not in _STOPS.values()]
    assert len(carried) == len(entries), (carried, entries)
    assert all(event is entry for event, entry in zip(carried, entries, strict=True)), (
        "a step's event does not carry its trace entry"
    )


def steps_story(
    events: Sequence[FlowEvent],
) -> tuple[list[tuple[str, str | None]], list[StepTrace | None]]:
    """What the stream says happened, as (name, error) in order, and the traces its step events
    carry; asserts the open/close rules."""
    running: set[str] = set()
    story: list[tuple[str, str | None]] = []
    carried: list[StepTrace | None] = []
    stopped = False
    for event in events:
        assert not stopped, f"{event.type} after the run stopped"
        name = event.step_name
        if not name:
            assert event.type in _STOPS, f"a run-level {event.type}"
            assert not running, f"still running at the stop: {sorted(running)}"
            stopped = True
            story.append((_STOPS[event.type], None))
        elif event.type == "step_start":
            assert name not in running, f"{name} started twice"
            running.add(name)
        elif event.type == "step_end":
            assert name in running, f"{name} ended without starting"
            running.discard(name)
            story.append((name, event.error))
            carried.append(event.step_trace)
        elif event.type == "step_skipped":
            assert name not in running, f"{name} skipped while running"
            story.append((name, "<skipped>"))
            carried.append(event.step_trace)
        else:
            assert event.type in _IN_STEP and name in running, f"{event.type} outside {name}"
    assert not running, f"never closed: {sorted(running)}"
    return story, carried
