"""The grammar every flow event stream follows, checked against the run's own trace.

A stream opens with ``flow_start`` and closes with ``flow_end``. In between, every ``step_start``
is closed exactly once: by its ``step_end``, or, if the step was still running when the run
stopped, by the run's stop event (a ``timeout`` or ``budget_exceeded`` that names no step). Nothing
of a step comes after its end, nor of any step after the stop. And the stream tells the trace's
story: its ``step_end``, ``step_skipped`` and stop events match the trace entries one to one, in
the same order.
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_arch_toolkit.core._trace import Trace
from ai_arch_toolkit.toolkit.flow import FlowEvent

_STOPS = {"timeout": "flow_timeout", "policy_decision": "budget_exceeded"}
_IN_STEP = {"retry", "timeout", "fallback", "policy_decision"}


def check_grammar(events: Sequence[FlowEvent], trace: Trace) -> None:
    """Fail with the first rule the stream breaks."""
    kinds = [event.type for event in events]
    assert kinds[:1] == ["flow_start"] and kinds[-1:] == ["flow_end"], kinds
    assert kinds.count("flow_start") == kinds.count("flow_end") == 1, kinds
    story = _story(events[1:-1])
    told = [(step.name, "<skipped>" if step.skipped else step.error) for step in trace.steps]
    told = [(name, None if name in _STOPS.values() else error) for name, error in told]
    assert story == told, f"stream says {story}, trace says {told}"


def _story(events: Sequence[FlowEvent]) -> list[tuple[str, str | None]]:
    """What the stream says happened, as (name, error) in order; asserts the open/close rules."""
    running: set[str] = set()
    story: list[tuple[str, str | None]] = []
    stopped = False
    for event in events:
        assert not stopped, f"{event.type} after the run stopped"
        name = event.step_name
        if not name:
            assert event.type in _STOPS, f"a run-level {event.type}"
            stopped = True
            running.clear()  # the stop closes every step still running
            story.append((_STOPS[event.type], None))
        elif event.type == "step_start":
            assert name not in running, f"{name} started twice"
            running.add(name)
        elif event.type == "step_end":
            assert name in running, f"{name} ended without starting"
            running.discard(name)
            story.append((name, event.error))
        elif event.type == "step_skipped":
            assert name not in running, f"{name} skipped while running"
            story.append((name, "<skipped>"))
        else:
            assert event.type in _IN_STEP and name in running, f"{event.type} outside {name}"
    assert not running, f"never closed: {sorted(running)}"
    return story
