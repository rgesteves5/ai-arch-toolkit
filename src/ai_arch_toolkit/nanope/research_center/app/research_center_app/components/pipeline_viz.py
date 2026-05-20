"""Pipeline visualization components: agent steps, budget meter, event log."""

from __future__ import annotations

import reflex as rx

from research_center_app.state import PipelineState

_AGENTS = ["manager", "researcher", "linker", "writer"]


def _agent_badge(agent_name: str) -> rx.Component:
    """Single agent badge, highlighted when active."""
    return rx.cond(
        PipelineState.current_agent == agent_name,
        rx.badge(agent_name.capitalize(), variant="solid", size="2", color_scheme="blue"),
        rx.badge(agent_name.capitalize(), variant="surface", size="2"),
    )


def agent_step_indicator() -> rx.Component:
    """Horizontal row of agent badges showing which agent is active."""
    return rx.hstack(
        *[_agent_badge(a) for a in _AGENTS],
        spacing="2",
        align="center",
    )


def budget_meter() -> rx.Component:
    """Progress bar showing budget consumption."""
    pct = rx.cond(
        PipelineState.budget > 0,
        ((PipelineState.total_spent / PipelineState.budget) * 100).to(int),  # type: ignore[union-attr]
        rx.Var.create(0),
    ).to(int)  # type: ignore[union-attr]
    return rx.vstack(
        rx.hstack(
            rx.text("Budget:", size="2", weight="medium"),
            rx.text(
                "$",
                PipelineState.total_spent.to(str),  # type: ignore[union-attr]
                " / $",
                PipelineState.budget.to(str),  # type: ignore[union-attr]
                size="2",
            ),
            spacing="2",
        ),
        rx.progress(
            value=pct,
            width="100%",
        ),
        width="100%",
        spacing="2",
    )


def _event_item(event: dict) -> rx.Component:  # noqa: ARG001
    """Render a single pipeline event in the log."""
    return rx.hstack(
        rx.badge(event["type"], variant="outline", size="1"),
        rx.cond(
            event["agent"] != "",
            rx.badge(event["agent"], variant="surface", size="1", color_scheme="blue"),
            rx.fragment(),
        ),
        rx.text(event["message"], size="1", color="var(--gray-a11)"),
        spacing="2",
        width="100%",
        padding_y="4px",
    )


def event_log() -> rx.Component:
    """Scrollable event log."""
    return rx.scroll_area(
        rx.vstack(
            rx.foreach(PipelineState.events, _event_item),
            spacing="1",
            width="100%",
        ),
        max_height="400px",
        width="100%",
        border="1px solid var(--gray-a5)",
        border_radius="8px",
        padding="12px",
    )


def _directive_accordion_item(directives: dict) -> rx.Component:  # noqa: ARG001
    """Render a single cycle's directives in an accordion."""
    return rx.accordion.item(
        header=rx.text("Decision: ", directives["decision"], weight="medium"),
        content=rx.vstack(
            rx.cond(
                directives["researcher_plan"] != "",
                rx.box(
                    rx.text("Researcher Plan", size="1", weight="bold"),
                    rx.text(directives["researcher_plan"], size="1"),
                    padding="8px",
                    background="var(--gray-a2)",
                    border_radius="4px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.cond(
                directives["linker_plan"] != "",
                rx.box(
                    rx.text("Linker Plan", size="1", weight="bold"),
                    rx.text(directives["linker_plan"], size="1"),
                    padding="8px",
                    background="var(--gray-a2)",
                    border_radius="4px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.cond(
                directives["writer_plan"] != "",
                rx.box(
                    rx.text("Writer Plan", size="1", weight="bold"),
                    rx.text(directives["writer_plan"], size="1"),
                    padding="8px",
                    background="var(--gray-a2)",
                    border_radius="4px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="2",
            width="100%",
        ),
    )


def directives_accordion() -> rx.Component:
    """Accordion showing manager directives per cycle."""
    return rx.accordion.root(
        rx.foreach(PipelineState.directives_by_cycle, _directive_accordion_item),
        type="multiple",
        width="100%",
    )
