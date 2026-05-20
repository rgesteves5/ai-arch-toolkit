"""Monitor page — pipeline control and live progress."""

from __future__ import annotations

import reflex as rx

from research_center_app.components.layout import page_shell, project_sidebar
from research_center_app.components.pipeline_viz import (
    agent_step_indicator,
    budget_meter,
    directives_accordion,
    event_log,
)
from research_center_app.state import PipelineState


def _tab_nav() -> rx.Component:
    return rx.hstack(
        rx.link(
            rx.button("Monitor", variant="solid", size="2"),
            href=f"/project/{PipelineState.project_id}",
        ),
        rx.link(
            rx.button("Wiki", variant="soft", size="2"),
            href=f"/project/{PipelineState.project_id}/wiki",
        ),
        rx.link(
            rx.button("Graph", variant="soft", size="2"),
            href=f"/project/{PipelineState.project_id}/graph",
        ),
        rx.link(
            rx.button("Report", variant="soft", size="2"),
            href=f"/project/{PipelineState.project_id}/report",
        ),
        spacing="2",
    )


def _pipeline_controls() -> rx.Component:
    """Pipeline control buttons: Run, Stop, Resume, Reset."""
    return rx.cond(
        PipelineState.is_running,
        # Running: show Stop button
        rx.hstack(
            rx.button(
                rx.spinner(size="1"),
                "Running...",
                disabled=True,
                size="3",
                flex="1",
            ),
            rx.button(
                "Stop",
                on_click=PipelineState.stop_pipeline,
                size="3",
                variant="soft",
                color_scheme="red",
            ),
            spacing="2",
            width="100%",
        ),
        # Not running
        rx.cond(
            PipelineState.is_stopped,
            # Stopped mid-run: show Resume + Reset
            rx.hstack(
                rx.button(
                    "Resume",
                    on_click=PipelineState.start_pipeline,
                    size="3",
                    variant="solid",
                    flex="1",
                ),
                rx.button(
                    "Reset",
                    on_click=PipelineState.reset_pipeline,
                    size="3",
                    variant="soft",
                    color_scheme="orange",
                ),
                spacing="2",
                width="100%",
            ),
            # Ready or completed: show Run
            rx.button(
                "Run Pipeline",
                on_click=PipelineState.start_pipeline,
                size="3",
                width="100%",
                variant="solid",
            ),
        ),
    )


@rx.page(route="/project/[id]", on_load=PipelineState.load_project)  # type: ignore[misc]
def monitor_page() -> rx.Component:
    """Pipeline monitor page with live progress updates."""
    return page_shell(
        rx.vstack(
            # Header
            rx.hstack(
                rx.heading(PipelineState.topic, size="5"),
                rx.spacer(),
                rx.cond(
                    PipelineState.is_running,
                    rx.badge("Running", variant="solid", color_scheme="blue", size="2"),
                    rx.cond(
                        PipelineState.is_stopped,
                        rx.badge("Stopped", variant="solid", color_scheme="orange", size="2"),
                        rx.cond(
                            PipelineState.report != "",
                            rx.badge("Complete", variant="solid", color_scheme="green", size="2"),
                            rx.badge("Ready", variant="surface", size="2"),
                        ),
                    ),
                ),
                width="100%",
                align="center",
            ),
            rx.separator(),
            _tab_nav(),
            rx.separator(),

            # Budget meter
            budget_meter(),

            # Agent step indicator
            rx.text("Agent Progress", size="2", weight="bold"),
            agent_step_indicator(),

            # Pipeline controls
            _pipeline_controls(),

            # Error display
            rx.cond(
                PipelineState.error != "",
                rx.callout(
                    PipelineState.error,
                    icon="triangle_alert",
                    color_scheme="red",
                    width="100%",
                ),
                rx.fragment(),
            ),

            # Directives accordion
            rx.cond(
                PipelineState.directives_by_cycle.length() > 0,  # type: ignore[union-attr]
                rx.box(
                    rx.text("Manager Directives", size="3", weight="bold"),
                    directives_accordion(),
                    width="100%",
                ),
                rx.fragment(),
            ),

            # Event log
            rx.cond(
                PipelineState.events.length() > 0,  # type: ignore[union-attr]
                rx.box(
                    rx.text("Event Log", size="3", weight="bold"),
                    event_log(),
                    width="100%",
                ),
                rx.fragment(),
            ),

            spacing="4",
            width="100%",
        ),
        with_sidebar=project_sidebar(PipelineState.project_id),
    )
