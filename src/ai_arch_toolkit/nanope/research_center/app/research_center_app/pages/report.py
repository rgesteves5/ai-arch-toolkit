"""Report page — final pipeline output and cost summary."""

from __future__ import annotations

import reflex as rx

from research_center_app.components.layout import page_shell, project_sidebar
from research_center_app.state import PipelineState


def _cost_summary() -> rx.Component:
    """Cost summary card with per-agent breakdown."""
    return rx.card(
        rx.vstack(
            rx.heading("Cost Summary", size="4"),
            rx.hstack(
                rx.vstack(
                    rx.text("Total Cost", size="1", color="var(--gray-a11)"),
                    rx.text("$", PipelineState.total_spent.to(str), size="3", weight="bold"),  # type: ignore[union-attr]
                    spacing="1",
                ),
                rx.vstack(
                    rx.text("Cycles", size="1", color="var(--gray-a11)"),
                    rx.text(PipelineState.current_cycle.to(str), size="3", weight="bold"),  # type: ignore[union-attr]
                    spacing="1",
                ),
                spacing="6",
            ),
            rx.cond(
                PipelineState.phase_costs.length() > 0,  # type: ignore[union-attr]
                rx.box(
                    rx.text("Per-Agent Breakdown", size="2", weight="medium"),
                    rx.foreach(
                        PipelineState.phase_costs,
                        lambda pc: rx.hstack(
                            rx.text("Cycle ", pc["cycle"], size="1"),
                            rx.badge(pc["agent"], size="1"),
                            rx.text("$", pc["cost"], size="1"),
                            spacing="2",
                        ),
                    ),
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="3",
        ),
        width="100%",
    )


@rx.page(  # type: ignore[misc]
    route="/project/[id]/report",
    on_load=PipelineState.load_project,
)
def report_page() -> rx.Component:
    """Report view page with cost summary and markdown report."""
    return page_shell(
        rx.vstack(
            # Header
            rx.hstack(
                rx.heading("Report", size="5"),
                rx.spacer(),
                rx.cond(
                    PipelineState.report != "",
                    rx.link(
                        rx.button("Download .md", variant="soft", size="2"),
                        href=rx.cond(  # type: ignore[arg-type]
                            PipelineState.report != "",
                            f"data:text/markdown;charset=utf-8,{PipelineState.report}",
                            "#",
                        ),
                        download=f"{PipelineState.topic}.md",
                    ),
                    rx.fragment(),
                ),
                width="100%",
                align="center",
            ),
            rx.separator(),

            # Tab nav
            rx.hstack(
                rx.link(
                    rx.button("Monitor", variant="soft", size="2"),
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
                    rx.button("Report", variant="solid", size="2"),
                    href=f"/project/{PipelineState.project_id}/report",
                ),
                spacing="2",
            ),
            rx.separator(),

            # Cost summary
            _cost_summary(),

            # Report body
            rx.cond(
                PipelineState.report != "",
                rx.box(
                    rx.markdown(PipelineState.report),
                    width="100%",
                    padding="16px",
                    border="1px solid var(--gray-a5)",
                    border_radius="8px",
                ),
                rx.center(
                    rx.text(
                        "No report yet. Run the pipeline first.",
                        color="var(--gray-a11)",
                    ),
                    padding_y="64px",
                ),
            ),
            spacing="4",
            width="100%",
        ),
        with_sidebar=project_sidebar(PipelineState.project_id),
    )
