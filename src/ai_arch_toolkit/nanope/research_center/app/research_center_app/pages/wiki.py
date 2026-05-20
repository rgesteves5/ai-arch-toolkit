"""Wiki page — knowledge graph browser."""

from __future__ import annotations

import reflex as rx

from research_center_app.components.layout import page_shell, project_sidebar
from research_center_app.components.wiki_browser import wiki_node_detail, wiki_search_panel
from research_center_app.state import PipelineState, WikiState


@rx.page(  # type: ignore[misc]
    route="/project/[id]/wiki",
    on_load=[
        PipelineState.load_project,
        WikiState.load_wiki_data,
    ],
)
def wiki_page() -> rx.Component:
    """Wiki browser page with search, categories, and node detail."""
    return page_shell(
        rx.vstack(
            # Header
            rx.hstack(
                rx.heading("Wiki", size="5"),
                rx.spacer(),
                rx.text(
                    f"{WikiState.node_count} nodes",
                    size="2",
                    color="var(--gray-a11)",
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
                    rx.button("Wiki", variant="solid", size="2"),
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
            ),
            rx.separator(),
            # Two-column layout
            rx.hstack(
                rx.box(
                    wiki_search_panel(),
                    width="40%",
                    min_width="280px",
                ),
                rx.separator(orientation="vertical"),
                rx.box(
                    wiki_node_detail(),
                    width="60%",
                ),
                spacing="4",
                width="100%",
                align="start",
            ),
            spacing="4",
            width="100%",
        ),
        with_sidebar=project_sidebar(PipelineState.project_id),
    )
