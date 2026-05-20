"""Graph page — interactive wiki graph exploration with algorithms."""

from __future__ import annotations

import reflex as rx

from research_center_app.components.force_graph import force_graph_3d
from research_center_app.components.graph_explorer import (
    algorithm_status,
    ego_panel,
    graph_stats_card,
    path_panel,
    ranking_panel,
    structure_panel,
    traversal_panel,
)
from research_center_app.components.layout import page_shell, project_sidebar
from research_center_app.state import GraphState, PipelineState


def _tab_nav() -> rx.Component:
    return rx.hstack(
        rx.link(
            rx.button("Monitor", variant="soft", size="2"),
            href=f"/project/{PipelineState.project_id}",
        ),
        rx.link(
            rx.button("Wiki", variant="soft", size="2"),
            href=f"/project/{PipelineState.project_id}/wiki",
        ),
        rx.link(
            rx.button("Graph", variant="solid", size="2"),
            href=f"/project/{PipelineState.project_id}/graph",
        ),
        rx.link(
            rx.button("Report", variant="soft", size="2"),
            href=f"/project/{PipelineState.project_id}/report",
        ),
        spacing="2",
    )


@rx.page(  # type: ignore[misc]
    route="/project/[id]/graph",
    on_load=[
        PipelineState.load_project,
        GraphState.load_graph_data,
    ],
)
def graph_page() -> rx.Component:
    """Interactive graph exploration page."""
    return page_shell(
        rx.vstack(
            # Header
            rx.hstack(
                rx.heading("Wiki Graph", size="5"),
                rx.spacer(),
                rx.text(
                    "Nodes: ", GraphState.stats["nodes"],
                    size="2",
                    color="var(--gray-a11)",
                ),
                width="100%",
                align="center",
            ),
            rx.separator(),
            _tab_nav(),
            rx.separator(),

            # 3D graph visualization
            rx.box(
                force_graph_3d(
                    graph_data=GraphState.graph_data,
                    width=900,
                    height=500,
                    background_color="#111118",
                    node_auto_color_by="group",
                    node_label="name",
                    link_directional_arrow_length=4.0,
                    link_directional_arrow_rel_pos=1.0,
                    link_width=1.0,
                    link_opacity=0.4,
                    node_rel_size=6.0,
                ),
                width="100%",
                border_radius="8px",
                overflow="hidden",
                border="1px solid var(--gray-a5)",
            ),

            # Algorithm status
            algorithm_status(),

            # Single collapsible for all algorithm tools
            rx.accordion.root(
                rx.accordion.item(
                    header=rx.text("Algorithm Tools", weight="medium", size="3"),
                    content=rx.vstack(
                        # Stats
                        graph_stats_card(),

                        # Two-column layout for algorithm panels
                        rx.hstack(
                            rx.vstack(
                                traversal_panel(),
                                path_panel(),
                                spacing="4",
                                width="50%",
                            ),
                            rx.vstack(
                                ego_panel(),
                                ranking_panel(),
                                spacing="4",
                                width="50%",
                            ),
                            spacing="4",
                            width="100%",
                            align="start",
                        ),

                        # Full-width structure panel
                        structure_panel(),

                        spacing="4",
                        width="100%",
                        padding_y="8px",
                    ),
                ),
                type="multiple",
                collapsible=True,
                width="100%",
            ),

            spacing="4",
            width="100%",
        ),
        with_sidebar=project_sidebar(PipelineState.project_id),
    )
