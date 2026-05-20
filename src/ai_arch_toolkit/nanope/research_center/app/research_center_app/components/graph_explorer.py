"""Interactive graph exploration components."""

from __future__ import annotations

import reflex as rx

from research_center_app.state import GraphState


def graph_stats_card() -> rx.Component:
    """Card showing graph statistics."""
    return rx.card(
        rx.vstack(
            rx.heading("Graph Stats", size="4"),
            rx.foreach(
                GraphState.stats,
                lambda item: rx.hstack(
                    rx.text(item[0], size="2", weight="medium"),
                    rx.badge(item[1], variant="surface", size="1"),
                    spacing="2",
                ),
            ),
            spacing="2",
        ),
        width="100%",
    )


def _result_item(item: dict) -> rx.Component:
    """Render a node result row."""
    return rx.hstack(
        rx.badge(item["type"], variant="surface", size="1"),
        rx.code(item["id"], size="1"),
        rx.text(item["preview"], size="1", color="var(--gray-a11)"),
        spacing="2",
        width="100%",
        padding_y="2px",
    )


def traversal_panel() -> rx.Component:
    """BFS/DFS traversal controls."""
    return rx.card(
        rx.vstack(
            rx.heading("Traversal (BFS / DFS)", size="3"),
            rx.input(
                placeholder="Start node ID...",
                value=GraphState.traversal_start,
                on_change=GraphState.set_traversal_start,
                width="100%",
                size="2",
            ),
            rx.hstack(
                rx.button("BFS", on_click=GraphState.run_bfs, size="2", variant="soft"),
                rx.button("DFS", on_click=GraphState.run_dfs, size="2", variant="soft"),
                spacing="2",
            ),
            rx.cond(
                GraphState.traversal_result.length() > 0,  # type: ignore[union-attr]
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(GraphState.traversal_result, _result_item),
                        spacing="1",
                        width="100%",
                    ),
                    max_height="250px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="3",
        ),
        width="100%",
    )


def path_panel() -> rx.Component:
    """Shortest path and all paths controls."""
    return rx.card(
        rx.vstack(
            rx.heading("Path Finding", size="3"),
            rx.hstack(
                rx.input(
                    placeholder="Source node ID",
                    value=GraphState.path_source,
                    on_change=GraphState.set_path_source,
                    size="2",
                ),
                rx.input(
                    placeholder="Target node ID",
                    value=GraphState.path_target,
                    on_change=GraphState.set_path_target,
                    size="2",
                ),
                spacing="2",
                width="100%",
            ),
            rx.hstack(
                rx.button(
                    "Shortest Path",
                    on_click=GraphState.run_shortest_path,
                    size="2",
                    variant="soft",
                ),
                rx.button(
                    "All Paths",
                    on_click=GraphState.run_find_all_paths,
                    size="2",
                    variant="soft",
                ),
                spacing="2",
            ),
            rx.cond(
                GraphState.path_result != "",
                rx.code_block(GraphState.path_result, language="log", width="100%"),
                rx.fragment(),
            ),
            spacing="3",
        ),
        width="100%",
    )


def ego_panel() -> rx.Component:
    """Ego graph exploration."""
    return rx.card(
        rx.vstack(
            rx.heading("Ego Graph", size="3"),
            rx.hstack(
                rx.input(
                    placeholder="Center node ID",
                    value=GraphState.ego_node,
                    on_change=GraphState.set_ego_node,
                    size="2",
                ),
                rx.input(
                    placeholder="Radius",
                    value=GraphState.ego_radius.to(str),  # type: ignore[union-attr]
                    on_change=GraphState.set_ego_radius,
                    size="2",
                    width="80px",
                ),
                spacing="2",
                width="100%",
            ),
            rx.button(
                "Explore Ego Graph",
                on_click=GraphState.run_ego_graph,
                size="2",
                variant="soft",
            ),
            rx.cond(
                GraphState.ego_result.length() > 0,  # type: ignore[union-attr]
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(GraphState.ego_result, _result_item),
                        spacing="1",
                        width="100%",
                    ),
                    max_height="250px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="3",
        ),
        width="100%",
    )


def ranking_panel() -> rx.Component:
    """PageRank and centrality controls."""
    return rx.card(
        rx.vstack(
            rx.heading("Ranking Algorithms", size="3"),
            rx.hstack(
                rx.button(
                    "PageRank",
                    on_click=GraphState.run_pagerank,
                    size="2",
                    variant="soft",
                ),
                rx.button(
                    "Degree Centrality",
                    on_click=GraphState.run_centrality,
                    size="2",
                    variant="soft",
                ),
                spacing="2",
            ),
            rx.cond(
                GraphState.ranking_result.length() > 0,  # type: ignore[union-attr]
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(GraphState.ranking_result, _result_item),
                        spacing="1",
                        width="100%",
                    ),
                    max_height="300px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="3",
        ),
        width="100%",
    )


def structure_panel() -> rx.Component:
    """Connected components and orphan nodes."""
    return rx.card(
        rx.vstack(
            rx.heading("Graph Structure", size="3"),
            rx.hstack(
                rx.button(
                    "Connected Components",
                    on_click=GraphState.run_connected_components,
                    size="2",
                    variant="soft",
                ),
                rx.button(
                    "Orphan Nodes",
                    on_click=GraphState.run_orphan_nodes,
                    size="2",
                    variant="soft",
                ),
                spacing="2",
            ),
            rx.cond(
                GraphState.components_result.length() > 0,  # type: ignore[union-attr]
                rx.box(
                    rx.text("Components", size="2", weight="bold"),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(GraphState.components_result, _result_item),
                            spacing="1",
                            width="100%",
                        ),
                        max_height="200px",
                        width="100%",
                    ),
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.cond(
                GraphState.orphan_nodes.length() > 0,  # type: ignore[union-attr]
                rx.box(
                    rx.text("Orphan Nodes", size="2", weight="bold"),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(GraphState.orphan_nodes, _result_item),
                            spacing="1",
                            width="100%",
                        ),
                        max_height="200px",
                        width="100%",
                    ),
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="3",
        ),
        width="100%",
    )


def algorithm_status() -> rx.Component:
    """Status bar showing last algorithm run."""
    return rx.cond(
        GraphState.algorithm_result != "",
        rx.callout(
            rx.hstack(
                rx.text(GraphState.algorithm_name, weight="bold", size="2"),
                rx.text(" — ", size="2"),
                rx.text(GraphState.algorithm_result, size="2"),
                spacing="1",
            ),
            icon="info",
            color_scheme="blue",
            width="100%",
        ),
        rx.fragment(),
    )
