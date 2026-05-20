"""Wiki browser components: search, category filter, node detail."""

from __future__ import annotations

import reflex as rx

from research_center_app.state import WikiState


def wiki_search_panel() -> rx.Component:
    """Left panel: search input, category filter, and result list."""
    return rx.vstack(
        rx.input(
            placeholder="Search wiki...",
            value=WikiState.search_query,
            on_change=WikiState.set_search_query,
            on_key_down=rx.cond(  # type: ignore[arg-type]
                rx.Var.create("Enter"),  # placeholder — submit on blur/button instead
                WikiState.search_wiki,
                None,
            ),
            width="100%",
        ),
        rx.button("Search", on_click=WikiState.search_wiki, width="100%", variant="soft"),
        rx.separator(),
        rx.text("Categories", size="2", weight="bold"),
        rx.vstack(
            rx.link(
                "All",
                on_click=[WikiState.set_category(""), WikiState.search_wiki],
                size="2",
                cursor="pointer",
            ),
            rx.foreach(
                WikiState.categories,
                lambda cat: rx.link(
                    f"{cat['name']} ({cat['count']})",
                    on_click=[WikiState.set_category(cat["name"]), WikiState.search_wiki],
                    size="2",
                    cursor="pointer",
                ),
            ),
            spacing="1",
            width="100%",
        ),
        rx.separator(),
        rx.text(f"{WikiState.node_count} nodes total", size="1", color="var(--gray-a11)"),
        rx.separator(),
        rx.text("Results", size="2", weight="bold"),
        rx.scroll_area(
            rx.vstack(
                rx.foreach(WikiState.search_results, _search_result_item),
                spacing="2",
                width="100%",
            ),
            max_height="500px",
            width="100%",
        ),
        spacing="3",
        width="100%",
    )


def _search_result_item(item: dict) -> rx.Component:
    """Render a single search result as a clickable card."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.badge(item["type"], variant="surface", size="1"),
                rx.cond(
                    item["score"] != "",
                    rx.text(f"score: {item['score']}", size="1", color="var(--gray-a11)"),
                    rx.fragment(),
                ),
                spacing="2",
            ),
            rx.text(item["content_preview"], size="2"),
            rx.hstack(
                rx.text(f"src: {item['source']}", size="1", color="var(--gray-a11)"),
                rx.text(item["confidence"], size="1", color="var(--gray-a11)"),
                spacing="2",
            ),
            spacing="1",
        ),
        on_click=WikiState.select_node(item["id"]),
        cursor="pointer",
        width="100%",
        _hover={"background": "var(--gray-a3)"},
    )


def wiki_node_detail() -> rx.Component:
    """Right panel: selected node detail with neighbors."""
    return rx.cond(
        WikiState.selected_node,
        rx.vstack(
            rx.hstack(
                rx.badge(
                    WikiState.selected_node["type"],
                    variant="solid",
                    size="2",
                    color_scheme="blue",
                ),
                rx.text(
                    f"Confidence: {WikiState.selected_node['confidence']}",
                    size="2",
                ),
                rx.text(
                    f"Source: {WikiState.selected_node['source']}",
                    size="2",
                    color="var(--gray-a11)",
                ),
                spacing="3",
            ),
            rx.separator(),
            rx.text("Content", size="3", weight="bold"),
            rx.code_block(
                WikiState.selected_node["content"],
                language="json",
                width="100%",
            ),
            rx.cond(
                WikiState.selected_node["metadata"] != "{}",
                rx.box(
                    rx.text("Metadata", size="3", weight="bold"),
                    rx.code_block(
                        WikiState.selected_node["metadata"],
                        language="json",
                        width="100%",
                    ),
                ),
                rx.fragment(),
            ),
            rx.hstack(
                rx.text(
                    f"Accessed: {WikiState.selected_node['access_count']}x",
                    size="1",
                    color="var(--gray-a11)",
                ),
                rx.text(
                    f"Created: {WikiState.selected_node['created_at']}",
                    size="1",
                    color="var(--gray-a11)",
                ),
                spacing="3",
            ),
            rx.separator(),
            rx.text("Neighbors", size="3", weight="bold"),
            rx.vstack(
                rx.foreach(WikiState.node_neighbors, _neighbor_item),
                spacing="2",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        rx.center(
            rx.text("Select a node to view details", color="var(--gray-a11)"),
            height="200px",
        ),
    )


def _neighbor_item(neighbor: dict) -> rx.Component:
    """Render a neighbor node link."""
    return rx.card(
        rx.hstack(
            rx.badge(neighbor["type"], variant="outline", size="1"),
            rx.text(neighbor["content_preview"], size="2"),
            spacing="2",
        ),
        on_click=WikiState.select_node(neighbor["id"]),
        cursor="pointer",
        width="100%",
        _hover={"background": "var(--gray-a3)"},
    )
