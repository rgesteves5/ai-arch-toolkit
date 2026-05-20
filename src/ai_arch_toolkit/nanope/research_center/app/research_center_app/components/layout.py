"""Sidebar and page layout shell."""

from __future__ import annotations

import reflex as rx


def sidebar() -> rx.Component:
    """Persistent sidebar with navigation."""
    return rx.box(
        rx.vstack(
            rx.heading("Research Center", size="5", weight="bold"),
            rx.separator(),
            rx.link("Projects", href="/", weight="medium", size="3"),
            rx.spacer(),
            rx.link(
                rx.hstack(rx.icon("plus", size=16), rx.text("New Project")),
                href="/",
                size="2",
            ),
            spacing="4",
            height="100%",
            padding="24px",
        ),
        width="240px",
        min_width="240px",
        height="100vh",
        border_right="1px solid var(--gray-a5)",
        position="fixed",
        left="0",
        top="0",
        background="var(--gray-a2)",
    )


def project_sidebar(project_id: str) -> rx.Component:
    """Sidebar with project-specific navigation."""
    return rx.box(
        rx.vstack(
            rx.heading("Research Center", size="5", weight="bold"),
            rx.separator(),
            rx.link("Projects", href="/", size="3"),
            rx.separator(),
            rx.text("Project", size="2", color="var(--gray-a11)"),
            rx.link("Monitor", href=f"/project/{project_id}", size="3", weight="medium"),
            rx.link("Wiki", href=f"/project/{project_id}/wiki", size="3", weight="medium"),
            rx.link("Graph", href=f"/project/{project_id}/graph", size="3", weight="medium"),
            rx.link("Report", href=f"/project/{project_id}/report", size="3", weight="medium"),
            rx.spacer(),
            rx.link(
                rx.hstack(rx.icon("plus", size=16), rx.text("New Project")),
                href="/",
                size="2",
            ),
            spacing="3",
            height="100%",
            padding="24px",
        ),
        width="240px",
        min_width="240px",
        height="100vh",
        border_right="1px solid var(--gray-a5)",
        position="fixed",
        left="0",
        top="0",
        background="var(--gray-a2)",
    )


def page_shell(*children: rx.Component, with_sidebar: rx.Component | None = None) -> rx.Component:
    """Page layout wrapper with sidebar + main content area."""
    sb = with_sidebar if with_sidebar is not None else sidebar()
    return rx.hstack(
        sb,
        rx.box(
            rx.container(
                *children,
                padding_y="32px",
                max_width="960px",
            ),
            margin_left="240px",
            width="100%",
            min_height="100vh",
        ),
        spacing="0",
        width="100%",
    )
