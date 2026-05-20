"""Projects page — project list and creation."""

from __future__ import annotations

import reflex as rx

from research_center_app.components.layout import page_shell
from research_center_app.components.project_form import project_form_dialog
from research_center_app.state import ProjectState


def _status_color(status: str) -> str:
    """Map status to badge color scheme."""
    if status == "complete":
        return "green"
    if status == "running":
        return "blue"
    return "gray"


def _project_card(project: dict) -> rx.Component:  # noqa: ARG001
    """Render a single project card."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading(project["topic"], size="4"),
                rx.spacer(),
                rx.badge(
                    project["status"],
                    variant="surface",
                    size="1",
                    color_scheme=rx.cond(
                        project["status"] == "complete",
                        "green",
                        rx.cond(project["status"] == "running", "blue", "gray"),
                    ),
                ),
                width="100%",
            ),
            rx.hstack(
                rx.text(
                    f"Nodes: {project['wiki_node_count']}",
                    size="2",
                    color="var(--gray-a11)",
                ),
                rx.text(
                    project["created_at"],
                    size="1",
                    color="var(--gray-a11)",
                ),
                spacing="3",
            ),
            rx.hstack(
                rx.link(
                    rx.button("Open", variant="solid", size="2"),
                    href=f"/project/{project['id']}",
                ),
                rx.button(
                    "Delete",
                    variant="soft",
                    color_scheme="red",
                    size="2",
                    on_click=ProjectState.delete_project(project["id"]),
                ),
                spacing="2",
            ),
            spacing="3",
        ),
        width="100%",
    )


@rx.page(route="/", on_load=ProjectState.load_projects)  # type: ignore[misc]
def projects_page() -> rx.Component:
    """Main projects listing page."""
    return page_shell(
        rx.vstack(
            rx.hstack(
                rx.heading("Projects", size="6"),
                rx.spacer(),
                project_form_dialog(),
                width="100%",
                align="center",
            ),
            rx.separator(),
            rx.cond(
                ProjectState.projects.length() > 0,  # type: ignore[union-attr]
                rx.vstack(
                    rx.foreach(ProjectState.projects, _project_card),
                    spacing="3",
                    width="100%",
                ),
                rx.center(
                    rx.vstack(
                        rx.text("No projects yet.", size="3", color="var(--gray-a11)"),
                        rx.text(
                            "Create your first research project to get started.",
                            size="2",
                            color="var(--gray-a11)",
                        ),
                        align="center",
                        spacing="2",
                    ),
                    padding_y="64px",
                ),
            ),
            spacing="4",
            width="100%",
        ),
    )
