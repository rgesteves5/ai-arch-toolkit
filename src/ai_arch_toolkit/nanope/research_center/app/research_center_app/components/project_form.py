"""New project creation dialog."""

from __future__ import annotations

import reflex as rx

from research_center_app.state import ProjectState


def project_form_dialog() -> rx.Component:
    """Dialog for creating a new research project."""
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.button(
                rx.icon("plus", size=16),
                "New Project",
                size="3",
                variant="solid",
            ),
        ),
        rx.dialog.content(
            rx.dialog.title("Create New Project"),
            rx.dialog.description("Configure your research pipeline."),
            rx.form(
                rx.vstack(
                    rx.text("Topic *", size="2", weight="medium"),
                    rx.input(
                        name="topic",
                        placeholder="e.g., Quantum Computing Applications",
                        required=True,
                    ),
                    rx.text("Owner Brief", size="2", weight="medium"),
                    rx.text_area(
                        name="brief",
                        placeholder="Instructions for the research pipeline...",
                        rows="4",
                    ),
                    rx.text("Budget (USD)", size="2", weight="medium"),
                    rx.input(
                        name="budget",
                        type="number",
                        default_value="1.00",
                        min="0.10",
                        max="5.00",
                        step="0.10",
                    ),
                    rx.text("Max Cycles", size="2", weight="medium"),
                    rx.select(
                        ["1", "2", "3", "4", "5"],
                        name="max_cycles",
                        default_value="3",
                    ),
                    rx.text("Grok Model", size="2", weight="medium"),
                    rx.select(
                        [
                            "grok-4-1-fast-reasoning",
                            "grok-3-fast",
                            "grok-3-mini-fast",
                        ],
                        name="grok_model",
                        default_value="grok-4-1-fast-reasoning",
                    ),
                    rx.text("Gemini Model", size="2", weight="medium"),
                    rx.select(
                        [
                            "gemini-3-flash",
                            "gemini-2.5-flash",
                            "gemini-2.5-pro",
                        ],
                        name="gemini_model",
                        default_value="gemini-3-flash",
                    ),
                    rx.hstack(
                        rx.dialog.close(rx.button("Cancel", variant="soft")),
                        rx.button("Create", type="submit"),
                        spacing="3",
                        justify="end",
                        width="100%",
                    ),
                    spacing="3",
                    width="100%",
                ),
                on_submit=ProjectState.create_project,
                reset_on_submit=True,
            ),
            max_width="480px",
        ),
    )
