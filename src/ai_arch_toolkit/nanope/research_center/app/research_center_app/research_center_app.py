"""Reflex app entry point — registers all pages."""

import reflex as rx

from research_center_app.pages.graph import graph_page  # noqa: F401
from research_center_app.pages.monitor import monitor_page  # noqa: F401
from research_center_app.pages.projects import projects_page  # noqa: F401
from research_center_app.pages.report import report_page  # noqa: F401
from research_center_app.pages.wiki import wiki_page  # noqa: F401

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="blue",
        radius="medium",
    ),
)
