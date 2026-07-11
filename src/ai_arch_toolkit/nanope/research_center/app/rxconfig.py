"""Reflex configuration for the Research Center app."""

import reflex as rx

config = rx.Config(
    app_name="research_center_app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="dark",
                accent_color="blue",
                radius="medium",
            )
        ),
    ],
)
