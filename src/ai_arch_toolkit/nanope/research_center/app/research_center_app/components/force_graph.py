"""Custom Reflex component wrapping react-force-graph-3d."""

from __future__ import annotations

import reflex as rx


class ForceGraph3D(rx.Component):
    """3D force-directed graph visualization."""

    library = "react-force-graph-3d@1.25"
    tag = "ForceGraph3D"
    is_default = True

    graph_data: rx.Var[dict] = {}  # type: ignore[assignment]
    width: rx.Var[int] = 800  # type: ignore[assignment]
    height: rx.Var[int] = 500  # type: ignore[assignment]
    background_color: rx.Var[str] = "#111118"  # type: ignore[assignment]
    node_auto_color_by: rx.Var[str] = "group"  # type: ignore[assignment]
    node_label: rx.Var[str] = "name"  # type: ignore[assignment]
    link_directional_arrow_length: rx.Var[float] = 4.0  # type: ignore[assignment]
    link_directional_arrow_rel_pos: rx.Var[float] = 1.0  # type: ignore[assignment]
    link_width: rx.Var[float] = 1.0  # type: ignore[assignment]
    link_opacity: rx.Var[float] = 0.4  # type: ignore[assignment]
    node_rel_size: rx.Var[float] = 6.0  # type: ignore[assignment]


force_graph_3d = ForceGraph3D.create
