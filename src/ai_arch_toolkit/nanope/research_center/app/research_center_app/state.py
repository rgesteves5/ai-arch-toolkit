"""State classes for the Research Center app."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import reflex as rx

from ai_arch_toolkit.nanope.research_center._db import _get_conn
from ai_arch_toolkit.nanope.research_center._pipeline import (
    PipelineEvent,
    run_pipeline,
)
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore


class _PipelineStopped(Exception):
    """Raised when the pipeline is stopped by user."""


def _migrate_from_json(conn: sqlite3.Connection) -> None:
    """One-time migration from JSON files to SQLite."""
    legacy_dir = Path.home() / ".research_center" / "projects"
    if not legacy_dir.is_dir():
        return
    # Only migrate if projects table is empty and legacy dir has subdirs
    row = conn.execute("SELECT COUNT(*) FROM projects").fetchone()
    if row[0] > 0:
        return
    subdirs = [d for d in legacy_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return

    with conn:
        for proj_dir in subdirs:
            project_id = proj_dir.name
            # Migrate meta.json -> projects
            meta_path = proj_dir / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            conn.execute(
                """INSERT OR IGNORE INTO projects
                   (id, topic, brief, budget, max_cycles, grok_model, gemini_model,
                    created_at, status, wiki_node_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id,
                    meta.get("topic", "Untitled"),
                    meta.get("brief", ""),
                    meta.get("budget", "1.00"),
                    meta.get("max_cycles", "3"),
                    meta.get("grok_model", "grok-4-1-fast-reasoning"),
                    meta.get("gemini_model", "gemini-3-flash"),
                    meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                    meta.get("status", "new"),
                    meta.get("wiki_node_count", "0"),
                ),
            )

            # Migrate result.json -> results + phase_costs
            result_path = proj_dir / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    saved = json.load(f)
                conn.execute(
                    """INSERT OR IGNORE INTO results
                       (project_id, topic, report, total_cost, cycles_completed,
                        budget_remaining, wiki_node_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        project_id,
                        saved.get("topic", ""),
                        saved.get("report", ""),
                        float(saved.get("total_cost", 0)),
                        int(saved.get("cycles_completed", 0)),
                        float(saved.get("budget_remaining", 0)),
                        int(saved.get("wiki_node_count", 0)),
                    ),
                )
                for pc in saved.get("phase_costs", []):
                    conn.execute(
                        """INSERT INTO phase_costs (project_id, cycle, agent, cost)
                           VALUES (?, ?, ?, ?)""",
                        (
                            project_id,
                            int(pc.get("cycle", 0)),
                            str(pc.get("agent", "")),
                            float(pc.get("cost", 0)),
                        ),
                    )

            # Migrate wiki.json -> wikis (raw text)
            wiki_path = proj_dir / "wiki.json"
            if wiki_path.exists():
                wiki_text = wiki_path.read_text()
                conn.execute(
                    "INSERT OR IGNORE INTO wikis (project_id, graph_json) VALUES (?, ?)",
                    (project_id, wiki_text),
                )

    # Rename legacy dir to backup
    backup_dir = Path.home() / ".research_center" / "projects_backup"
    if not backup_dir.exists():
        legacy_dir.rename(backup_dir)


# Initialize DB at module load (_get_conn handles table creation)
_migrate_from_json(_get_conn())


# ---------------------------------------------------------------------------
# Module-level wiki cache (GraphStore is not serializable)
# ---------------------------------------------------------------------------

_wiki_cache: dict[str, GraphStore] = {}
_stop_flags: dict[str, asyncio.Event] = {}  # project_id -> stop event


async def get_wiki(project_id: str) -> GraphStore:
    """Get or load the wiki GraphStore for a project."""
    if project_id in _wiki_cache:
        return _wiki_cache[project_id]

    conn = _get_conn()
    row = conn.execute(
        "SELECT graph_json FROM wikis WHERE project_id = ?", (project_id,)
    ).fetchone()

    if row and row["graph_json"] != "{}":
        from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

        wiki = await GraphStore.from_dict(json.loads(row["graph_json"]), NetworkXBackend())
    else:
        from ai_arch_toolkit.nanope.research_center._pipeline import create_wiki

        wiki = await create_wiki()

    _wiki_cache[project_id] = wiki
    return wiki


async def _save_wiki(project_id: str) -> None:
    """Persist wiki to database if cached."""
    if project_id in _wiki_cache:
        data = await _wiki_cache[project_id].to_dict()
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO wikis (project_id, graph_json) VALUES (?, ?)",
            (project_id, json.dumps(data, default=str)),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# ProjectState — project CRUD
# ---------------------------------------------------------------------------


class ProjectState(rx.State):
    """Manages the list of research projects."""

    projects: list[dict[str, str]] = []

    def load_projects(self) -> None:
        """Load projects from SQLite."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
        self.projects = [dict(row) for row in rows]

    def create_project(self, form_data: dict) -> rx.event.EventSpec:
        """Insert a new project into the database."""
        project_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()

        meta = {
            "id": project_id,
            "topic": form_data.get("topic", "Untitled"),
            "brief": form_data.get("brief", ""),
            "budget": form_data.get("budget", "1.00"),
            "max_cycles": form_data.get("max_cycles", "3"),
            "grok_model": form_data.get("grok_model", "grok-4-1-fast-reasoning"),
            "gemini_model": form_data.get("gemini_model", "gemini-3-flash"),
            "created_at": created_at,
            "status": "new",
            "wiki_node_count": "0",
        }
        conn = _get_conn()
        conn.execute(
            """INSERT INTO projects
               (id, topic, brief, budget, max_cycles, grok_model, gemini_model,
                created_at, status, wiki_node_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                meta["id"],
                meta["topic"],
                meta["brief"],
                meta["budget"],
                meta["max_cycles"],
                meta["grok_model"],
                meta["gemini_model"],
                meta["created_at"],
                meta["status"],
                meta["wiki_node_count"],
            ),
        )
        conn.commit()

        self.projects.insert(0, meta)
        return rx.redirect(f"/project/{project_id}")

    def delete_project(self, project_id: str) -> None:
        """Remove a project from the database (CASCADE handles children)."""
        conn = _get_conn()
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
        _wiki_cache.pop(project_id, None)
        self.projects = [p for p in self.projects if p.get("id") != project_id]


# ---------------------------------------------------------------------------
# PipelineState — pipeline execution and monitoring
# ---------------------------------------------------------------------------


class PipelineState(rx.State):
    """Manages pipeline execution and live progress tracking."""

    # Config
    project_id: str = ""
    topic: str = ""
    owner_brief: str = ""
    budget: float = 1.0
    max_cycles: int = 3
    grok_model: str = "grok-4-1-fast-reasoning"
    gemini_model: str = "gemini-3-flash"

    # Progress
    is_running: bool = False
    is_stopped: bool = False
    current_agent: str = ""
    current_cycle: int = 0
    total_spent: float = 0.0
    events: list[dict] = []
    directives_by_cycle: list[dict] = []

    # Result
    report: str = ""
    result: dict = {}
    phase_costs: list[dict[str, str]] = []
    error: str = ""

    def load_project(self) -> None:
        """Load project config and any saved result from SQLite."""
        project_id = self.router._page.params.get("id", "")
        if not project_id:
            return

        # Reset all state before loading — prevents stale data from a previous project
        self.project_id = project_id
        self.is_running = False
        self.is_stopped = False
        self.current_agent = ""
        self.current_cycle = 0
        self.total_spent = 0.0
        self.events = []
        self.directives_by_cycle = []
        self.report = ""
        self.result = {}
        self.phase_costs = []
        self.error = ""

        conn = _get_conn()

        # Load project config
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row:
            self.topic = row["topic"]
            self.owner_brief = row["brief"]
            self.budget = float(row["budget"])
            self.max_cycles = int(row["max_cycles"])
            self.grok_model = row["grok_model"]
            self.gemini_model = row["gemini_model"]

        # Load result
        result_row = conn.execute(
            "SELECT * FROM results WHERE project_id = ?", (project_id,)
        ).fetchone()
        if result_row:
            self.report = result_row["report"]
            self.total_spent = float(result_row["total_cost"])
            self.current_cycle = int(result_row["cycles_completed"])
            self.result = dict(result_row)

        # Load phase costs
        cost_rows = conn.execute(
            "SELECT cycle, agent, cost FROM phase_costs WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
        if cost_rows:
            self.phase_costs = [
                {
                    "cycle": str(r["cycle"]),
                    "agent": r["agent"],
                    "cost": str(r["cost"]),
                }
                for r in cost_rows
            ]

        # Load pipeline events (survive reload)
        event_rows = conn.execute(
            "SELECT event_json FROM pipeline_events WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
        if event_rows:
            self.events = [json.loads(r["event_json"]) for r in event_rows]
            # Reconstruct directives_by_cycle from manager_decision events
            self.directives_by_cycle = [
                ev["directives"]
                for ev in self.events
                if ev.get("type") == "manager_decision"
            ]

    def stop_pipeline(self) -> None:
        """Signal the running pipeline to stop."""
        if self.project_id and self.project_id in _stop_flags:
            _stop_flags[self.project_id].set()

    def reset_pipeline(self) -> None:
        """Reset pipeline state back to initial (keeps wiki data)."""
        self.is_running = False
        self.is_stopped = False
        self.current_agent = ""
        self.current_cycle = 0
        self.total_spent = 0.0
        self.events = []
        self.directives_by_cycle = []
        self.report = ""
        self.result = {}
        self.phase_costs = []
        self.error = ""
        # Clear stop flag
        _stop_flags.pop(self.project_id, None)
        # Clear persisted pipeline data (keep wiki)
        if self.project_id:
            conn = _get_conn()
            conn.execute(
                "DELETE FROM pipeline_events WHERE project_id = ?", (self.project_id,)
            )
            conn.execute(
                "DELETE FROM results WHERE project_id = ?", (self.project_id,)
            )
            conn.execute(
                "DELETE FROM phase_costs WHERE project_id = ?", (self.project_id,)
            )
            conn.execute(
                "UPDATE projects SET status = 'new' WHERE id = ?", (self.project_id,)
            )
            conn.commit()

    @rx.event(background=True)
    async def start_pipeline(self) -> None:
        """Run the research pipeline with live event updates."""
        async with self:
            if self.is_running:
                return
            self.is_running = True
            self.is_stopped = False
            self.error = ""
            # Only clear events/directives on fresh start (not resume)
            if not self.is_stopped:
                self.events = []
                self.directives_by_cycle = []
                self.report = ""
                self.result = {}
            project_id = self.project_id
            topic = self.topic
            owner_brief = self.owner_brief
            budget = self.budget
            max_cycles = self.max_cycles
            grok_model = self.grok_model
            gemini_model = self.gemini_model

        # Set up stop flag
        stop_event = asyncio.Event()
        _stop_flags[project_id] = stop_event

        wiki = await get_wiki(project_id)
        conn = _get_conn()

        async def on_event(ev: PipelineEvent) -> None:
            # Check stop flag
            if stop_event.is_set():
                raise _PipelineStopped("Pipeline stopped by user")
            # Persist event to DB
            ev_dict = asdict(ev)
            conn.execute(
                "INSERT INTO pipeline_events (project_id, event_json) VALUES (?, ?)",
                (project_id, json.dumps(ev_dict, default=str)),
            )
            conn.commit()
            async with self:
                self.current_agent = ev.agent
                self.current_cycle = ev.cycle
                self.total_spent = ev.total_spent
                self.events = [*self.events, ev_dict]
                if ev.type == "manager_decision":
                    self.directives_by_cycle = [*self.directives_by_cycle, ev.directives]

        try:
            pipeline_result = await run_pipeline(
                topic,
                wiki,
                owner_brief=owner_brief,
                budget=budget,
                max_cycles=max_cycles,
                grok_model=grok_model,
                gemini_model=gemini_model,
                project_id=project_id,
                on_event=on_event,
            )

            # Persist wiki
            await _save_wiki(project_id)

            # Persist result
            conn.execute(
                """INSERT OR REPLACE INTO results
                   (project_id, topic, report, total_cost, cycles_completed,
                    budget_remaining, wiki_node_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id,
                    pipeline_result.topic,
                    pipeline_result.report,
                    pipeline_result.total_cost,
                    pipeline_result.cycles_completed,
                    pipeline_result.budget_remaining,
                    pipeline_result.wiki_node_count,
                ),
            )

            # Persist phase costs
            for pc in pipeline_result.phase_costs:
                conn.execute(
                    """INSERT INTO phase_costs (project_id, cycle, agent, cost)
                       VALUES (?, ?, ?, ?)""",
                    (
                        project_id,
                        int(pc.get("cycle", 0)),
                        str(pc.get("agent", "")),
                        float(pc.get("cost", 0)),
                    ),
                )

            # Update project status
            conn.execute(
                "UPDATE projects SET status = 'complete', wiki_node_count = ? WHERE id = ?",
                (str(pipeline_result.wiki_node_count), project_id),
            )
            conn.commit()

            result_dict = {
                "topic": pipeline_result.topic,
                "report": pipeline_result.report,
                "total_cost": pipeline_result.total_cost,
                "cycles_completed": pipeline_result.cycles_completed,
                "budget_remaining": pipeline_result.budget_remaining,
                "wiki_node_count": pipeline_result.wiki_node_count,
                "phase_costs": pipeline_result.phase_costs,
            }

            async with self:
                self.report = pipeline_result.report
                self.result = result_dict
                self.phase_costs = [
                    {
                        "cycle": str(pc.get("cycle", "")),
                        "agent": str(pc.get("agent", "")),
                        "cost": str(pc.get("cost", "")),
                    }
                    for pc in pipeline_result.phase_costs
                ]
                self.is_running = False

        except _PipelineStopped:
            # Save wiki progress on stop
            await _save_wiki(project_id)
            async with self:
                self.is_running = False
                self.is_stopped = True
                self.error = ""

        except Exception as e:
            async with self:
                self.error = str(e)
                self.is_running = False

        finally:
            _stop_flags.pop(project_id, None)


# ---------------------------------------------------------------------------
# WikiState — wiki browsing
# ---------------------------------------------------------------------------


class WikiState(rx.State):
    """Manages wiki knowledge graph browsing."""

    search_query: str = ""
    selected_category: str = ""
    categories: list[dict[str, str]] = []
    search_results: list[dict] = []
    selected_node: dict = {}
    node_neighbors: list[dict] = []
    node_count: int = 0
    _project_id: str = ""

    def set_project(self, project_id: str) -> None:
        """Set the active project for wiki browsing."""
        self._project_id = project_id

    @rx.event(background=True)
    async def load_wiki_data(self) -> None:
        """Load wiki categories and stats."""
        async with self:
            project_id = self.router._page.params.get("id", "")
        if not project_id:
            return
        wiki = await get_wiki(project_id)
        count = await wiki.count()

        # Get categories from type index
        cat_list: list[dict[str, str]] = []
        if hasattr(wiki, "_type_index") and wiki._type_index:
            for cat_name, node_ids in wiki._type_index.items():
                cat_list.append({"name": cat_name, "count": str(len(node_ids))})
        cat_list.sort(key=lambda c: c["name"])

        async with self:
            self._project_id = project_id
            self.node_count = count
            self.categories = cat_list
            self.search_results = []
            self.selected_node = {}
            self.node_neighbors = []

    @rx.event(background=True)
    async def search_wiki(self) -> None:
        """Search wiki nodes by query."""
        async with self:
            query = self.search_query
            project_id = self._project_id
            category = self.selected_category

        if not project_id:
            return

        wiki = await get_wiki(project_id)
        if query:
            results = await wiki.search(
                query, type=category if category else None, k=20
            )
            items = [
                {
                    "id": r.node.id,
                    "type": r.node.type,
                    "score": f"{r.score:.2f}",
                    "content_preview": _preview_content(r.node.content),
                    "source": r.node.source,
                    "confidence": f"{r.node.confidence:.1%}",
                }
                for r in results
            ]
        elif category:
            nodes = await wiki.list(type=category, limit=50)
            items = [
                {
                    "id": n.id,
                    "type": n.type,
                    "score": "",
                    "content_preview": _preview_content(n.content),
                    "source": n.source,
                    "confidence": f"{n.confidence:.1%}",
                }
                for n in nodes
            ]
        else:
            nodes = await wiki.list(limit=50)
            items = [
                {
                    "id": n.id,
                    "type": n.type,
                    "score": "",
                    "content_preview": _preview_content(n.content),
                    "source": n.source,
                    "confidence": f"{n.confidence:.1%}",
                }
                for n in nodes
            ]

        async with self:
            self.search_results = items

    @rx.event(background=True)
    async def select_node(self, node_id: str) -> None:
        """Load full detail for a wiki node."""
        async with self:
            project_id = self._project_id

        if not project_id:
            return

        wiki = await get_wiki(project_id)
        node = await wiki.get(node_id)
        if node is None:
            return

        neighbors = await wiki.neighbors(node_id, depth=1)
        neighbor_dicts = [
            {
                "id": n.id,
                "type": n.type,
                "content_preview": _preview_content(n.content),
            }
            for n in neighbors
        ]

        async with self:
            self.selected_node = {
                "id": node.id,
                "type": node.type,
                "content": json.dumps(node.content, indent=2),
                "metadata": json.dumps(node.metadata, indent=2),
                "source": node.source,
                "confidence": f"{node.confidence:.1%}",
                "access_count": str(node.access_count),
                "created_at": node.created_at.isoformat() if node.created_at else "",
                "timestamp": node.timestamp.isoformat() if node.timestamp else "",
            }
            self.node_neighbors = neighbor_dicts

    def set_search_query(self, value: str) -> None:
        """Update search query."""
        self.search_query = value

    def set_category(self, value: str) -> None:
        """Update selected category."""
        self.selected_category = value


# ---------------------------------------------------------------------------
# GraphState — interactive graph exploration with algorithms
# ---------------------------------------------------------------------------


class GraphState(rx.State):
    """Interactive graph exploration with algorithms."""

    _project_id: str = ""

    # Stats
    stats: dict[str, str] = {}

    # Algorithm results
    algorithm_name: str = ""
    algorithm_result: str = ""

    # BFS/DFS
    traversal_start: str = ""
    traversal_result: list[dict[str, str]] = []

    # Shortest path / all paths
    path_source: str = ""
    path_target: str = ""
    path_result: str = ""

    # Node inspection
    ego_node: str = ""
    ego_radius: int = 1
    ego_result: list[dict[str, str]] = []

    # Orphan nodes
    orphan_nodes: list[dict[str, str]] = []

    # Components
    components_result: list[dict[str, str]] = []

    # PageRank / Centrality
    ranking_result: list[dict[str, str]] = []

    # 3D graph visualization data
    graph_data: dict = {}

    @rx.event(background=True)
    async def load_graph_data(self) -> None:
        """Load graph stats and 3D visualization data."""
        async with self:
            project_id = self.router._page.params.get("id", "")
        if not project_id:
            return

        wiki = await get_wiki(project_id)
        count = await wiki.count()

        stats: dict[str, str] = {"nodes": str(count)}
        if hasattr(wiki, "_type_index") and wiki._type_index:
            for cat, ids in sorted(wiki._type_index.items()):
                stats[f"cat:{cat}"] = str(len(ids))

        # Build 3D graph data from backend
        backend = wiki._backend
        viz_nodes: list[dict] = []
        viz_links: list[dict] = []

        if hasattr(backend, "_graph"):
            stats["edges"] = str(backend._graph.number_of_edges())
            nx_graph = backend._graph
            for node_id, data in nx_graph.nodes(data=True):
                node_obj = data.get("node")
                node_type = getattr(node_obj, "type", "") if node_obj else ""
                content = getattr(node_obj, "content", {}) if node_obj else {}
                label = _preview_content(content) if content else str(node_id)
                viz_nodes.append({
                    "id": str(node_id),
                    "name": label[:60],
                    "group": node_type,
                })
            for _, _, data in nx_graph.edges(data=True):
                edge = data.get("edge")
                if edge:
                    viz_links.append({
                        "source": str(edge.source),
                        "target": str(edge.target),
                    })

        graph_data = {"nodes": viz_nodes, "links": viz_links}

        async with self:
            self._project_id = project_id
            self.stats = stats
            self.graph_data = graph_data

    @rx.event(background=True)
    async def run_bfs(self) -> None:
        """Run BFS from a start node."""
        async with self:
            project_id = self._project_id
            start = self.traversal_start.strip()
        if not project_id or not start:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "bfs"):
            async with self:
                self.algorithm_result = "Backend does not support graph algorithms."
            return

        nodes = await backend.bfs(start)
        items = [
            {"id": n.id, "type": str(getattr(n, "type", "")),
             "preview": _preview_content(getattr(n, "content", {}))}
            for n in nodes[:50]
        ]
        async with self:
            self.algorithm_name = f"BFS from {start}"
            self.traversal_result = items
            self.algorithm_result = f"Found {len(nodes)} nodes via BFS"

    @rx.event(background=True)
    async def run_dfs(self) -> None:
        """Run DFS from a start node."""
        async with self:
            project_id = self._project_id
            start = self.traversal_start.strip()
        if not project_id or not start:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "dfs"):
            async with self:
                self.algorithm_result = "Backend does not support graph algorithms."
            return

        nodes = await backend.dfs(start)
        items = [
            {"id": n.id, "type": str(getattr(n, "type", "")),
             "preview": _preview_content(getattr(n, "content", {}))}
            for n in nodes[:50]
        ]
        async with self:
            self.algorithm_name = f"DFS from {start}"
            self.traversal_result = items
            self.algorithm_result = f"Found {len(nodes)} nodes via DFS"

    @rx.event(background=True)
    async def run_shortest_path(self) -> None:
        """Find shortest path between two nodes."""
        async with self:
            project_id = self._project_id
            source = self.path_source.strip()
            target = self.path_target.strip()
        if not project_id or not source or not target:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "shortest_path"):
            async with self:
                self.path_result = "Backend does not support graph algorithms."
            return

        path = await backend.shortest_path(source, target)
        if path is None:
            async with self:
                self.path_result = f"No path found between {source} and {target}"
        else:
            node_ids = [n.id for n in path]
            async with self:
                self.path_result = " → ".join(node_ids)

    @rx.event(background=True)
    async def run_find_all_paths(self) -> None:
        """Find all paths between two nodes."""
        async with self:
            project_id = self._project_id
            source = self.path_source.strip()
            target = self.path_target.strip()
        if not project_id or not source or not target:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "find_all_paths"):
            async with self:
                self.path_result = "Backend does not support graph algorithms."
            return

        paths = await backend.find_all_paths(source, target, max_depth=6)
        lines = [f"Found {len(paths)} path(s):"]
        for i, p in enumerate(paths[:10]):
            lines.append(f"  {i + 1}. {' → '.join(p)}")
        async with self:
            self.path_result = "\n".join(lines)

    @rx.event(background=True)
    async def run_pagerank(self) -> None:
        """Run PageRank on the graph."""
        async with self:
            project_id = self._project_id
        if not project_id:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "pagerank"):
            async with self:
                self.algorithm_result = "Backend does not support PageRank."
            return

        scores = await backend.pagerank()
        # Sort by score descending, get top 20
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
        items = []
        for node_id, score in sorted_scores:
            node = await wiki.get(node_id)
            preview = _preview_content(node.content) if node else node_id
            items.append({
                "id": node_id,
                "type": getattr(node, "type", "") if node else "",
                "preview": f"{preview} (score: {score:.4f})",
            })
        async with self:
            self.algorithm_name = "PageRank (top 20)"
            self.ranking_result = items
            self.algorithm_result = f"PageRank computed for {len(scores)} nodes"

    @rx.event(background=True)
    async def run_centrality(self) -> None:
        """Run degree centrality on the graph."""
        async with self:
            project_id = self._project_id
        if not project_id:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "centrality"):
            async with self:
                self.algorithm_result = "Backend does not support centrality."
            return

        scores = await backend.centrality()
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
        items = []
        for node_id, score in sorted_scores:
            node = await wiki.get(node_id)
            preview = _preview_content(node.content) if node else node_id
            items.append({
                "id": node_id,
                "type": getattr(node, "type", "") if node else "",
                "preview": f"{preview} (centrality: {score:.4f})",
            })
        async with self:
            self.algorithm_name = "Degree Centrality (top 20)"
            self.ranking_result = items
            self.algorithm_result = f"Centrality computed for {len(scores)} nodes"

    @rx.event(background=True)
    async def run_connected_components(self) -> None:
        """Find connected components."""
        async with self:
            project_id = self._project_id
        if not project_id:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "connected_components"):
            async with self:
                self.algorithm_result = "Backend does not support connected components."
            return

        components = await backend.connected_components()
        items = []
        for i, comp in enumerate(components[:20]):
            ids_preview = ", ".join(list(comp)[:5])
            suffix = f"... +{len(comp) - 5}" if len(comp) > 5 else ""
            items.append({
                "id": str(i + 1),
                "type": f"{len(comp)} nodes",
                "preview": ids_preview + suffix,
            })
        async with self:
            self.algorithm_name = f"Connected Components ({len(components)} found)"
            self.components_result = items
            self.algorithm_result = f"Found {len(components)} connected component(s)"

    @rx.event(background=True)
    async def run_ego_graph(self) -> None:
        """Get the ego graph around a node."""
        async with self:
            project_id = self._project_id
            node_id = self.ego_node.strip()
            radius = self.ego_radius
        if not project_id or not node_id:
            return

        wiki = await get_wiki(project_id)
        backend = wiki._backend
        if not hasattr(backend, "ego_graph"):
            async with self:
                self.algorithm_result = "Backend does not support ego graph."
            return

        ego_backend = await backend.ego_graph(node_id, radius=radius)
        nodes = await ego_backend.list_nodes()
        items = [
            {"id": n.id, "type": str(getattr(n, "type", "")),
             "preview": _preview_content(getattr(n, "content", {}))}
            for n in nodes[:50]
        ]
        async with self:
            self.algorithm_name = f"Ego Graph: {node_id} (radius={radius})"
            self.ego_result = items
            self.algorithm_result = f"Ego graph has {len(nodes)} nodes"

    @rx.event(background=True)
    async def run_orphan_nodes(self) -> None:
        """Find nodes with no connections."""
        async with self:
            project_id = self._project_id
        if not project_id:
            return

        wiki = await get_wiki(project_id)
        # Get all nodes and check which have no edges
        all_nodes = await wiki.list(limit=500)
        orphans = []
        for node in all_nodes:
            edges_out = await wiki.edges(node.id, direction="out")
            edges_in = await wiki.edges(node.id, direction="in")
            if len(edges_out) == 0 and len(edges_in) == 0:
                orphans.append({
                    "id": node.id,
                    "type": node.type,
                    "preview": _preview_content(node.content),
                })
        async with self:
            self.algorithm_name = f"Orphan Nodes ({len(orphans)} found)"
            self.orphan_nodes = orphans
            self.algorithm_result = f"Found {len(orphans)} orphan nodes (no connections)"

    def set_traversal_start(self, value: str) -> None:
        self.traversal_start = value

    def set_path_source(self, value: str) -> None:
        self.path_source = value

    def set_path_target(self, value: str) -> None:
        self.path_target = value

    def set_ego_node(self, value: str) -> None:
        self.ego_node = value

    def set_ego_radius(self, value: str) -> None:
        try:
            self.ego_radius = int(value)
        except ValueError:
            pass


def _preview_content(content: dict) -> str:
    """Create a short preview of node content."""
    if not content:
        return "(empty)"
    # Try common keys first
    for key in ("title", "name", "summary", "text", "body"):
        if key in content:
            val = str(content[key])
            return val[:120] + "..." if len(val) > 120 else val
    # Fallback: first value
    first_val = str(next(iter(content.values())))
    return first_val[:120] + "..." if len(first_val) > 120 else first_val
