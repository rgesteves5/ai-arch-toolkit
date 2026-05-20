"""Database infrastructure for the Research Center app."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_conn_cache: sqlite3.Connection | None = None


def _db_path() -> Path:
    """Return the path to the SQLite database, ensuring parent dir exists."""
    p = Path.home() / ".research_center" / "research_center.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _get_conn() -> sqlite3.Connection:
    """Return a lazily-cached SQLite connection, initializing tables on first use."""
    global _conn_cache
    if _conn_cache is not None:
        return _conn_cache
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _conn_cache = conn
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            brief TEXT DEFAULT '',
            budget TEXT DEFAULT '1.00',
            max_cycles TEXT DEFAULT '3',
            grok_model TEXT DEFAULT 'grok-4-1-fast-reasoning',
            gemini_model TEXT DEFAULT 'gemini-3-flash',
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'new',
            wiki_node_count TEXT DEFAULT '0'
        );

        CREATE TABLE IF NOT EXISTS results (
            project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
            topic TEXT DEFAULT '',
            report TEXT DEFAULT '',
            total_cost REAL DEFAULT 0.0,
            cycles_completed INTEGER DEFAULT 0,
            budget_remaining REAL DEFAULT 0.0,
            wiki_node_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS phase_costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            cycle INTEGER NOT NULL,
            agent TEXT NOT NULL,
            cost REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pipeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            event_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS wikis (
            project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
            graph_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS manager_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_project_key
            ON manager_notes(project_id, key);
    """)
    conn.commit()
