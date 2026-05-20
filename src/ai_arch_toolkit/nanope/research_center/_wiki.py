"""Wiki — shared knowledge memory and tools for the research center."""

from __future__ import annotations

from collections.abc import Sequence

from ai_arch_toolkit.core._tools import ToolGroup
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.toolkit.memory._types import Node, NodeType, SearchResult
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore


def wiki_read_tools(store: GraphStore) -> ToolGroup:
    """Read-only tools for querying the wiki memory.

    Tools:
        - wiki_search: Search for knowledge nodes by keyword.
        - wiki_read: Read full content of a specific node.
        - wiki_categories: List all node types (categories) and their counts.
        - wiki_explore: Get a node and its connected neighbors.
        - wiki_find_duplicates: Check for existing nodes on a subject before creating.
    """

    @tool
    async def wiki_search(query: str, k: int = 5, category: str = "") -> str:
        """Search the wiki for knowledge nodes matching a query.

        Args:
            query: Search text to find related knowledge.
            k: Maximum number of results to return.
            category: Optional filter by node category (empty for all).
        """
        type_filter: NodeType | None = category if category else None
        results: Sequence[SearchResult] = await store.search(query, type=type_filter, k=k)
        if not results:
            return "No matching entries found in the wiki."
        lines: list[str] = []
        for r in results:
            text = _node_text(r.node)
            subject = r.node.content.get("subject", "")
            prefix = f"{subject}: " if subject else ""
            lines.append(
                f"[{r.node.type}] id={r.node.id} confidence={r.node.confidence:.1f}: "
                f"{prefix}{text[:120]}"
            )
        return "\n".join(lines)

    @tool
    async def wiki_read(node_id: str) -> str:
        """Read the full content of a wiki node by its ID.

        Args:
            node_id: The ID of the node to read.
        """
        node = await store.get(node_id)
        if node is None:
            return f"Node not found: {node_id}"
        lines = [
            f"[{node.type}] id={node.id}",
            f"  source: {node.source}",
            f"  confidence: {node.confidence}",
            f"  accessed: {node.access_count}x",
        ]
        # Show structured fields first
        content = node.content
        for key in ("title", "summary", "subject", "details"):
            if key in content:
                lines.append(f"  {key}: {content[key]}")
        # Show any extra keys beyond the structured fields
        structured = ("title", "summary", "subject", "details")
        extras = {k: v for k, v in content.items() if k not in structured}
        for key, val in extras.items():
            lines.append(f"  {key}: {val}")
        neighbors = await store.neighbors(node_id, depth=1)
        if neighbors:
            lines.append(f"Connected nodes ({len(neighbors)}):")
            for n in neighbors:
                lines.append(f"  [{n.type}] id={n.id}: {_node_text(n)[:80]}")
        return "\n".join(lines)

    @tool
    async def wiki_categories() -> str:
        """List all node categories in the wiki with their counts."""
        type_index = store._type_index
        if not type_index:
            return "Wiki is empty — no categories yet."
        lines = ["Wiki categories:"]
        for cat, ids in sorted(type_index.items()):
            lines.append(f"  {cat}: {len(ids)} nodes")
        total = sum(len(ids) for ids in type_index.values())
        lines.append(f"Total: {total} nodes")
        return "\n".join(lines)

    @tool
    async def wiki_explore(node_id: str, depth: int = 1) -> str:
        """Explore a node and its neighbors in the wiki graph.

        Args:
            node_id: The ID of the node to explore.
            depth: How many hops to traverse from the node.
        """
        node = await store.get(node_id)
        if node is None:
            return f"Node not found: {node_id}"
        lines = [f"Node [{node.type}] id={node.id}: {_node_text(node)[:100]}"]
        lines.append(f"  source={node.source}, confidence={node.confidence}")
        neighbors = await store.neighbors(node_id, depth=depth)
        if neighbors:
            lines.append(f"Neighbors ({len(neighbors)}):")
            for n in neighbors:
                lines.append(f"  [{n.type}] id={n.id}: {_node_text(n)[:80]}")
        else:
            lines.append("No connected neighbors.")
        return "\n".join(lines)

    @tool
    async def wiki_find_duplicates(subject: str) -> str:
        """Check for existing wiki nodes about a subject before creating new ones.

        Always call this before wiki_remember to avoid duplicates.

        Args:
            subject: The subject or topic to check for.
        """
        results: Sequence[SearchResult] = await store.search(subject, k=10)
        if not results:
            return f"No existing nodes about '{subject}' — safe to create."
        matches: list[str] = []
        for r in results:
            node_subject = r.node.content.get("subject", "")
            if node_subject and node_subject.lower() == subject.lower():
                matches.append(
                    f"  EXACT: [{r.node.type}] id={r.node.id}: {_node_text(r.node)[:100]}"
                )
            elif r.score > 0.5:
                matches.append(
                    f"  SIMILAR (score={r.score:.2f}): [{r.node.type}] id={r.node.id}: "
                    f"{_node_text(r.node)[:100]}"
                )
        if not matches:
            return f"No close matches for '{subject}' — safe to create."
        return (
            f"Found {len(matches)} potential duplicate(s) for '{subject}':\n"
            + "\n".join(matches)
            + "\nAvoid creating duplicates — only add if your content is distinct."
        )

    return ToolGroup(wiki_search, wiki_read, wiki_categories, wiki_explore, wiki_find_duplicates)


def wiki_write_tools(store: GraphStore) -> ToolGroup:
    """Write tools for adding and connecting knowledge in the wiki.

    Tools:
        - wiki_remember: Store a new knowledge node with structured fields.
        - wiki_connect: Create a relation edge between two nodes.
        - wiki_update_confidence: Update a node's confidence score.
    """

    @tool
    async def wiki_remember(
        title: str,
        summary: str,
        category: str = "fact",
        source: str = "researcher",
        subject: str = "",
        details: str = "",
    ) -> str:
        """Store a new knowledge entry in the wiki.

        Args:
            title: Short descriptive title for this knowledge entry.
            summary: Concise summary of the knowledge (1-2 sentences).
            category: Type of knowledge — use any fitting category such as fact,
                concept, definition, event, biography, law, theory, method, etc.
            source: How this was obtained (researcher, wikipedia, dictionary, web).
            subject: The main entity or topic this is about.
            details: Additional details, examples, or elaboration (optional).
        """
        content: dict[str, str] = {"title": title, "summary": summary}
        if subject:
            content["subject"] = subject
        if details:
            content["details"] = details
        node = Node(type=category, content=content, source=source)
        added = await store.add(node)
        return f"Stored [{added.type}] id={added.id}: {title}"

    @tool
    async def wiki_connect(
        source_id: str,
        target_id: str,
        relation: str,
    ) -> str:
        """Create a relation between two wiki nodes.

        Use specific relation types that describe the connection:
            - related_to: general topical relation
            - subtopic_of: X is a narrower topic within Y
            - supports: X provides evidence for Y
            - contradicts: X conflicts with Y
            - defines: X is a definition relevant to Y
            - discovered_by: X was discovered/created by Y
            - precedes: X comes before Y chronologically
            - follows: X comes after Y chronologically
            - generalizes: X is a broader version of Y
            - specializes: X is a more specific version of Y
            - derived_from: X is derived or adapted from Y
            - applied_to: X is applied in the context of Y
            - part_of: X is a component or part of Y
            - causes: X causes or leads to Y
            - enables: X enables or makes Y possible
        Create new relation types as needed to accurately describe connections.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            relation: Type of relation between the nodes.
        """
        src = await store.get(source_id)
        if src is None:
            return f"Source node not found: {source_id}"
        tgt = await store.get(target_id)
        if tgt is None:
            return f"Target node not found: {target_id}"
        edge = await store.connect(source_id, target_id, relation)
        return f"Connected {source_id} --[{edge.relation}]--> {target_id}"

    @tool
    async def wiki_update_confidence(node_id: str, confidence: float) -> str:
        """Update the confidence score of a wiki node.

        Args:
            node_id: The ID of the node to update.
            confidence: New confidence score (0.0 to 1.0).
        """
        confidence = max(0.0, min(1.0, confidence))
        updated = await store.update(node_id, confidence=confidence)
        if updated is None:
            return f"Node not found: {node_id}"
        return f"Updated confidence for {node_id} to {confidence:.2f}"

    return ToolGroup(wiki_remember, wiki_connect, wiki_update_confidence)


def wiki_analysis_tools(store: GraphStore) -> ToolGroup:
    """Analysis tools for reviewing wiki graph quality.

    Tools:
        - wiki_graph_stats: Get overall graph statistics.
        - wiki_find_orphans: List orphan nodes with no connections.
    """

    @tool
    async def wiki_graph_stats() -> str:
        """Get statistics about the wiki graph: nodes, edges, orphans, distributions."""
        count = await store.count()
        if count == 0:
            return "Wiki is empty — no nodes."

        lines = [f"Total nodes: {count}"]

        # Category distribution
        type_index = store._type_index
        if type_index:
            lines.append("Category distribution:")
            for cat, ids in sorted(type_index.items(), key=lambda x: -len(x[1])):
                pct = len(ids) / count * 100
                lines.append(f"  {cat}: {len(ids)} ({pct:.0f}%)")

        # Edge stats
        backend = store._backend
        edge_count = 0
        relation_counts: dict[str, int] = {}
        orphan_count = 0
        if hasattr(backend, "_graph"):
            edge_count = backend._graph.number_of_edges()
            lines.append(f"Total edges: {edge_count}")
            for _, _, data in backend._graph.edges(data=True):
                edge_obj = data.get("edge")
                if edge_obj:
                    rel = getattr(edge_obj, "relation", "unknown")
                    relation_counts[rel] = relation_counts.get(rel, 0) + 1
            if relation_counts:
                lines.append("Edge relation distribution:")
                for rel, cnt in sorted(relation_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  {rel}: {cnt}")
            # Orphan count
            for node_id in backend._graph.nodes():
                if backend._graph.degree(node_id) == 0:
                    orphan_count += 1
            orphan_pct = orphan_count / count * 100 if count else 0
            lines.append(f"Orphan nodes: {orphan_count} ({orphan_pct:.0f}%)")

        return "\n".join(lines)

    @tool
    async def wiki_find_orphans(limit: int = 20) -> str:
        """List wiki nodes that have no connections (orphans).

        Args:
            limit: Maximum number of orphan nodes to return.
        """
        backend = store._backend
        orphans: list[str] = []
        if hasattr(backend, "_graph"):
            for node_id in backend._graph.nodes():
                if backend._graph.degree(node_id) == 0:
                    node = await backend.get_node(node_id)
                    if node:
                        orphans.append(f"[{node.type}] id={node.id}: {_node_text(node)[:100]}")
                    if len(orphans) >= limit:
                        break

        if not orphans:
            return "No orphan nodes found — all nodes are connected."
        return f"Orphan nodes ({len(orphans)}):\n" + "\n".join(orphans)

    return ToolGroup(wiki_graph_stats, wiki_find_orphans)


def wiki_notes_tools(project_id: str) -> ToolGroup:
    """Manager notes tools backed by SQLite.

    Tools:
        - notes_write: Write or update a note.
        - notes_read: Read one or all notes.
        - notes_list: List all note keys.
    """

    @tool
    def notes_write(key: str, content: str) -> str:
        """Write or update a persistent note. Notes survive across pipeline cycles.

        Args:
            key: Short identifier for the note (e.g. 'todo', 'gaps', 'strategy').
            content: The note content.
        """
        from ai_arch_toolkit.nanope.research_center._db import _get_conn

        conn = _get_conn()
        conn.execute(
            """INSERT INTO manager_notes (project_id, key, content)
               VALUES (?, ?, ?)
               ON CONFLICT(project_id, key) DO UPDATE SET
                   content = excluded.content,
                   updated_at = datetime('now')""",
            (project_id, key, content),
        )
        conn.commit()
        return f"Note '{key}' saved."

    @tool
    def notes_read(key: str = "") -> str:
        """Read a specific note or all notes.

        Args:
            key: Note key to read. Leave empty to read all notes.
        """
        from ai_arch_toolkit.nanope.research_center._db import _get_conn

        conn = _get_conn()
        if key:
            row = conn.execute(
                "SELECT key, content, updated_at FROM manager_notes "
                "WHERE project_id = ? AND key = ?",
                (project_id, key),
            ).fetchone()
            if not row:
                return f"No note found with key '{key}'."
            return f"[{row['key']}] (updated {row['updated_at']}):\n{row['content']}"
        else:
            rows = conn.execute(
                "SELECT key, content, updated_at FROM manager_notes "
                "WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
            if not rows:
                return "No notes yet."
            lines: list[str] = []
            for row in rows:
                lines.append(f"[{row['key']}] (updated {row['updated_at']}):")
                lines.append(f"  {row['content'][:200]}")
            return "\n".join(lines)

    @tool
    def notes_list() -> str:
        """List all note keys for this project."""
        from ai_arch_toolkit.nanope.research_center._db import _get_conn

        conn = _get_conn()
        rows = conn.execute(
            "SELECT key, updated_at FROM manager_notes WHERE project_id = ? ORDER BY key",
            (project_id,),
        ).fetchall()
        if not rows:
            return "No notes yet."
        return "Notes:\n" + "\n".join(f"  {r['key']} (updated {r['updated_at']})" for r in rows)

    return ToolGroup(notes_write, notes_read, notes_list)


def _node_text(node: Node) -> str:
    """Extract readable text from a node's content."""
    content = node.content
    # Prefer structured title — summary format
    title = content.get("title", "")
    summary = content.get("summary", "")
    if title and summary:
        return f"{title} — {summary}"
    if title:
        return title
    # Fallback for old {text, subject} format nodes
    parts = [str(v) for v in content.values() if isinstance(v, str)]
    return " ".join(parts)
