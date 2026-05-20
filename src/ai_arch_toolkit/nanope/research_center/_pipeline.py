"""Pipeline — budget-controlled research loop orchestrating all agents."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.nanope.research_center._agents import (
    linker_agent,
    manager_agent,
    researcher_agent,
    review_agent,
    writer_agent,
)
from ai_arch_toolkit.toolkit.agents.flows._generate_review import generate_review_initial_state
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore

logger = logging.getLogger(__name__)

type PipelineEventType = Literal[
    "pipeline_start",
    "cycle_start",
    "agent_start",
    "agent_end",
    "manager_decision",
    "pipeline_end",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineEvent:
    """Structured event emitted during pipeline execution."""

    type: PipelineEventType
    agent: str = ""
    cycle: int = 0
    max_cycles: int = 0
    cost: float = 0.0
    total_spent: float = 0.0
    budget: float = 0.0
    decision: str = ""
    directives: dict[str, str] = field(default_factory=dict)
    message: str = ""


_DEFAULT_GROK = "grok-4-1-fast-reasoning"
_DEFAULT_GEMINI = "gemini-3-flash"


@dataclass(slots=True, kw_only=True)
class PipelineResult:
    """Result of a complete research pipeline run."""

    topic: str
    report: str
    total_cost: float
    cycles_completed: int
    budget_remaining: float
    wiki_node_count: int
    phase_costs: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Wiki project persistence
# ---------------------------------------------------------------------------


async def create_wiki() -> GraphStore:
    """Create a fresh wiki GraphStore backed by NetworkX."""
    from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

    return GraphStore(NetworkXBackend())


async def load_wiki(path: str | Path) -> GraphStore:
    """Load an existing wiki project from a JSON file.

    Args:
        path: Path to the saved wiki JSON file.
    """
    from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

    return await GraphStore.load(path, NetworkXBackend())


async def save_wiki(wiki: GraphStore, path: str | Path) -> None:
    """Save the wiki project to a JSON file for later reuse.

    Args:
        wiki: The GraphStore to persist.
        path: Destination file path (will be created/overwritten).
    """
    await wiki.save(path)


def create_wiki_sync() -> GraphStore:
    """Synchronous wrapper for create_wiki."""
    from ai_arch_toolkit.core._sync import _run_sync

    return _run_sync(create_wiki())


def load_wiki_sync(path: str | Path) -> GraphStore:
    """Synchronous wrapper for load_wiki."""
    from ai_arch_toolkit.core._sync import _run_sync

    return _run_sync(load_wiki(path))


def save_wiki_sync(wiki: GraphStore, path: str | Path) -> None:
    """Synchronous wrapper for save_wiki."""
    from ai_arch_toolkit.core._sync import _run_sync

    _run_sync(save_wiki(wiki, path))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    topic: str,
    wiki: GraphStore,
    *,
    owner_brief: str = "",
    budget: float = 1.0,
    max_cycles: int = 3,
    grok_model: str = _DEFAULT_GROK,
    gemini_model: str = _DEFAULT_GEMINI,
    max_gen_iterations: int = 8,
    max_review_iterations: int = 3,
    project_id: str = "",
    on_event: Callable[[PipelineEvent], Awaitable[None]] | None = None,
) -> PipelineResult:
    """Run the full research pipeline with Reviewer in the loop.

    The owner controls the research through ``owner_brief`` — specifying what to
    research, the desired report style, depth, focus areas, audience, etc.
    The Manager translates the owner's brief into actionable plans for each agent.

    Args:
        topic: The research topic.
        wiki: Shared GraphStore (new or previously saved project).
        owner_brief: The owner's instructions — what to research, report format,
            focus areas, audience, depth, constraints, etc. This is passed to the
            Manager who translates it into plans for each agent.
        budget: Maximum total cost in USD.
        max_cycles: Maximum number of research cycles.
        grok_model: Model for researcher, linker, manager, and reviewer agents.
        gemini_model: Model for the writer agent.
        max_gen_iterations: Max ReAct iterations for generate phases.
        max_review_iterations: Max ReAct iterations for review phases.
        project_id: Optional project ID for manager notes persistence.
        on_event: Optional async callback for structured pipeline events.
    """
    grok_llm = LLM(grok_model)
    gemini_llm = LLM(gemini_model)

    spent = 0.0
    phase_costs: list[dict[str, Any]] = []
    cycles = 0
    report = ""
    directives = _empty_directives()
    reviewer_report = ""

    async def _emit(event: PipelineEvent) -> None:
        if on_event is not None:
            await on_event(event)

    # Build the context string the manager sees every time
    owner_context = _build_owner_context(topic, owner_brief)

    await _emit(
        PipelineEvent(
            type="pipeline_start",
            max_cycles=max_cycles,
            budget=budget,
            message=f"Starting research pipeline for: {topic}",
        )
    )

    # --- First cycle: Manager sets the initial plan ---
    if _has_budget(spent, budget):
        logger.info("=== Initial planning | spent=$%.4f / $%.4f ===", spent, budget)
        await _emit(
            PipelineEvent(
                type="agent_start",
                agent="manager",
                cycle=0,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                message="Initial planning",
            )
        )
        spent, directives = await _run_manager(
            grok_llm,
            wiki,
            owner_context,
            spent,
            budget,
            0,
            max_cycles,
            max_gen_iterations,
            max_review_iterations,
            phase_costs,
            project_id=project_id,
            on_event=on_event,
        )

    for cycle in range(max_cycles):
        cycles = cycle + 1

        logger.info("=== Cycle %d/%d | spent=$%.4f / $%.4f ===", cycles, max_cycles, spent, budget)
        await _emit(
            PipelineEvent(
                type="cycle_start",
                cycle=cycles,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                message=f"Cycle {cycles}/{max_cycles}",
            )
        )

        # --- Researcher (with manager's plan) ---
        if not _has_budget(spent, budget):
            logger.info("Budget exhausted before researcher.")
            break

        researcher_plan = directives["researcher_plan"]
        research_task = f"Research the topic: {topic}\n\nManager instructions:\n{researcher_plan}"
        logger.info("[Researcher] %s", researcher_plan[:100])
        await _emit(
            PipelineEvent(
                type="agent_start",
                agent="researcher",
                cycle=cycles,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                message=researcher_plan[:100],
            )
        )
        r_flow = researcher_agent(
            grok_llm,
            wiki,
            max_gen_iterations=max_gen_iterations,
            max_review_iterations=max_review_iterations,
        )
        r_state = State(operational=generate_review_initial_state(research_task))
        r_result = await r_flow.run(r_state)
        r_cost = r_result.total_cost
        spent += r_cost
        phase_costs.append({"cycle": cycles, "agent": "researcher", "cost": r_cost})
        logger.info("[Researcher] Done. Cost=$%.4f", r_cost)
        await _emit(
            PipelineEvent(
                type="agent_end",
                agent="researcher",
                cycle=cycles,
                max_cycles=max_cycles,
                cost=r_cost,
                total_spent=spent,
                budget=budget,
                message="Researcher done",
            )
        )

        # --- Linker (with manager's plan) ---
        if not _has_budget(spent, budget):
            logger.info("Budget exhausted before linker.")
            break

        linker_plan = directives["linker_plan"]
        logger.info("[Linker] %s", linker_plan[:100])
        await _emit(
            PipelineEvent(
                type="agent_start",
                agent="linker",
                cycle=cycles,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                message=linker_plan[:100],
            )
        )
        l_flow = linker_agent(
            grok_llm,
            wiki,
            max_gen_iterations=max_gen_iterations,
            max_review_iterations=max_review_iterations,
        )
        l_state = State(
            operational=generate_review_initial_state(
                f"Analyze and connect knowledge about: {topic}\n\n"
                f"Manager instructions:\n{linker_plan}"
            )
        )
        l_result = await l_flow.run(l_state)
        l_cost = l_result.total_cost
        spent += l_cost
        phase_costs.append({"cycle": cycles, "agent": "linker", "cost": l_cost})
        logger.info("[Linker] Done. Cost=$%.4f", l_cost)
        await _emit(
            PipelineEvent(
                type="agent_end",
                agent="linker",
                cycle=cycles,
                max_cycles=max_cycles,
                cost=l_cost,
                total_spent=spent,
                budget=budget,
                message="Linker done",
            )
        )

        # --- Reviewer (graph quality check) ---
        if not _has_budget(spent, budget):
            logger.info("Budget exhausted before reviewer.")
            break

        reviewer_plan = directives.get("reviewer_plan", "Review graph quality.")
        logger.info("[Reviewer] %s", reviewer_plan[:100])
        await _emit(
            PipelineEvent(
                type="agent_start",
                agent="reviewer",
                cycle=cycles,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                message=reviewer_plan[:100],
            )
        )
        rev_flow = review_agent(
            grok_llm,
            wiki,
            max_gen_iterations=max_gen_iterations,
            max_review_iterations=max_review_iterations,
        )
        rev_task = (
            f"Review the wiki graph quality for topic: {topic}\n\n"
            f"Manager instructions:\n{reviewer_plan}"
        )
        rev_state = State(operational=generate_review_initial_state(rev_task))
        rev_result = await rev_flow.run(rev_state)
        rev_cost = rev_result.total_cost
        spent += rev_cost
        phase_costs.append({"cycle": cycles, "agent": "reviewer", "cost": rev_cost})
        reviewer_report = rev_state.get("answer", rev_state.get("last_answer", ""))
        logger.info("[Reviewer] Done. Cost=$%.4f", rev_cost)
        await _emit(
            PipelineEvent(
                type="agent_end",
                agent="reviewer",
                cycle=cycles,
                max_cycles=max_cycles,
                cost=rev_cost,
                total_spent=spent,
                budget=budget,
                message=f"Reviewer done. Report:\n{reviewer_report[:200]}",
            )
        )

        # --- Writer (produce/update report each cycle) ---
        if not _has_budget(spent, budget):
            logger.info("Budget exhausted before writer.")
            break

        writer_plan = directives["writer_plan"]
        writer_strategy = directives["writer_strategy"]
        logger.info("[Writer] Strategy: %s | %s", writer_strategy, writer_plan[:100])
        await _emit(
            PipelineEvent(
                type="agent_start",
                agent="writer",
                cycle=cycles,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                message=f"Strategy: {writer_strategy} | {writer_plan[:100]}",
            )
        )
        w_flow = writer_agent(
            gemini_llm,
            wiki,
            max_gen_iterations=max_gen_iterations,
            max_review_iterations=max_review_iterations + 1,
        )
        writer_task = (
            f"Write a comprehensive report on: {topic}\n\n"
            f"Manager instructions:\n{writer_plan}\n\n"
            f"Use the reasoning strategy '{writer_strategy}' to structure your work. "
            f"Call reasoning_strategy with ['{writer_strategy}'] to get the full strategy."
        )
        w_state = State(operational=generate_review_initial_state(writer_task))
        w_result = await w_flow.run(w_state)
        w_cost = w_result.total_cost
        spent += w_cost
        phase_costs.append({"cycle": cycles, "agent": "writer", "cost": w_cost})
        report = w_state.get("answer", w_state.get("last_answer", ""))
        logger.info("[Writer] Done. Cost=$%.4f", w_cost)
        await _emit(
            PipelineEvent(
                type="agent_end",
                agent="writer",
                cycle=cycles,
                max_cycles=max_cycles,
                cost=w_cost,
                total_spent=spent,
                budget=budget,
                message="Writer done",
            )
        )

        # --- Manager (re-assess after full cycle) ---
        if not _has_budget(spent, budget):
            logger.info("Budget exhausted before manager.")
            break

        await _emit(
            PipelineEvent(
                type="agent_start",
                agent="manager",
                cycle=cycles,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                message="Re-assessing research quality",
            )
        )
        spent, directives = await _run_manager(
            grok_llm,
            wiki,
            owner_context,
            spent,
            budget,
            cycles,
            max_cycles,
            max_gen_iterations,
            max_review_iterations,
            phase_costs,
            reviewer_report=reviewer_report,
            project_id=project_id,
            on_event=on_event,
        )

    if not report:
        report = "(Budget exhausted before writing phase)"

    node_count = await wiki.count()

    await _emit(
        PipelineEvent(
            type="pipeline_end",
            cycle=cycles,
            max_cycles=max_cycles,
            total_spent=spent,
            budget=budget,
            message=f"Pipeline complete. {node_count} wiki nodes, ${spent:.4f} spent.",
        )
    )

    return PipelineResult(
        topic=topic,
        report=report,
        total_cost=spent,
        cycles_completed=cycles,
        budget_remaining=budget - spent,
        wiki_node_count=node_count,
        phase_costs=phase_costs,
    )


async def _run_manager(
    llm: LLM,
    wiki: GraphStore,
    owner_context: str,
    spent: float,
    budget: float,
    cycle: int,
    max_cycles: int,
    max_gen_iterations: int,
    max_review_iterations: int,
    phase_costs: list[dict[str, Any]],
    *,
    reviewer_report: str = "",
    project_id: str = "",
    on_event: Callable[[PipelineEvent], Awaitable[None]] | None = None,
) -> tuple[float, dict[str, str]]:
    """Run the manager agent and return (updated_spent, parsed_directives)."""
    logger.info("[Manager] Reviewing research quality...")
    m_flow = manager_agent(
        llm,
        wiki,
        project_id=project_id,
        max_gen_iterations=max_gen_iterations,
        max_review_iterations=max_review_iterations,
    )
    wiki_summary = await _build_wiki_summary(wiki, reviewer_report=reviewer_report)
    m_state = State(
        operational=generate_review_initial_state(
            f"{owner_context}\n\n"
            f"Budget remaining: ${budget - spent:.4f}. Cycle {cycle + 1}/{max_cycles}.\n"
            f"{wiki_summary}\n\n"
            f"IMPORTANT: The Writer can ONLY write about knowledge that exists in the wiki. "
            f"If the wiki is empty or missing key topics, the Writer cannot produce a good "
            f"report. Ensure sufficient research is done before planning the Writer's work."
        )
    )
    m_result = await m_flow.run(m_state)
    m_cost = m_result.total_cost
    spent += m_cost
    phase_costs.append({"cycle": cycle + 1, "agent": "manager", "cost": m_cost})
    logger.info("[Manager] Done. Cost=$%.4f", m_cost)

    manager_answer = m_state.get("answer", m_state.get("last_answer", ""))
    directives = _parse_manager_directives(manager_answer)
    logger.info("[Manager] Decision: %s", directives["decision"])

    if on_event is not None:
        await on_event(
            PipelineEvent(
                type="agent_end",
                agent="manager",
                cycle=cycle,
                max_cycles=max_cycles,
                cost=m_cost,
                total_spent=spent,
                budget=budget,
                message="Manager done",
            )
        )
        await on_event(
            PipelineEvent(
                type="manager_decision",
                agent="manager",
                cycle=cycle,
                max_cycles=max_cycles,
                total_spent=spent,
                budget=budget,
                decision=directives["decision"],
                directives=directives,
                message=f"Decision: {directives['decision']}",
            )
        )

    return spent, directives


def run_pipeline_sync(
    topic: str,
    wiki: GraphStore,
    **kwargs: Any,
) -> PipelineResult:
    """Synchronous wrapper for run_pipeline."""
    from ai_arch_toolkit.core._sync import _run_sync

    return _run_sync(run_pipeline(topic, wiki, **kwargs))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_owner_context(topic: str, owner_brief: str) -> str:
    """Build the static owner context (topic + brief). Wiki state is added dynamically."""
    parts = [f"Research topic: {topic}"]
    if owner_brief:
        parts.append(f"\nOwner's instructions:\n{owner_brief}")
    return "\n".join(parts)


async def _build_wiki_summary(
    wiki: GraphStore,
    *,
    reviewer_report: str = "",
) -> str:
    """Build a summary of the current wiki state for the manager."""
    count = await wiki.count()
    if count == 0:
        return "\nWiki state: EMPTY — no knowledge has been gathered yet."

    lines = [f"\nWiki state: {count} knowledge nodes."]

    # Category breakdown with percentages
    if hasattr(wiki, "_type_index") and wiki._type_index:
        lines.append("Categories:")
        for cat, ids in sorted(wiki._type_index.items(), key=lambda x: -len(x[1])):
            pct = len(ids) / count * 100
            lines.append(f"  {cat}: {len(ids)} ({pct:.0f}%)")

    # Edge stats
    backend = wiki._backend
    if hasattr(backend, "_graph"):
        edge_count = backend._graph.number_of_edges()
        lines.append(f"Edges: {edge_count}")

        # Relation distribution
        relation_counts: dict[str, int] = {}
        for _, _, data in backend._graph.edges(data=True):
            edge_obj = data.get("edge")
            if edge_obj:
                rel = getattr(edge_obj, "relation", "unknown")
                relation_counts[rel] = relation_counts.get(rel, 0) + 1
        if relation_counts:
            lines.append("Edge relations:")
            for rel, cnt in sorted(relation_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {rel}: {cnt}")

        # Orphan count
        orphan_count = sum(1 for n in backend._graph.nodes() if backend._graph.degree(n) == 0)
        orphan_pct = orphan_count / count * 100 if count else 0
        lines.append(f"Orphan nodes: {orphan_count} ({orphan_pct:.0f}%)")

    # Top subjects
    subject_counts: dict[str, int] = {}
    all_nodes = await wiki.list(limit=500)
    for node in all_nodes:
        subj = node.content.get("subject", "")
        if subj:
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
    if subject_counts:
        top_subjects = sorted(subject_counts.items(), key=lambda x: -x[1])[:10]
        lines.append("Top subjects:")
        for subj, cnt in top_subjects:
            lines.append(f"  {subj}: {cnt}")

    # Reviewer report
    if reviewer_report:
        lines.append(f"\nReviewer report:\n{reviewer_report}")

    return "\n".join(lines)


def _has_budget(spent: float, budget: float) -> bool:
    return spent < budget


def _empty_directives() -> dict[str, str]:
    """Default directives when no manager has run yet."""
    return {
        "decision": "RESEARCH_MORE",
        "researcher_plan": "Research the topic broadly — cover key subtopics and definitions.",
        "linker_plan": "Connect related concepts and establish category hierarchies.",
        "reviewer_plan": (
            "Review graph quality — check orphans, duplicates, and category diversity."
        ),
        "writer_plan": "Write a structured report organized by themes.",
        "writer_strategy": "synthesis",
    }


def _parse_manager_directives(text: str) -> dict[str, str]:
    """Parse the manager's structured output into directives for each agent."""
    directives = _empty_directives()

    # Parse DECISION
    upper = text.upper()
    if "READY_TO_WRITE" in upper:
        directives["decision"] = "READY_TO_WRITE"
    elif "DONE" in upper and "DECISION" in upper:
        directives["decision"] = "DONE"
    else:
        directives["decision"] = "RESEARCH_MORE"

    # Parse sections
    directives["researcher_plan"] = _extract_section(text, "RESEARCHER_PLAN")
    directives["linker_plan"] = _extract_section(text, "LINKER_PLAN")
    directives["reviewer_plan"] = _extract_section(text, "REVIEWER_PLAN")
    directives["writer_plan"] = _extract_section(text, "WRITER_PLAN")

    # Parse WRITER_STRATEGY (single line value)
    strategy = _extract_section(text, "WRITER_STRATEGY")
    if strategy and strategy.lower() != "n/a":
        # Clean up — take just the first word/phrase (strategy name)
        directives["writer_strategy"] = strategy.split("\n")[0].strip().strip("'\"")

    return directives


def _extract_section(text: str, header: str) -> str:
    """Extract content between a header and the next header or end of text."""
    # Look for "HEADER:" or "HEADER:\n"
    markers = [f"{header}:", f"{header}:\n", f"**{header}**:", f"**{header}:**"]
    start = -1
    for marker in markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            start = idx + len(marker)
            break

    if start == -1:
        return ""

    # Find the next section header (all-caps word followed by colon)
    remaining = text[start:]
    end_markers = [
        "COVERAGE:",
        "DECISION:",
        "RESEARCHER_PLAN:",
        "LINKER_PLAN:",
        "REVIEWER_PLAN:",
        "WRITER_PLAN:",
        "WRITER_STRATEGY:",
    ]
    end = len(remaining)
    for marker in end_markers:
        idx = remaining.upper().find(marker)
        if idx > 0:
            end = min(end, idx)

    return remaining[:end].strip()
