"""Research Center — multi-agent research pipeline with shared wiki memory."""

from __future__ import annotations

from ai_arch_toolkit.nanope.research_center._agents import (
    linker_agent,
    manager_agent,
    researcher_agent,
    review_agent,
    writer_agent,
)
from ai_arch_toolkit.nanope.research_center._pipeline import (
    PipelineEvent,
    PipelineResult,
    create_wiki,
    create_wiki_sync,
    load_wiki,
    load_wiki_sync,
    run_pipeline,
    run_pipeline_sync,
    save_wiki,
    save_wiki_sync,
)
from ai_arch_toolkit.nanope.research_center._wiki import (
    wiki_analysis_tools,
    wiki_notes_tools,
    wiki_read_tools,
    wiki_write_tools,
)
from ai_arch_toolkit.nanope.research_center.reasoning import (
    load_reasoning_systems,
    make_reasoning_tool,
)

__all__ = [
    "PipelineEvent",
    "PipelineResult",
    "create_wiki",
    "create_wiki_sync",
    "linker_agent",
    "load_reasoning_systems",
    "load_wiki",
    "load_wiki_sync",
    "make_reasoning_tool",
    "manager_agent",
    "researcher_agent",
    "review_agent",
    "run_pipeline",
    "run_pipeline_sync",
    "save_wiki",
    "save_wiki_sync",
    "wiki_analysis_tools",
    "wiki_notes_tools",
    "wiki_read_tools",
    "wiki_write_tools",
    "writer_agent",
]
