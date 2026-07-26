"""47 — Per-phase agent configuration.

Multi-phase strategies accept per-phase overrides through two buckets:
runtime objects (LLMs, tools) as ``deps`` and prompts as ``knobs``. The same
capability is declarative in agent manifests via ``strategy.phases``.

Part 1 wires a cheap planner model and a custom planner prompt in code.
Part 2 declares the same thing in a manifest and assembles it with
``agent_from_manifest``.
"""

import json
import tempfile
from pathlib import Path

from ai_arch_toolkit import LLM, ToolGroup, tool
from ai_arch_toolkit.toolkit.agents import (
    Agent,
    ReasoningSpec,
    agent_from_manifest,
    load_agent_manifest,
)


@tool
def search(query: str) -> str:
    """Search for information on a topic.

    Args:
        query: The search query.
    """
    return f"Search result for '{query}': relevant information found."


tools = ToolGroup(search)
llm = LLM("claude-sonnet-4-20250514")  # default for every phase
haiku = LLM("claude-haiku-4-5-20251001")  # cheap planner

# --- Part 1: per-phase overrides in code -----------------------------------
# deps carry runtime objects per phase (planner_llm, executor_tools, ...);
# knobs carry per-phase prompts (planner_system, solver_system, ...). The
# "{tools}" token is the only substitution the framework performs: it renders
# the executor's tool catalog exactly where you place it — a prompt without
# the token is never modified.
spec = ReasoningSpec(
    strategy="plan_execute",
    knobs={
        "planner_system": (
            "Plan in at most three numbered steps.\nOnly rely on these tools:\n{tools}"
        ),
        "max_replans": 0,
    },
)
agent = Agent(spec, llm, tools, deps={"planner_llm": haiku})

result = agent.run_sync("Find one interesting fact about the Atacama desert.")
print("Part 1 answer:", result.text)
print(f"Part 1 cost: ${result.cost:.4f}")

# --- Part 2: the same, declaratively ---------------------------------------
# Phase prompts live in the manifest (inline or via system_file, both covered
# by the fingerprint); phase model configs are validated data the application
# resolves — the loader never constructs LLMs.
manifest_data = {
    "version": 1,
    "id": "examples.per-phase",
    "strategy": {
        "name": "plan_execute",
        "knobs": {"max_replans": 0},
        "phases": {
            "planner": {
                "system": ("Plan in at most three numbered steps.\nTools:\n{tools}"),
                "model": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            },
        },
    },
}

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "per_phase.agent.json"
    path.write_text(json.dumps(manifest_data), encoding="utf-8")

    manifest = load_agent_manifest(path)
    print("Fingerprint:", manifest.fingerprint)

    agent = agent_from_manifest(
        manifest,
        llm,
        tools,
        llm_factory=lambda phase, cfg: LLM(cfg["model"]),
    )
    result = agent.run_sync("Find one interesting fact about the Mariana Trench.")
    print("Part 2 answer:", result.text)
    print(f"Part 2 cost: ${result.cost:.4f}")
