"""Assemble an Agent from a resolved manifest plus application runtime objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import OutputSchema
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._agent import Agent
from ai_arch_toolkit.toolkit.agents._builders import FlowStrategy, get_strategy
from ai_arch_toolkit.toolkit.agents._manifest import ResolvedAgentManifest

__all__ = ["agent_from_manifest"]


def agent_from_manifest(
    manifest: ResolvedAgentManifest,
    llm: LLM,
    tools: ToolGroup | None = None,
    *,
    llm_factory: Callable[[str, Mapping[str, Any]], LLM] | None = None,
    deps: Mapping[str, Any] | None = None,
    system: str | None = None,
    output_schema: OutputSchema | type | None = None,
    policy: Policy | None = None,
) -> Agent:
    """Build an Agent from a manifest, resolving per-phase models via ``llm_factory``.

    Per-phase ``model`` configs under ``strategy.phases`` are validated data, not
    runtime objects; ``llm_factory(phase, model_config)`` turns each into an LLM,
    bound as the canonical ``<phase>_llm`` dep. An explicit ``deps`` entry for a
    phase wins over the factory, and a declared phase model with neither is an
    error — a manifest's model choice must not be silently ignored. ``system``,
    ``output_schema``, and ``policy`` pass through to ``reasoning_spec``.

    The spec and strategy validate *before* the factory runs, so an invalid
    manifest cannot leak half-created clients. LLMs the factory creates belong
    to the caller — retain references in the factory closure and ``close()``
    them when the agent is done.
    """
    spec = manifest.reasoning_spec(system=system, output_schema=output_schema, policy=policy)
    builder = get_strategy(spec.strategy)
    if isinstance(builder, FlowStrategy):
        builder.validate_spec(spec)
    explicit = dict(deps or {})
    resolved: dict[str, Any] = {}
    for phase, model_config in manifest.phase_models().items():
        key = f"{phase}_llm"
        if key in explicit:
            continue
        if (
            isinstance(builder, FlowStrategy)
            and builder.allowed_deps is not None
            and key not in builder.allowed_deps
        ):
            raise ValueError(
                f"strategy {spec.strategy!r} does not accept an LLM binding for "
                f"phase {phase!r} ({key!r} is not a recognized dep)"
            )
        if llm_factory is None:
            raise ValueError(
                f"manifest declares a model for phase {phase!r}; pass llm_factory "
                f"or an explicit {key!r} dep"
            )
        resolved[key] = llm_factory(phase, model_config)
    return Agent(spec, llm, tools, deps={**resolved, **explicit})
