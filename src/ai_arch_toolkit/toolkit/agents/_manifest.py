"""Strict, file-backed configuration for public configurable agents.

The loader owns generic configuration mechanics only: version checks, inheritance,
profiles, relative paths, override governance, provenance, and deterministic
fingerprints. Applications may keep domain registry ids in the documented extension
fields and turn the resolved strategy section into a :class:`ReasoningSpec`.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ai_arch_toolkit.core import OutputSchema, Policy
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.budget import BudgetPolicy

__all__ = [
    "AgentManifestCycleError",
    "AgentManifestError",
    "AgentOverrideError",
    "ResolvedAgentManifest",
    "load_agent_manifest",
]

_SUFFIXES = (".agent.yaml", ".agent.yml", ".agent.json", ".agent.toml")
_RENDERED_SUFFIXES = (".prompt.yaml", ".prompt.yml", ".prompt.json", ".prompt.toml", *_SUFFIXES)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "description",
        "extends",
        "id",
        "limits",
        "metadata",
        "model",
        "output",
        "override_policy",
        "phase",
        "profiles",
        "prompts",
        "result_adapter",
        "strategy",
        "tools",
        "version",
    }
)
_PROFILE_FIELDS = _TOP_LEVEL_FIELDS - {"extends", "id", "profiles", "version"}
_STRATEGY_FIELDS = frozenset(
    {
        "final_answer_hint",
        "knobs",
        "llm_kwargs",
        "max_iterations",
        "name",
        "parallel_tool_calls",
        "phases",
        "show_turn_counter",
        "strip_tools_on_final",
        "system",
        "timeout",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "base_url",
        "max_tokens",
        "model",
        "profile",
        "provider",
        "structured_output_mode",
        "temperature",
    }
)
_PHASE_FIELDS = frozenset({"model", "system", "system_file"})
_PROMPT_FIELDS = frozenset(
    {
        "input_adapter",
        "request_template",
        "request_variables",
        "system",
        "system_manifest",
        "user",
    }
)
_REQUEST_VARIABLE_FIELDS = frozenset({"optional", "required"})
_OUTPUT_FIELDS = frozenset({"schema"})
_TOOLS_FIELDS = frozenset({"factory", "manifest"})
_LIMIT_FIELDS = frozenset(
    {
        "max_cost",
        "max_input_tokens",
        "max_llm_calls",
        "max_output_tokens",
        "max_total_tokens",
        "max_tool_calls",
        "max_wall_s",
        "reserve",
        "timeout_seconds",
        "unpriced",
    }
)
_OVERRIDE_FIELDS = frozenset({"allow", "deny"})
_PATH_FIELDS = (
    ("prompts", "request_template"),
    ("prompts", "system_manifest"),
    ("tools", "manifest"),
)
_SECRET_NAMES = frozenset({"api_key", "apikey", "authorization", "password", "secret", "token"})
_SECRET_SUFFIXES = ("_key", "_password", "_secret", "_token")


class AgentManifestError(ValueError):
    """An agent manifest could not be loaded or validated."""


class AgentManifestCycleError(AgentManifestError):
    """Agent-manifest inheritance contains a cycle."""


class AgentOverrideError(AgentManifestError):
    """A requested override is unknown, denied, or malformed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedAgentManifest:
    """A fully inherited, profiled, overridden, and fingerprinted manifest."""

    source: Path
    data: Mapping[str, Any]
    fingerprint: str
    sources: tuple[Path, ...]
    source_fingerprints: Mapping[str, str]
    referenced_fingerprints: Mapping[str, str]
    allowed_roots: tuple[Path, ...]
    profile: str | None = None

    @property
    def id(self) -> str | None:
        """Configured agent id, or ``None`` for a reusable profile manifest."""
        value = self.data.get("id")
        return value if isinstance(value, str) else None

    @property
    def version(self) -> int:
        """Manifest format version."""
        return int(self.data["version"])

    def as_dict(self) -> dict[str, Any]:
        """Return a detached mutable copy of the resolved configuration."""
        return _thaw(self.data)

    def reasoning_spec(
        self,
        *,
        system: str | None = None,
        output_schema: OutputSchema | type | None = None,
        policy: Policy | None = None,
    ) -> ReasoningSpec:
        """Build the public runtime spec from the manifest's strategy and limits.

        Provider/model construction, prompt rendering, tool factories, and schema-id
        registries remain application concerns. ``system`` and ``output_schema`` let
        an application supply their already-resolved runtime values.

        Per-phase prompts declared under ``strategy.phases`` fold into the spec's
        canonical ``<phase>_system`` knobs; ``system_file`` content is read here,
        at bridge time, and re-verified against the load-time fingerprint — a
        file that changed since the manifest was loaded raises instead of running
        unaudited content.
        """
        strategy = _mapping(self.data.get("strategy", {}), "strategy")
        knobs = dict(_mapping(strategy.get("knobs", {}), "strategy.knobs"))
        for field in (
            "final_answer_hint",
            "parallel_tool_calls",
            "show_turn_counter",
            "strip_tools_on_final",
        ):
            if field in strategy:
                knobs[field] = strategy[field]
        for name, value in _mapping(strategy.get("phases", {}), "strategy.phases").items():
            phase = _mapping(value, f"strategy.phases.{name}")
            if "system" in phase:
                knobs[f"{name}_system"] = str(phase["system"])
            elif "system_file" in phase:
                knobs[f"{name}_system"] = self._read_phase_prompt(name, str(phase["system_file"]))
        limits = _mapping(self.data.get("limits", {}), "limits")
        timeout = strategy.get("timeout", limits.get("timeout_seconds"))
        configured_system = strategy.get("system", "")
        return ReasoningSpec(
            strategy=str(strategy.get("name", "react")),
            system=system if system is not None else str(configured_system),
            max_iterations=int(strategy.get("max_iterations", 10)),
            knobs=knobs,
            policy=policy,
            timeout=float(timeout) if timeout is not None else None,
            llm_kwargs=dict(_mapping(strategy.get("llm_kwargs", {}), "strategy.llm_kwargs")),
            output_schema=output_schema,
        )

    def _read_phase_prompt(self, name: str, raw_path: str) -> str:
        """Read a phase ``system_file``, re-verifying it against the load-time hash."""
        path = Path(raw_path)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise AgentManifestError(
                f"could not read strategy.phases.{name}.system_file: {exc}"
            ) from exc
        key = f"strategy.phases.{name}.system_file:{_portable(path, self.allowed_roots)}"
        expected = self.referenced_fingerprints.get(key)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if expected is not None and digest != expected:
            raise AgentManifestError(
                f"strategy.phases.{name}.system_file changed since the manifest was "
                f"loaded; reload the manifest (expected {expected}, found {digest})"
            )
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AgentManifestError(
                f"strategy.phases.{name}.system_file must be UTF-8 text: {exc}"
            ) from exc

    def phase_models(self) -> dict[str, dict[str, Any]]:
        """Return per-phase model configurations declared under ``strategy.phases``.

        Mirrors the top-level ``model`` section contract: validated data that the
        application resolves into runtime LLMs (directly, or via
        ``agent_from_manifest``'s ``llm_factory``).
        """
        strategy = _mapping(self.data.get("strategy", {}), "strategy")
        models: dict[str, dict[str, Any]] = {}
        for name, value in _mapping(strategy.get("phases", {}), "strategy.phases").items():
            phase = _mapping(value, f"strategy.phases.{name}")
            model = phase.get("model")
            if model is not None:
                models[str(name)] = _thaw(model)
        return models

    def budget_policy(self) -> BudgetPolicy | None:
        """Build hard run-level budget caps declared under ``limits``.

        ``timeout_seconds`` belongs to the reasoning spec and is deliberately not
        duplicated into the budget. ``None`` means the manifest declares no budget
        settings, so callers can omit the policy entirely.
        """
        limits = _mapping(self.data.get("limits", {}), "limits")
        fields = (
            "max_wall_s",
            "max_llm_calls",
            "max_tool_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
            "max_cost",
            "reserve",
            "unpriced",
        )
        values = {field: limits[field] for field in fields if field in limits}
        return BudgetPolicy(**values) if values else None

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> ResolvedAgentManifest:
        """Return a new resolved manifest with governed dotted overrides applied."""
        if not overrides:
            return self
        data = self.as_dict()
        _apply_overrides(data, overrides)
        return _finalize_manifest(
            source=self.source,
            data=data,
            sources=list(self.sources),
            roots=self.allowed_roots,
            profile=self.profile,
        )


def load_agent_manifest(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    profile: str | None = None,
    allowed_roots: Iterable[str | Path] | None = None,
    max_inheritance_depth: int = 16,
) -> ResolvedAgentManifest:
    """Load and resolve a version-1 ``*.agent.*`` manifest.

    Resolution order is deterministic: inherited parents (left to right), the
    child, the selected embedded profile, then explicit dotted-path overrides.
    ``deny`` always wins over ``allow``. Local inherited and referenced paths must
    remain under ``allowed_roots``; by default only the entry manifest's directory
    is allowed.
    """
    source = Path(path).expanduser().resolve()
    _validate_manifest_suffix(source, context="entry manifest")
    if max_inheritance_depth < 1:
        raise ValueError("max_inheritance_depth must be at least 1")
    roots = tuple(
        Path(root).expanduser().resolve()
        for root in (allowed_roots if allowed_roots is not None else (source.parent,))
    )
    if not roots:
        raise ValueError("allowed_roots must contain at least one path")
    _require_allowed(source, roots, context="entry manifest")

    sources: list[Path] = []
    merged = _load_merged(
        source,
        roots=roots,
        stack=(),
        sources=sources,
        max_depth=max_inheritance_depth,
    )
    profiles = merged.pop("profiles", {})
    if profile is not None:
        profile_map = _mapping(profiles, "profiles")
        if profile not in profile_map:
            known = ", ".join(sorted(str(name) for name in profile_map)) or "(none)"
            raise AgentManifestError(f"unknown agent profile {profile!r}; known: {known}")
        selected = _mapping(profile_map[profile], f"profiles.{profile}")
        _validate_profile(selected, source)
        merged = _deep_merge(merged, dict(selected))

    _apply_overrides(merged, dict(overrides or {}))
    return _finalize_manifest(
        source=source,
        data=merged,
        sources=sources,
        roots=roots,
        profile=profile,
    )


def _finalize_manifest(
    *,
    source: Path,
    data: dict[str, Any],
    sources: list[Path],
    roots: tuple[Path, ...],
    profile: str | None,
) -> ResolvedAgentManifest:
    """Validate and fingerprint a fully merged manifest."""
    canonical = _canonical_config(data, "resolved agent manifest")
    assert isinstance(canonical, dict)
    # Paths declared by source files are already absolute at merge time. Resolve any
    # relative path introduced later by a selected profile or dotted override against
    # the entry manifest, the only declaration site available for runtime overrides.
    merged = _resolve_relative_paths(canonical, source.parent, roots)
    _validate_manifest(merged, source, resolved=True)
    _assert_no_secrets(merged)

    source_hashes = {_source_key(item, roots): _file_fingerprint(item) for item in sources}
    referenced_hashes = _referenced_fingerprints(merged, roots)
    portable = _portable_config(merged, roots)
    payload = {
        "config": portable,
        "profile": profile,
        "referenced_content": referenced_hashes,
    }
    fingerprint = _fingerprint(payload)
    return ResolvedAgentManifest(
        source=source,
        data=_freeze(merged),
        fingerprint=fingerprint,
        sources=tuple(sources),
        source_fingerprints=MappingProxyType(source_hashes),
        referenced_fingerprints=MappingProxyType(referenced_hashes),
        allowed_roots=roots,
        profile=profile,
    )


def _load_merged(
    path: Path,
    *,
    roots: tuple[Path, ...],
    stack: tuple[Path, ...],
    sources: list[Path],
    max_depth: int,
) -> dict[str, Any]:
    canonical = path.expanduser().resolve()
    _require_allowed(canonical, roots, context="inherited manifest")
    _validate_manifest_suffix(canonical, context="inherited manifest")
    if canonical in stack:
        cycle = " -> ".join(str(item) for item in (*stack, canonical))
        raise AgentManifestCycleError(f"agent manifest cycle detected: {cycle}")
    if len(stack) >= max_depth:
        raise AgentManifestError(
            f"agent manifest inheritance exceeds maximum {max_depth}: {canonical}"
        )
    data = _read_manifest(canonical)
    # Scan every source before inheritance/profile selection can discard content. The
    # final merged value is scanned again after runtime overrides.
    _assert_no_secrets(data)
    _validate_manifest(data, canonical, resolved=False)
    data = _resolve_relative_paths(data, canonical.parent, roots)

    merged: dict[str, Any] = {}
    for parent in _extends(data):
        parent_path = (canonical.parent / parent).resolve()
        merged = _deep_merge(
            merged,
            _load_merged(
                parent_path,
                roots=roots,
                stack=(*stack, canonical),
                sources=sources,
                max_depth=max_depth,
            ),
        )
    if canonical not in sources:
        sources.append(canonical)
    child = {key: value for key, value in data.items() if key != "extends"}
    return _deep_merge(merged, child)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentManifestError(f"could not read agent manifest {path}: {exc}") from exc
    try:
        if path.name.endswith((".agent.yaml", ".agent.yml")):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - exercised without yaml extra
                raise ImportError(
                    "pyyaml is required for YAML agent manifests: "
                    "pip install 'ai-arch-toolkit[yaml]'"
                ) from exc
            loaded = yaml.safe_load(text)
        elif path.name.endswith(".agent.json"):
            loaded = json.loads(text)
        else:
            loaded = tomllib.loads(text)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AgentManifestError(f"could not parse agent manifest {path}: {exc}") from exc
    except Exception as exc:
        if exc.__class__.__module__.startswith("yaml"):
            raise AgentManifestError(f"could not parse agent manifest {path}: {exc}") from exc
        raise
    if not isinstance(loaded, Mapping):
        raise AgentManifestError(f"agent manifest {path} must contain an object")
    canonical = _canonical_config(loaded, f"agent manifest {path}")
    assert isinstance(canonical, dict)
    return canonical


def _validate_manifest(data: Mapping[str, Any], path: Path, *, resolved: bool) -> None:
    _reject_unknown(data, _TOP_LEVEL_FIELDS, f"agent manifest {path}")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise AgentManifestError(
            f"agent manifest {path} must declare integer version: 1; got {version!r}"
        )
    if resolved and "extends" in data:
        raise AgentManifestError(f"resolved agent manifest {path} still contains extends")
    _optional_string(data, "id", f"agent manifest {path}")
    _optional_string(data, "phase", f"agent manifest {path}")
    _optional_string(data, "description", f"agent manifest {path}")
    _optional_string(data, "result_adapter", f"agent manifest {path}")
    _extends(data)

    strategy = _optional_mapping(data, "strategy", path)
    if strategy is not None:
        _reject_unknown(strategy, _STRATEGY_FIELDS, "strategy")
        _optional_string(strategy, "name", "strategy")
        _positive_int(strategy, "max_iterations", "strategy")
        _positive_number(strategy, "timeout", "strategy")
        for field in (
            "final_answer_hint",
            "parallel_tool_calls",
            "show_turn_counter",
            "strip_tools_on_final",
        ):
            _optional_bool(strategy, field, "strategy")
        for field in ("knobs", "llm_kwargs"):
            if field in strategy:
                _mapping(strategy[field], f"strategy.{field}")
        _validate_phases(strategy)

    model = _optional_mapping(data, "model", path)
    if model is not None:
        _validate_model_section(model, "model")

    prompts = _optional_mapping(data, "prompts", path)
    if prompts is not None:
        _reject_unknown(prompts, _PROMPT_FIELDS, "prompts")
        for field in _PROMPT_FIELDS - {"request_variables"}:
            _optional_string(prompts, field, "prompts")
        variables = prompts.get("request_variables")
        if variables is not None:
            variables_map = _mapping(variables, "prompts.request_variables")
            _reject_unknown(variables_map, _REQUEST_VARIABLE_FIELDS, "prompts.request_variables")
            for field in _REQUEST_VARIABLE_FIELDS:
                if field in variables_map:
                    _string_sequence(variables_map[field], f"prompts.request_variables.{field}")

    output = _optional_mapping(data, "output", path)
    if output is not None:
        _reject_unknown(output, _OUTPUT_FIELDS, "output")
        _optional_string(output, "schema", "output")
    tools = _optional_mapping(data, "tools", path)
    if tools is not None:
        _reject_unknown(tools, _TOOLS_FIELDS, "tools")
        _optional_string(tools, "factory", "tools")
        _optional_string(tools, "manifest", "tools")
    limits = _optional_mapping(data, "limits", path)
    if limits is not None:
        _reject_unknown(limits, _LIMIT_FIELDS, "limits")
        _nonnegative_number(limits, "max_cost", "limits")
        for field in ("max_wall_s", "timeout_seconds"):
            _positive_number(limits, field, "limits")
        for field in (
            "max_input_tokens",
            "max_llm_calls",
            "max_output_tokens",
            "max_tool_calls",
            "max_total_tokens",
        ):
            _nonnegative_int(limits, field, "limits")
        for field in ("reserve", "unpriced"):
            _optional_string(limits, field, "limits")
        reserve = limits.get("reserve")
        if reserve is not None and reserve not in {"none", "strict"}:
            raise AgentManifestError("limits.reserve must be 'none' or 'strict'")
        unpriced = limits.get("unpriced")
        if unpriced is not None and unpriced not in {"fail_closed", "allow"}:
            raise AgentManifestError("limits.unpriced must be 'fail_closed' or 'allow'")
    override_policy = _optional_mapping(data, "override_policy", path)
    if override_policy is not None:
        _reject_unknown(override_policy, _OVERRIDE_FIELDS, "override_policy")
        for field in _OVERRIDE_FIELDS:
            if field in override_policy:
                _string_sequence(override_policy[field], f"override_policy.{field}")
    if "metadata" in data:
        _mapping(data["metadata"], "metadata")
    if "profiles" in data:
        profiles = _mapping(data["profiles"], "profiles")
        for name, value in profiles.items():
            if not isinstance(name, str) or not name:
                raise AgentManifestError("agent profile names must be non-empty strings")
            profile = _mapping(value, f"profiles.{name}")
            _validate_profile(profile, path)


def _validate_model_section(model: Mapping[str, Any], context: str) -> None:
    _reject_unknown(model, _MODEL_FIELDS, context)
    for field in ("base_url", "model", "profile", "provider", "structured_output_mode"):
        _optional_string(model, field, context)
    _positive_int(model, "max_tokens", context)
    if "temperature" in model:
        value = model["temperature"]
        if not _is_number(value) or not 0 <= float(value) <= 2:
            raise AgentManifestError(f"{context}.temperature must be a number between 0 and 2")


def _validate_phases(strategy: Mapping[str, Any]) -> None:
    """Validate the ``strategy.phases`` section shape (loader-level, registry-agnostic)."""
    phases = strategy.get("phases")
    if phases is None:
        return
    phases_map = _mapping(phases, "strategy.phases")
    knobs = strategy.get("knobs")
    knob_map = knobs if isinstance(knobs, Mapping) else {}
    for name, value in phases_map.items():
        if not isinstance(name, str) or not name:
            raise AgentManifestError("strategy.phases names must be non-empty strings")
        context = f"strategy.phases.{name}"
        phase = _mapping(value, context)
        _reject_unknown(phase, _PHASE_FIELDS, context)
        for field in ("system", "system_file"):
            _optional_string(phase, field, context)
        if "system" in phase and "system_file" in phase:
            raise AgentManifestError(f"{context} must declare system or system_file, not both")
        file_value = phase.get("system_file")
        if isinstance(file_value, str) and file_value.endswith(_RENDERED_SUFFIXES):
            raise AgentManifestError(
                f"{context}.system_file must reference verbatim prompt text, not a "
                "prompt/agent manifest; render templates in the application and "
                "declare the result"
            )
        if ("system" in phase or "system_file" in phase) and f"{name}_system" in knob_map:
            raise AgentManifestError(
                f"{context} and strategy.knobs.{name}_system are both set; "
                "declare the phase prompt in one place"
            )
        model = phase.get("model")
        if model is not None:
            _validate_model_section(_mapping(model, f"{context}.model"), f"{context}.model")


def _validate_profile(data: Mapping[str, Any], path: Path) -> None:
    _reject_unknown(data, _PROFILE_FIELDS, f"agent profile in {path}")
    candidate = {"version": 1, **dict(data)}
    _validate_manifest(candidate, path, resolved=False)


def _apply_overrides(data: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    policy = _mapping(data.get("override_policy", {}), "override_policy")
    allow = _string_sequence(policy.get("allow", ()), "override_policy.allow")
    deny = _string_sequence(policy.get("deny", ()), "override_policy.deny")
    items = list(overrides.items())
    for path, _value in items:
        if not isinstance(path, str) or not path:
            raise AgentOverrideError("override paths must be non-empty strings")
    paths = sorted(path for path, _value in items)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if _paths_overlap(path, other):
                raise AgentOverrideError(
                    f"overlapping override paths are ambiguous: {path!r} and {other!r}"
                )

    for path, value in sorted(items, key=lambda item: item[0]):
        if any(_paths_overlap(entry, path) for entry in deny):
            raise AgentOverrideError(f"override {path!r} is denied by the manifest policy")
        if not any(_covers(entry, path) for entry in allow):
            raise AgentOverrideError(f"override {path!r} is not allowed by the manifest policy")
        canonical = _canonical_config(value, f"override {path!r}")
        _set_dotted(data, path, canonical)


def _set_dotted(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if any(not part for part in parts):
        raise AgentOverrideError(f"invalid override path {path!r}")
    current = data
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            current[part] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise AgentOverrideError(f"override {path!r} traverses non-object field {part!r}")
    current[parts[-1]] = value


def _resolve_relative_paths(
    data: dict[str, Any], base: Path, roots: tuple[Path, ...]
) -> dict[str, Any]:
    resolved = deepcopy(data)
    _resolve_path_fields(resolved, base, roots)
    profiles = resolved.get("profiles")
    if isinstance(profiles, dict):
        for name, profile in profiles.items():
            if isinstance(profile, dict):
                _resolve_path_fields(
                    profile,
                    base,
                    roots,
                    context_prefix=f"profiles.{name}.",
                )
    return resolved


def _resolve_path_fields(
    data: dict[str, Any],
    base: Path,
    roots: tuple[Path, ...],
    *,
    context_prefix: str = "",
) -> None:
    """Resolve path fields in one manifest/profile object in place."""
    for section, field in _PATH_FIELDS:
        section_value = data.get(section)
        if not isinstance(section_value, dict):
            continue
        value = section_value.get(field)
        if not isinstance(value, str) or not value or "://" in value:
            continue
        path = (base / value).resolve()
        _require_allowed(path, roots, context=f"{context_prefix}{section}.{field}")
        section_value[field] = str(path)
    for name, phase in _phase_sections(data):
        value = phase.get("system_file")
        if not isinstance(value, str) or not value or "://" in value:
            continue
        path = (base / value).resolve()
        _require_allowed(
            path, roots, context=f"{context_prefix}strategy.phases.{name}.system_file"
        )
        phase["system_file"] = str(path)


def _phase_sections(data: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return mutable (name, phase) pairs under ``strategy.phases``, if present."""
    strategy = data.get("strategy")
    if not isinstance(strategy, dict):
        return []
    phases = strategy.get("phases")
    if not isinstance(phases, dict):
        return []
    return [(str(name), phase) for name, phase in phases.items() if isinstance(phase, dict)]


def _validate_manifest_suffix(path: Path, *, context: str) -> None:
    if not path.name.endswith(_SUFFIXES):
        raise AgentManifestError(
            f"{context} {path} must use an .agent.yaml, .agent.yml, .agent.json, "
            "or .agent.toml filename"
        )


def _referenced_fingerprints(data: Mapping[str, Any], roots: tuple[Path, ...]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for section, field in _PATH_FIELDS:
        section_value = data.get(section)
        if not isinstance(section_value, Mapping):
            continue
        value = section_value.get(field)
        if not isinstance(value, str) or not value or "://" in value:
            continue
        path = Path(value).resolve()
        _require_allowed(path, roots, context=f"{section}.{field}")
        if not path.is_file():
            raise AgentManifestError(f"referenced file does not exist: {path}")
        fingerprints[f"{section}.{field}:{_portable(path, roots)}"] = _file_fingerprint(path)
    for name, phase in _phase_sections(data):
        value = phase.get("system_file")
        if not isinstance(value, str) or not value or "://" in value:
            continue
        path = Path(value).resolve()
        _require_allowed(path, roots, context=f"strategy.phases.{name}.system_file")
        if not path.is_file():
            raise AgentManifestError(f"referenced file does not exist: {path}")
        key = f"strategy.phases.{name}.system_file:{_portable(path, roots)}"
        fingerprints[key] = _file_fingerprint(path)
    return dict(sorted(fingerprints.items()))


def _portable_config(data: Mapping[str, Any], roots: tuple[Path, ...]) -> dict[str, Any]:
    portable = deepcopy(dict(data))
    for section, field in _PATH_FIELDS:
        section_value = portable.get(section)
        if not isinstance(section_value, dict):
            continue
        value = section_value.get(field)
        if isinstance(value, str) and value and "://" not in value:
            section_value[field] = _portable(Path(value), roots)
    for _name, phase in _phase_sections(portable):
        value = phase.get("system_file")
        if isinstance(value, str) and value and "://" not in value:
            phase["system_file"] = _portable(Path(value), roots)
    return portable


def _extends(data: Mapping[str, Any]) -> tuple[str, ...]:
    value = data.get("extends", ())
    if isinstance(value, str):
        entries = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        entries = tuple(value)
    else:
        raise AgentManifestError("agent manifest extends must be a path or list of paths")
    if any(not isinstance(entry, str) or not entry for entry in entries):
        raise AgentManifestError("agent manifest extends paths must be non-empty strings")
    return entries


def _deep_merge(base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _reject_unknown(data: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(str(field) for field in set(data) - allowed)
    if unknown:
        raise AgentManifestError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _optional_mapping(data: Mapping[str, Any], field: str, path: Path) -> Mapping[str, Any] | None:
    value = data.get(field)
    if value is None:
        return None
    return _mapping(value, f"{field} in {path}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentManifestError(f"{context} must be an object")
    return value


def _optional_string(data: Mapping[str, Any], field: str, context: str) -> None:
    value = data.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise AgentManifestError(f"{context}.{field} must be a non-empty string")


def _string_sequence(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise AgentManifestError(f"{context} must be a list of strings")
    entries = tuple(value)
    if any(not isinstance(entry, str) or not entry for entry in entries):
        raise AgentManifestError(f"{context} must contain only non-empty strings")
    return entries


def _optional_bool(data: Mapping[str, Any], field: str, context: str) -> None:
    value = data.get(field)
    if value is not None and not isinstance(value, bool):
        raise AgentManifestError(f"{context}.{field} must be a boolean")


def _positive_int(data: Mapping[str, Any], field: str, context: str) -> None:
    value = data.get(field)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise AgentManifestError(f"{context}.{field} must be a positive integer")


def _nonnegative_int(data: Mapping[str, Any], field: str, context: str) -> None:
    value = data.get(field)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise AgentManifestError(f"{context}.{field} must be a non-negative integer")


def _positive_number(data: Mapping[str, Any], field: str, context: str) -> None:
    value = data.get(field)
    if value is not None and (
        not _is_number(value) or not math.isfinite(float(value)) or float(value) <= 0
    ):
        raise AgentManifestError(f"{context}.{field} must be a finite positive number")


def _nonnegative_number(data: Mapping[str, Any], field: str, context: str) -> None:
    value = data.get(field)
    if value is not None and (
        not _is_number(value) or not math.isfinite(float(value)) or float(value) < 0
    ):
        raise AgentManifestError(f"{context}.{field} must be a finite non-negative number")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _covers(entry: str, path: str) -> bool:
    return path == entry or path.startswith(entry + ".")


def _paths_overlap(first: str, second: str) -> bool:
    """Return whether writing either dotted path can affect the other."""
    return _covers(first, second) or _covers(second, first)


def _canonical_config(
    value: Any,
    context: str,
    *,
    _active: set[int] | None = None,
) -> Any:
    """Return the canonical JSON-like manifest value or reject unsupported input.

    This is the single trust boundary shared by parsed files and runtime overrides.
    It prevents executable/process-local objects from entering resolved data or its
    deterministic fingerprint.
    """
    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AgentManifestError(f"{context} contains a non-finite number")
        return float(value)

    active = _active if _active is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise AgentManifestError(f"{context} contains a cyclic object")
        active.add(identity)
        try:
            canonical: dict[str, Any] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise AgentManifestError(
                        f"{context} contains a non-string object key: {key!r}"
                    )
                canonical[key] = _canonical_config(
                    child,
                    f"{context}.{key}" if key else context,
                    _active=active,
                )
            return canonical
        finally:
            active.remove(identity)

    if isinstance(value, list | tuple):
        identity = id(value)
        if identity in active:
            raise AgentManifestError(f"{context} contains a cyclic array")
        active.add(identity)
        try:
            return [
                _canonical_config(child, f"{context}[{index}]", _active=active)
                for index, child in enumerate(value)
            ]
        finally:
            active.remove(identity)

    raise AgentManifestError(
        f"{context} contains unsupported value type {type(value).__name__}; "
        "agent manifests accept only JSON-like data"
    )


def _require_allowed(path: Path, roots: tuple[Path, ...], *, context: str) -> None:
    if not any(path == root or path.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise AgentManifestError(f"{context} path {path} is outside allowed roots: {allowed}")


def _portable(path: Path, roots: tuple[Path, ...]) -> str:
    canonical = path.resolve()
    for root in roots:
        if canonical == root or canonical.is_relative_to(root):
            relative = canonical.relative_to(root).as_posix()
            return relative or "."
    return canonical.name  # pragma: no cover - callers validate first


def _source_key(path: Path, roots: tuple[Path, ...]) -> str:
    """Return an unambiguous, machine-independent provenance key."""
    canonical = path.resolve()
    for index, root in enumerate(roots):
        if canonical == root or canonical.is_relative_to(root):
            relative = canonical.relative_to(root).as_posix() or "."
            return relative if len(roots) == 1 else f"root[{index}]:{relative}"
    return f"outside-roots:{canonical.name}"  # pragma: no cover - callers validate first


def _file_fingerprint(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AgentManifestError(f"could not fingerprint {path}: {exc}") from exc
    return f"sha256:{digest}"


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _assert_no_secrets(value: Any, trail: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _SECRET_NAMES or lowered.endswith(_SECRET_SUFFIXES):
                raise AgentManifestError(
                    f"secret-like field {trail}{key!s} must not be stored in an agent manifest"
                )
            _assert_no_secrets(child, f"{trail}{key}.")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _assert_no_secrets(child, trail)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return deepcopy(value)
