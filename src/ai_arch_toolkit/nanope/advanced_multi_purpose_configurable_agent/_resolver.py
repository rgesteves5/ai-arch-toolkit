"""Config resolution and runtime override handling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core import redact
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._config import (
    AgentConfig,
    CapabilityProfile,
    OverridePolicy,
    agent_config_from_mapping,
)

_LIST_APPEND_KEYS = {
    "capability_profiles",
    "context.goals",
    "context.tasks",
    "context.constraints",
    "context.extra_sections",
    "model.fallback",
    "tools.enabled",
    "tools.disabled",
    "override_policy.allow",
    "override_policy.deny",
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "reasoning": {
        "strategy": "react",
        "max_iterations": 10,
        "parallel_tool_calls": True,
        "final_answer_hint": True,
        "strip_tools_on_final": False,
        "show_turn_counter": False,
    },
    "model": {
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "observability": {
        "trace": True,
        "debug_prompts": False,
    },
}


@dataclass(frozen=True, slots=True, kw_only=True)
class OverrideReport:
    """Accepted and rejected runtime override paths."""

    accepted: dict[str, Any] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedAgentConfig:
    """Resolved config plus override report."""

    config: AgentConfig
    override_report: OverrideReport = field(default_factory=OverrideReport)


def resolve_agent_config(
    config: AgentConfig | Mapping[str, Any],
    *,
    profiles: Mapping[str, CapabilityProfile | Mapping[str, Any]] | None = None,
    environment_config: Mapping[str, Any] | None = None,
    session_overrides: Mapping[str, Any] | None = None,
    run_overrides: Mapping[str, Any] | None = None,
    step_overrides: Mapping[str, Any] | None = None,
) -> ResolvedAgentConfig:
    """Resolve defaults, profiles, config, and runtime overrides into one config."""
    config_dict = (
        config.to_dict(include_secrets=True) if isinstance(config, AgentConfig) else _plain(config)
    )
    profile_registry = profiles or {}

    merged = _merge_static_layers(
        config_dict,
        profile_names=_profile_names(config),
        profiles=profile_registry,
    )
    if environment_config:
        merged = _deep_merge(merged, environment_config)

    # Parse once here so override policy can come from profiles/config/environment.
    policy = agent_config_from_mapping(merged).override_policy
    override_layers = (session_overrides, run_overrides, step_overrides)
    merged, report = _apply_override_layers(merged, override_layers, policy)

    if any(_is_capability_profile_path(path) for path in report.accepted):
        merged = _merge_static_layers(
            config_dict,
            profile_names=_profile_names(merged),
            profiles=profile_registry,
        )
        if environment_config:
            merged = _deep_merge(merged, environment_config)
        merged, report = _apply_override_layers(merged, override_layers, policy)

    return ResolvedAgentConfig(
        config=agent_config_from_mapping(merged),
        override_report=report,
    )


def _merge_static_layers(
    config_dict: Mapping[str, Any],
    *,
    profile_names: tuple[str, ...],
    profiles: Mapping[str, CapabilityProfile | Mapping[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = _deep_merge({}, _DEFAULT_CONFIG)
    for name in profile_names:
        profile_data = _profile_data(name, profiles)
        merged = _deep_merge(merged, profile_data)
    return _deep_merge(merged, config_dict)


def _apply_override_layers(
    merged: Mapping[str, Any],
    layers: tuple[Mapping[str, Any] | None, ...],
    policy: OverridePolicy,
) -> tuple[dict[str, Any], OverrideReport]:
    result = _plain(merged)
    accepted: dict[str, Any] = {}
    rejected: dict[str, str] = {}

    for layer in layers:
        if not layer:
            continue
        result, report = apply_overrides(result, layer, policy)
        accepted.update(report.accepted)
        rejected.update(report.rejected)
    return result, OverrideReport(accepted=accepted, rejected=rejected)


def apply_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    policy: OverridePolicy,
) -> tuple[dict[str, Any], OverrideReport]:
    """Apply dotted-path overrides to a copy of base according to policy."""
    result = _plain(base)
    accepted: dict[str, Any] = {}
    rejected: dict[str, str] = {}

    for path, value in _flatten_overrides(overrides).items():
        if _matches_any(path, policy.deny):
            rejected[path] = "Denied by override_policy"
            continue
        if policy.allow and not _matches_any(path, policy.allow):
            rejected[path] = "Not allowed by override_policy"
            continue
        if path in _LIST_APPEND_KEYS:
            value = _append_unique(_get_path(result, path), value)
        _set_path(result, path, value)
        accepted[path] = _override_report_value(path, value)

    return result, OverrideReport(accepted=accepted, rejected=rejected)


def _override_report_value(path: str, value: Any) -> Any:
    return redact({path: value})[path]


def _profile_names(config: AgentConfig | Mapping[str, Any]) -> tuple[str, ...]:
    if isinstance(config, AgentConfig):
        return config.capability_profiles
    value = config.get("capability_profiles", ())
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _is_capability_profile_path(path: str) -> bool:
    return path == "capability_profiles" or path.startswith("capability_profiles.")


def _profile_data(
    name: str,
    profiles: Mapping[str, CapabilityProfile | Mapping[str, Any]],
) -> dict[str, Any]:
    profile = profiles.get(name)
    if profile is None:
        raise KeyError(f"Unknown capability profile: {name!r}")
    if isinstance(profile, CapabilityProfile):
        return profile.to_dict()
    return _plain(profile)


def _deep_merge(
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    merged = _plain(base)
    for key, value in _plain(overlay).items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
            and path not in _LIST_APPEND_KEYS
        ):
            merged[key] = _deep_merge(merged[key], value, prefix=path)
        elif path in _LIST_APPEND_KEYS:
            merged[key] = _append_unique(merged.get(key, ()), value)
        else:
            merged[key] = value
    return merged


def _append_unique(left: Any, right: Any) -> list[Any]:
    values: list[Any] = []
    for item in _as_list(left) + _as_list(right):
        if item not in values:
            values.append(item)
    return values


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _flatten_overrides(overrides: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in overrides.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and not any("." in str(k) for k in value):
            flat.update(_flatten_overrides(value, path))
        else:
            flat[path] = value
    return flat


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches(path, pattern) for pattern in patterns)


def _matches(path: str, pattern: str) -> bool:
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return path == prefix or path.startswith(f"{prefix}.")
    return path == pattern or path.startswith(f"{pattern}.")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value
