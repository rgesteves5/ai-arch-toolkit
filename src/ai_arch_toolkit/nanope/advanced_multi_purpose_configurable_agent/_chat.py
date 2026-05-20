"""Terminal chat for manually testing the configurable agent with real LLM calls."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._agent import (
    AgentRunResult,
    ConfigurableAgent,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._config import (
    AgentConfig,
    agent_config_from_mapping,
    load_agent_config,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._memory import (
    create_private_memory,
    load_private_memory_sync,
    save_private_memory_sync,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._profiles import (
    SAFE_CHAT_TOOLS,
    built_in_profiles,
    profile_details,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._tools import (
    built_in_tool_registry,
)

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TOOLS = SAFE_CHAT_TOOLS


@dataclass(slots=True)
class ChatSession:
    """Small stateful terminal chat wrapper around ConfigurableAgent."""

    agent: ConfigurableAgent
    history_turns: int = 8
    max_history_chars: int | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    history: list[tuple[str, str]] = field(default_factory=list)
    last_result: AgentRunResult | None = None

    def ask(self, message: str) -> AgentRunResult:
        """Run one chat turn."""
        task = self._build_task(message)
        result = self.agent.run_sync(task, overrides=self.overrides)
        self.last_result = result
        if result.final_text:
            self.history.append((message, result.final_text))
            self.history = self.history[-self.history_turns :]
        return result

    def reset(self) -> None:
        """Clear chat history."""
        self.history.clear()
        self.last_result = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize chat-visible session state."""
        return {
            "history": [{"user": user, "assistant": assistant} for user, assistant in self.history]
        }

    def load_dict(self, data: Mapping[str, Any]) -> None:
        """Load chat-visible session state."""
        history: list[tuple[str, str]] = []
        for item in data.get("history", ()):
            if isinstance(item, Mapping):
                history.append((str(item.get("user", "")), str(item.get("assistant", ""))))
        self.history = history[-self.history_turns :]

    def save(self, path: str | Path) -> None:
        """Save chat-visible session state to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def load(self, path: str | Path) -> None:
        """Load chat-visible session state from JSON if it exists."""
        session_path = Path(path)
        if session_path.exists():
            self.load_dict(json.loads(session_path.read_text()))

    def _build_task(self, message: str) -> str:
        if not self.history:
            return message

        lines = ["Conversation so far:"]
        for user_text, assistant_text in self.history[-self.history_turns :]:
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {assistant_text}")
        lines.append("")
        lines.append(f"New user message: {message}")
        task = "\n".join(lines)
        if self.max_history_chars is None or len(task) <= self.max_history_chars:
            return task
        return task[-self.max_history_chars :]


def build_chat_agent(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_iterations: int = 6,
    tools: Sequence[str] = DEFAULT_TOOLS,
    profiles: Sequence[str] = (),
    strategy: str = "react",
    memory_enabled: bool = False,
    memory: Any = None,
    config: AgentConfig | Mapping[str, Any] | None = None,
    tool_permissions: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
) -> ConfigurableAgent:
    """Build a default terminal-chat agent."""
    if config is not None:
        return ConfigurableAgent(
            config,
            profiles=built_in_profiles(),
            tool_registry=built_in_tool_registry(),
            memory=memory,
        )

    return ConfigurableAgent(
        {
            "identity": {
                "name": "manual_chat_agent",
                "description": "A configurable agent used for manual terminal chat testing.",
            },
            "context": {
                "role": "Helpful general-purpose assistant",
                "goals": [
                    "answer the user's messages clearly",
                    "use tools when they materially improve the answer",
                    "keep continuity with the visible chat history",
                ],
                "style": "concise, direct, and practical",
            },
            "model": {
                "name": model,
                "temperature": temperature,
            },
            "capability_profiles": list(profiles),
            "reasoning": {
                "strategy": strategy,
                "max_iterations": max_iterations,
                "final_answer_hint": True,
            },
            "tools": {
                "enabled": list(tools),
                "permissions": dict(tool_permissions or {}),
            },
            "memory": {
                "private_enabled": memory_enabled,
                "read": memory_enabled,
                "write": memory_enabled,
            },
            "output": dict(output or {}),
            "override_policy": {
                "allow": [
                    "model.temperature",
                    "model.max_tokens",
                    "reasoning.max_iterations",
                    "reasoning.strategy",
                    "reasoning.strategy_kwargs",
                    "tools.enabled",
                    "tools.disabled",
                    "tools.permissions",
                    "memory.read",
                    "memory.write",
                    "output.schema",
                    "output.name",
                    "output.strict",
                ],
                "deny": [
                    "identity.name",
                    "identity.description",
                ],
            },
        },
        profiles=built_in_profiles(),
        tool_registry=built_in_tool_registry(),
        memory=memory,
    )


def run_terminal_chat(
    *,
    config_path: str | Path | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_iterations: int | None = None,
    tools: Sequence[str] | None = None,
    profiles: Sequence[str] = (),
    strategy: str | None = None,
    memory_enabled: bool = False,
    memory_path: str | Path | None = None,
    history_turns: int = 8,
    max_history_chars: int | None = None,
    session_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    tool_permissions: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    load_env: bool = True,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., Any] = print,
) -> None:
    """Run the interactive terminal chat."""
    if load_env:
        load_dotenv()

    loaded_config: AgentConfig | None = load_agent_config(config_path) if config_path else None
    if loaded_config is not None:
        loaded_data = loaded_config.to_dict()
        loaded_data = _apply_config_cli_updates(
            loaded_data,
            model=model,
            temperature=temperature,
            max_iterations=max_iterations,
            tools=tools,
            profiles=profiles,
            strategy=strategy,
        )
        loaded_config = agent_config_from_mapping(loaded_data)
        memory_enabled = (
            loaded_config.memory.private_enabled
            or "private_memory_user" in loaded_config.capability_profiles
            or memory_enabled
        )
        if memory_enabled:
            loaded_data["memory"] = {
                **loaded_data.get("memory", {}),
                "private_enabled": True,
                "read": True,
                "write": True,
            }
        if output:
            loaded_data["output"] = {**loaded_data.get("output", {}), **dict(output)}
        if tool_permissions and any(
            value is not None and value is not False for value in tool_permissions.values()
        ):
            loaded_data["tools"] = {
                **loaded_data.get("tools", {}),
                "permissions": {
                    **loaded_data.get("tools", {}).get("permissions", {}),
                    **dict(tool_permissions),
                },
            }
        loaded_config = agent_config_from_mapping(loaded_data)
    memory_enabled = memory_enabled or "private_memory_user" in profiles

    memory = None
    if memory_enabled:
        memory = (
            load_private_memory_sync(memory_path)
            if memory_path is not None
            else create_private_memory()
        )

    agent = build_chat_agent(
        model=model if model is not None else DEFAULT_MODEL,
        temperature=temperature if temperature is not None else 0.2,
        max_iterations=max_iterations if max_iterations is not None else 6,
        tools=tools if tools is not None else DEFAULT_TOOLS,
        profiles=profiles,
        strategy=strategy if strategy is not None else "react",
        memory_enabled=memory_enabled,
        memory=memory,
        config=loaded_config,
        tool_permissions=tool_permissions,
        output=output,
    )
    session = ChatSession(
        agent=agent,
        history_turns=history_turns,
        max_history_chars=max_history_chars,
        overrides=dict(overrides or {}),
    )
    if session_path is not None:
        session.load(session_path)
    resolved = agent.resolve_config(run_overrides=overrides).config
    tool_names = agent.resolve_tool_names(overrides=overrides)

    print_fn("Advanced configurable agent chat")
    profile_text = (
        ", ".join(resolved.capability_profiles) if resolved.capability_profiles else "none"
    )
    print_fn(
        f"model={resolved.model.name} strategy={resolved.reasoning.strategy} "
        f"profiles={profile_text}"
    )
    print_fn(
        f"tools={', '.join(tool_names) if tool_names else 'none'} "
        f"memory={resolved.memory.private_enabled}"
    )
    print_fn(
        "Commands: /help, /reset, /tools, /profiles, /prompt, /capabilities, /last-result, /exit"
    )

    while True:
        try:
            message = input_fn("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print_fn("\nbye")
            return

        if not message:
            continue
        if message in {"/exit", "/quit"}:
            print_fn("bye")
            return
        if message == "/help":
            _print_help(print_fn)
            continue
        if message == "/reset":
            session.reset()
            if session_path is not None:
                session.save(session_path)
            print_fn("history reset")
            continue
        if message == "/tools":
            names = agent.resolve_tool_names(overrides=overrides)
            print_fn(", ".join(names) if names else "no tools enabled")
            if resolved.tools.disabled:
                print_fn(f"disabled: {', '.join(resolved.tools.disabled)}")
            continue
        if message == "/profiles":
            print_fn(json.dumps(profile_details(), indent=2, sort_keys=True))
            continue
        if message.startswith("/profile "):
            profile_name = message.removeprefix("/profile ").strip()
            try:
                details = profile_details(profile_name)
            except KeyError:
                print_fn(f"unknown profile: {profile_name}")
            else:
                print_fn(json.dumps(details, indent=2, sort_keys=True))
            continue
        if message == "/strategies":
            print_fn(
                "react, plan_execute, reflexion, self_discovery, generate_review, "
                "rewoo, llm_compiler, tot, lats"
            )
            continue
        if message == "/prompt":
            print_fn(agent.render_prompt(resolved).system)
            continue
        if message == "/last-prompt":
            if session.last_result is None:
                print_fn("no run yet")
            else:
                print_fn(f"prompt_fingerprint={session.last_result.prompt_fingerprint}")
            continue
        if message == "/capabilities":
            print_fn(agent.describe_capabilities(overrides=overrides))
            continue
        if message == "/last-tools":
            _print_last(session, print_fn, "enabled_tools")
            continue
        if message == "/last-cost":
            _print_last_cost(session, print_fn)
            continue
        if message == "/last-result":
            _print_last_result(session, print_fn)
            continue
        if message == "/last-trace":
            _print_last_trace(session, print_fn)
            continue
        if message == "/save-session":
            if session_path is None:
                print_fn("no --session-path configured")
            else:
                session.save(session_path)
                print_fn(f"saved session to {session_path}")
            continue

        result = session.ask(message)
        if memory_enabled and memory_path is not None and memory is not None:
            save_private_memory_sync(memory, memory_path)
        if session_path is not None:
            session.save(session_path)
        print_fn(f"\nAgent> {result.final_text}")
        print_fn(
            f"[status={result.status} cost=${result.cost:.6f} "
            f"tokens={result.usage.input_tokens + result.usage.output_tokens}]"
        )
        if result.errors:
            print_fn("errors:", "; ".join(result.errors))


def load_dotenv(path: str | Path = ".env") -> None:
    """Load a simple KEY=VALUE .env file into os.environ when keys are unset."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Manual chat for the configurable agent.")
    parser.add_argument("--config", default="", help="TOML/YAML agent config file.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--strategy",
        default=None,
        choices=[
            "react",
            "plan_execute",
            "reflexion",
            "self_discovery",
            "generate_review",
            "rewoo",
            "llm_compiler",
            "tot",
            "lats",
        ],
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated built-in capability profiles.",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Comma-separated tool names, 'all', or 'none'.",
    )
    parser.add_argument("--memory", action="store_true", help="Enable private memory tools.")
    parser.add_argument(
        "--memory-path",
        default="",
        help="Optional JSON file for private memory persistence.",
    )
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--max-history-chars", type=int, default=0)
    parser.add_argument("--session-path", default="", help="Optional JSON chat session file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Runtime override using dotted paths, e.g. model.temperature=0.4.",
    )
    parser.add_argument(
        "--allow-dangerous-tools",
        action="store_true",
        help="Allow run_command and python_repl when those tools are enabled.",
    )
    parser.add_argument(
        "--dry-run-tools",
        action="store_true",
        help="Return tool dry-run messages instead of executing tools.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Maximum total tool calls per agent run.",
    )
    parser.add_argument(
        "--output-schema",
        default="",
        help="Optional JSON Schema file for structured output on supported strategies.",
    )
    parser.add_argument("--no-env", action="store_true", help="Do not load .env from cwd.")
    args = parser.parse_args(argv)

    tools = _parse_tools(args.tools) if args.tools is not None else None
    profiles = _parse_csv(args.profiles)
    overrides = _parse_overrides(args.override)
    output = _load_output_schema(args.output_schema) if args.output_schema else None
    tool_permissions = {
        "allow_dangerous": args.allow_dangerous_tools,
        "dry_run": args.dry_run_tools,
        "max_calls": args.max_tool_calls,
    }
    run_terminal_chat(
        config_path=args.config or None,
        model=args.model if args.model is not None else (None if args.config else DEFAULT_MODEL),
        temperature=args.temperature
        if args.temperature is not None
        else (None if args.config else 0.2),
        max_iterations=(
            args.max_iterations
            if args.max_iterations is not None
            else (None if args.config else 6)
        ),
        tools=tools if tools is not None else (None if args.config else DEFAULT_TOOLS),
        profiles=profiles,
        strategy=args.strategy
        if args.strategy is not None
        else (None if args.config else "react"),
        memory_enabled=args.memory,
        memory_path=args.memory_path or None,
        history_turns=args.history_turns,
        max_history_chars=args.max_history_chars or None,
        session_path=args.session_path or None,
        overrides=overrides,
        tool_permissions=tool_permissions,
        output=output,
        load_env=not args.no_env,
    )


def _parse_tools(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"", "none", "no", "false"}:
        return ()
    if value.strip().lower() == "all":
        return ("all",)
    return _parse_csv(value)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _apply_config_cli_updates(
    data: Mapping[str, Any],
    *,
    model: str | None,
    temperature: float | None,
    max_iterations: int | None,
    tools: Sequence[str] | None,
    profiles: Sequence[str],
    strategy: str | None,
) -> dict[str, Any]:
    updated = _plain_dict(data)
    if model is not None:
        _set_config_path(updated, "model.name", model)
    if temperature is not None:
        _set_config_path(updated, "model.temperature", temperature)
    if max_iterations is not None:
        _set_config_path(updated, "reasoning.max_iterations", max_iterations)
    if strategy is not None:
        _set_config_path(updated, "reasoning.strategy", strategy)
    if tools is not None:
        _set_config_path(updated, "tools.enabled", list(tools))
    if profiles:
        existing = list(updated.get("capability_profiles", ()))
        for profile in profiles:
            if profile not in existing:
                existing.append(profile)
        updated["capability_profiles"] = existing
    return updated


def _set_config_path(data: dict[str, Any], path: str, value: Any) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _plain_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            plain[str(key)] = _plain_dict(item)
        elif isinstance(item, tuple | list):
            plain[str(key)] = [_plain_value(v) for v in item]
        else:
            plain[str(key)] = item
    return plain


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_dict(value)
    if isinstance(value, tuple | list):
        return [_plain_value(v) for v in value]
    return value


def _parse_overrides(values: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Override must use PATH=VALUE format: {value!r}")
        path, raw_value = value.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(f"Override path is empty: {value!r}")
        overrides[path] = _parse_value(raw_value.strip())
    return overrides


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    if value.startswith(("[", "{", '"')):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_output_schema(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if "schema" in data:
        return data
    return {"schema": data}


def _print_last(session: ChatSession, print_fn: Callable[..., Any], field_name: str) -> None:
    if session.last_result is None:
        print_fn("no run yet")
        return
    print_fn(getattr(session.last_result, field_name))


def _print_last_cost(session: ChatSession, print_fn: Callable[..., Any]) -> None:
    if session.last_result is None:
        print_fn("no run yet")
        return
    result = session.last_result
    print_fn(
        {
            "cost": result.cost,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
        }
    )


def _print_last_result(session: ChatSession, print_fn: Callable[..., Any]) -> None:
    if session.last_result is None:
        print_fn("no run yet")
        return
    result = session.last_result
    print_fn(
        json.dumps(
            {
                "agent_name": result.agent_name,
                "status": result.status,
                "final_text": result.final_text,
                "cost": result.cost,
                "enabled_tools": list(result.enabled_tools),
                "errors": list(result.errors),
                "override_report": {
                    "accepted": result.override_report.accepted,
                    "rejected": result.override_report.rejected,
                },
                "memory_report": dict(result.memory_report),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_last_trace(session: ChatSession, print_fn: Callable[..., Any]) -> None:
    if session.last_result is None or session.last_result.flow_result is None:
        print_fn("no run yet")
        return
    trace = session.last_result.flow_result.trace
    print_fn(
        {
            "step_count": len(trace.steps),
            "total_cost": session.last_result.flow_result.total_cost,
            "total_usage": {
                "input_tokens": trace.total_usage.input_tokens,
                "output_tokens": trace.total_usage.output_tokens,
            },
        }
    )


def _print_help(print_fn: Callable[..., Any]) -> None:
    print_fn(
        "Commands:\n"
        "/reset clears chat history\n"
        "/tools shows enabled tools\n"
        "/profiles shows built-in profile details\n"
        "/profile NAME shows one profile\n"
        "/strategies lists available reasoning strategies\n"
        "/prompt shows the rendered system prompt\n"
        "/last-prompt shows the last prompt fingerprint\n"
        "/capabilities shows resolved capabilities\n"
        "/last-tools shows tools used for the last run\n"
        "/last-cost shows last run cost and token usage\n"
        "/last-result shows structured metadata for the last run\n"
        "/last-trace shows compact flow trace metadata\n"
        "/save-session saves chat history when --session-path is configured\n"
        "/exit quits"
    )
