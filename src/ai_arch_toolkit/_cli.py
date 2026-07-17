"""Dependency-free command-line interface for local toolkit workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import PromptError, load_prompt
from ai_arch_toolkit.toolkit.resources import ResourceError, load_resource


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(prog="ai-arch")
    commands = parser.add_subparsers(dest="command", required=True)
    prompt = commands.add_parser("prompt", help="validate, inspect, or render prompt manifests")
    prompt_commands = prompt.add_subparsers(dest="prompt_command", required=True)

    validate = prompt_commands.add_parser("validate", help="validate a prompt manifest")
    validate.add_argument("path", type=Path)
    _add_knowledge_arguments(validate)

    inspect = prompt_commands.add_parser("inspect", help="inspect a prompt definition")
    inspect.add_argument("path", type=Path)
    _add_knowledge_arguments(inspect)

    render = prompt_commands.add_parser("render", help="render a prompt manifest locally")
    render.add_argument("path", type=Path)
    render.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    render.add_argument("--vars", type=Path, metavar="FILE")
    render.add_argument("--layout", choices=("json", "markdown", "text", "xml"))
    _add_knowledge_arguments(render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit status."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prompt":
            return _run_prompt(args)
    except (PromptError, ResourceError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


def _run_prompt(args: argparse.Namespace) -> int:
    template = load_prompt(args.path, knowledge=_load_knowledge(args))
    if args.prompt_command == "validate":
        template.validate()
        print(
            f"valid prompt {template.name or args.path.name!r}: "
            f"{_count_sections(template.sections)} sections, "
            f"{len(template.variables)} variables"
        )
        return 0
    if args.prompt_command == "inspect":
        print(json.dumps(_inspect_template(template), indent=2, ensure_ascii=False, default=str))
        return 0
    if args.prompt_command == "render":
        variables = _load_variables_file(args.vars) if args.vars else {}
        for assignment in args.var:
            name, value = _parse_assignment(assignment)
            variables[name] = value
        rendered = template.render(layout=args.layout, **variables)
        print(rendered.text)
        return 0
    raise ValueError(f"unknown prompt command {args.prompt_command!r}")


def _add_knowledge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knowledge-dir", type=Path, metavar="DIR")
    parser.add_argument("--knowledge-recursive", action="store_true")
    parser.add_argument("--knowledge", action="append", default=[], metavar="KEY=FILE")


def _load_knowledge(args: argparse.Namespace) -> KnowledgeRegistry | None:
    registry: KnowledgeRegistry | None = None
    if args.knowledge_dir is not None:
        registry = KnowledgeRegistry.from_directory(
            args.knowledge_dir,
            recursive=args.knowledge_recursive,
        )
    for assignment in args.knowledge:
        key, raw_path = _split_assignment(assignment, option="--knowledge")
        if not raw_path:
            raise ValueError("--knowledge value cannot be empty")
        registry = registry or KnowledgeRegistry()
        registry.load(key, Path(raw_path))
    return registry


def _count_sections(sections: Sequence[Any]) -> int:
    return sum(1 + _count_sections(section.sections) for section in sections)


def _inspect_template(template: Any) -> dict[str, Any]:
    return dict(template.inspect())


def _load_variables_file(path: Path) -> dict[str, Any]:
    resource = load_resource(path)
    if not isinstance(resource.data, Mapping):
        raise ValueError(f"variables file {path} must contain an object")
    return dict(resource.data)


def _parse_assignment(assignment: str) -> tuple[str, Any]:
    name, raw = _split_assignment(assignment, option="--var")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return name, value


def _split_assignment(assignment: str, *, option: str) -> tuple[str, str]:
    if "=" not in assignment:
        raise ValueError(f"{option} must use NAME=VALUE syntax: {assignment!r}")
    name, raw = assignment.split("=", 1)
    if not name:
        raise ValueError(f"{option} name cannot be empty")
    return name, raw


__all__ = ["build_parser", "main"]
