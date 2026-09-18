"""Executable ownership boundaries, with synthetic violations as canaries."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/ai_arch_toolkit/core"


def imported_modules(source: str, package: str) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            modules.append(resolve_name(name, package) if node.level else name)
    return modules


def forbidden_toolkit_imports(source: str, package: str) -> list[str]:
    return [
        name
        for name in imported_modules(source, package)
        if name == "ai_arch_toolkit.toolkit" or name.startswith("ai_arch_toolkit.toolkit.")
    ]


def test_core_never_imports_toolkit() -> None:
    violations = {}
    for path in CORE.rglob("*.py"):
        package = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts[:-1])
        found = forbidden_toolkit_imports(path.read_text(), package)
        if found:
            violations[str(path.relative_to(ROOT))] = found
    assert not violations


def test_import_boundary_detector_rejects_absolute_and_relative_imports() -> None:
    assert forbidden_toolkit_imports(
        "from ai_arch_toolkit.toolkit.budget import BudgetPolicy", "ai_arch_toolkit.core"
    )
    assert forbidden_toolkit_imports("from ..toolkit import budget", "ai_arch_toolkit.core")
    assert not forbidden_toolkit_imports("from ._response import Usage", "ai_arch_toolkit.core")


def capability_discovery(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("getattr", "hasattr")
    ]


def test_call_pipeline_has_explicit_capabilities() -> None:
    for name in ("_llm.py", "_attempts.py", "_response.py", "_sync.py"):
        path = CORE / name
        assert path.exists(), name
        assert not capability_discovery(path.read_text()), name


def test_capability_detector_rejects_magic_finalizer_hooks() -> None:
    assert capability_discovery('getattr(finalizer, "_stream_abandon", None)')
    assert capability_discovery('hasattr(controller, "failure_bound")')
    assert not capability_discovery("lifecycle.abandon()")


def test_call_facade_is_small_and_functions_are_bounded() -> None:
    facade = CORE / "_llm.py"
    assert len(facade.read_text().splitlines()) < 800
    for path in (facade, CORE / "_attempts.py"):
        tree = ast.parse(path.read_text())
        lengths = {
            f"{node.name}:{node.lineno}": node.end_lineno - node.lineno + 1
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert max(lengths.values()) <= 60, lengths


def test_provider_dispatch_and_attempt_start_have_one_home() -> None:
    attempts = ast.parse((CORE / "_attempts.py").read_text())
    facade = ast.parse((CORE / "_llm.py").read_text())
    dispatch = next(
        node
        for node in attempts.body
        if isinstance(node, ast.FunctionDef) and node.name == "dispatch"
    )

    def provider_calls(tree: ast.AST) -> set[int]:
        return {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("complete", "stream", "stream_events")
            and ast.unparse(node.func.value) in ("provider", "self._provider")
        }

    assert len(provider_calls(dispatch)) == 3
    assert provider_calls(attempts) == provider_calls(dispatch)
    assert not provider_calls(facade)
    starts = [
        node
        for node in ast.walk(attempts)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mark_started"
    ]
    assert len(starts) == 1
    old_names = {
        "_try_with_tracking",
        "_StreamRun",
        "_FallbackStreamRun",
        "_single_stream",
        "_stream_with_fallbacks",
        "_meter_request",
    }
    assert (
        not {
            node.name
            for node in ast.walk(facade)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        & old_names
    )
