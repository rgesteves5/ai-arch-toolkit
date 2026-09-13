"""BBEH solvers approve python_repl, so the Python the model writes actually runs."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._decorator import tool


@pytest.fixture
def solvers(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Import the solvers with Inspect stubbed out; it is not a dependency of the test suite."""
    inspect_ai = ModuleType("inspect_ai")
    inspect_solver = ModuleType("inspect_ai.solver")
    inspect_solver.TaskState = object  # type: ignore[attr-defined]
    inspect_solver.solver = lambda fn: fn  # type: ignore[attr-defined]
    inspect_ai.solver = inspect_solver  # type: ignore[attr-defined]
    inspect_evals = ModuleType("inspect_evals")
    inspect_bbeh = ModuleType("inspect_evals.bbeh")
    inspect_bbeh.bbeh_mini = object  # type: ignore[attr-defined]
    inspect_evals.bbeh = inspect_bbeh  # type: ignore[attr-defined]
    stubs = {
        "inspect_ai": inspect_ai,
        "inspect_ai.solver": inspect_solver,
        "inspect_evals": inspect_evals,
        "inspect_evals.bbeh": inspect_bbeh,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    loaded = set(sys.modules)

    yield importlib.import_module("ai_arch_toolkit.nanope.bbeh._solvers")

    for name in set(sys.modules) - loaded:
        if name.startswith("ai_arch_toolkit.nanope.bbeh"):
            del sys.modules[name]


def test_a_solver_tool_group_runs_python_repl(solvers: Any) -> None:
    group = solvers._solver_tools(solvers.think, solvers.python_repl)

    result = group.execute(ToolCall(id="tc1", name="python_repl", input={"code": "2 + 2"}))

    assert result.ok, result.to_model_text()
    assert result.value == "4"


def test_other_tools_that_require_approval_stay_denied(solvers: Any) -> None:
    @tool(requires_approval=True)
    def deploy(target: str) -> str:
        """Deploy somewhere."""
        return target

    group = solvers._solver_tools(deploy)

    result = group.execute(ToolCall(id="tc1", name="deploy", input={"target": "prod"}))

    assert result.error is not None
    assert result.error.type == "approval_denied"
