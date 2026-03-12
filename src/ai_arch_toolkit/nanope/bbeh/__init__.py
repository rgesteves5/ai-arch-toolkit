"""BBEH mini benchmark — evaluate ai-arch-toolkit flows via Inspect AI."""

from __future__ import annotations

from inspect_evals.bbeh import bbeh_mini

from ._solvers import (
    baseline_solver,
    generate_review_solver,
    react_full_solver,
    react_pyeval_only_solver,
    react_tools_solver,
    react_ts_only_solver,
    react_ts_pyeval_solver,
    self_discovery_solver,
)

__all__ = [
    "baseline_solver",
    "bbeh_task",
    "generate_review_solver",
    "react_full_solver",
    "react_pyeval_only_solver",
    "react_tools_solver",
    "react_ts_only_solver",
    "react_ts_pyeval_solver",
    "self_discovery_solver",
]


def bbeh_task(
    strategy: str = "baseline",
    model: str = "gpt-5-nano",
    **kwargs: object,
) -> object:
    """Create an Inspect task with one of our solvers.

    Args:
        strategy: One of the registered solver names.
        model: Model identifier for ai-arch-toolkit LLM.
        **kwargs: Forwarded to the solver factory.
    """
    solvers = {
        "baseline": baseline_solver,
        "self_discovery": self_discovery_solver,
        "react_tools": react_tools_solver,
        "react_ts_only": react_ts_only_solver,
        "react_pyeval_only": react_pyeval_only_solver,
        "react_ts_pyeval": react_ts_pyeval_solver,
        "react_full": react_full_solver,
        "generate_review": generate_review_solver,
    }
    if strategy not in solvers:
        raise ValueError(f"Unknown strategy {strategy!r}, choose from {list(solvers)}")
    solver = solvers[strategy]
    return bbeh_mini(solver=solver(model=model, **kwargs))
