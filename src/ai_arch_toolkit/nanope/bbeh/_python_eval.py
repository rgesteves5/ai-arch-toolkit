"""Backward-compat shim — the evaluator now lives in toolkit/tools/_python.py."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._python import python_repl

# Backward-compat alias
python_eval = python_repl

__all__ = ["python_eval", "python_repl"]
