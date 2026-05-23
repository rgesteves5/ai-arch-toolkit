"""Examples smoke test — catch public-API drift without spending a cent.

Walks every script under ``examples/`` and verifies:

1. The file parses as valid Python (AST).
2. Every ``from ai_arch_toolkit...`` import resolves to a real symbol on
   the named module *right now*. Renaming or deleting a public symbol
   without updating the example will fail this test.

No LLM instances are constructed, no network is touched. Runs in the
default CI matrix alongside the unit tests.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
_EXAMPLES = sorted(_EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.name)
class TestExampleSmoke:
    def test_parses(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - error path
            pytest.fail(f"{path.name} has a syntax error: {exc}")

    def test_toolkit_imports_resolve(self, path: Path) -> None:
        """Every ``from ai_arch_toolkit...`` import must resolve at import time."""
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("ai_arch_toolkit"):
                continue

            module = importlib.import_module(node.module)
            for alias in node.names:
                name = alias.name
                if name == "*":
                    continue
                if not hasattr(module, name):
                    pytest.fail(
                        f"{path.name} imports {name!r} from {node.module!r}, "
                        f"but that symbol is not exported."
                    )


def test_examples_directory_not_empty() -> None:
    """Sanity check that the parametrize above isn't silently no-op."""
    assert len(_EXAMPLES) >= 30, f"Only {len(_EXAMPLES)} examples found; expected the full set."
