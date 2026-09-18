"""Ruff's complexity debt is explicit and can only shrink."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("quality_baseline.json")


def function_names(tree: ast.AST, parents: tuple[str, ...] = ()) -> dict[int, str]:
    names: dict[int, str] = {}
    for node in ast.iter_child_nodes(tree):
        nested = parents
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            nested = (*parents, node.name)
            names[node.lineno] = ".".join(nested)
        names.update(function_names(node, nested))
    return names


def quality_diagnostics(paths: list[str], root: Path = ROOT) -> dict[str, int]:
    ruff = shutil.which("ruff")
    assert ruff, "ruff must be available in the dev environment"
    result = subprocess.run(
        [ruff, "check", "--select", "C901,PLR0912,PLR0915", "--output-format", "json", *paths],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    diagnostics: dict[str, int] = {}
    for item in json.loads(result.stdout):
        path = Path(item["filename"])
        names = function_names(ast.parse(path.read_text()))
        line = item["location"]["row"]
        match = re.search(r"\((\d+) > \d+\)", item["message"])
        assert match, item
        key = f"{path.relative_to(root)}:{names[line]}:{item['code']}"
        diagnostics[key] = int(match[1])
    return diagnostics


def test_quality_debt_can_only_shrink() -> None:
    actual = quality_diagnostics(["src/ai_arch_toolkit/core", "src/ai_arch_toolkit/toolkit"])
    expected = json.loads(BASELINE.read_text())
    assert actual == expected, (
        "New/worse debt is forbidden; improved or removed debt must be removed from baseline",
        {key: value for key, value in actual.items() if expected.get(key) != value},
        sorted(expected.keys() - actual.keys()),
    )


def test_quality_detector_catches_a_new_complex_function(tmp_path: Path) -> None:
    source = "def branching(x):\n" + "".join(
        f"    if x == {i}:\n        return {i}\n" for i in range(12)
    )
    path = tmp_path / "canary.py"
    path.write_text(source)
    assert quality_diagnostics([str(path)], tmp_path)["canary.py:branching:C901"] == 13
