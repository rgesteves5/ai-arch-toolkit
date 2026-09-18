"""python_repl runs the nodes in its two tables, and refuses every other node the language has."""

from __future__ import annotations

import ast

import pytest

from ai_arch_toolkit.toolkit.tools._python import (
    _EXPRESSIONS,
    _STATEMENTS,
    _SafeEvaluator,
    python_repl,
)

# Every node class the running Python parses into; a new one fails the coverage test below until
# it is placed in a table or refused on purpose.
_EXPRESSION_NODES = sorted(
    name
    for name, cls in vars(ast).items()
    if isinstance(cls, type) and issubclass(cls, ast.expr) and cls is not ast.expr
    if not name.startswith("_")
)
_STATEMENT_NODES = sorted(
    name
    for name, cls in vars(ast).items()
    if isinstance(cls, type) and issubclass(cls, ast.stmt) and cls is not ast.stmt
    if not name.startswith("_")
)

# One program per allowed node, and what python_repl answers.
_RUNS: dict[str, tuple[str, str]] = {
    "Constant": ("42", "42"),
    "Name": ("x = 3\nx", "3"),
    "List": ("[1, 2]", "[1, 2]"),
    "Tuple": ("(1, 2)", "(1, 2)"),
    "Set": ("sorted({2, 1})", "[1, 2]"),
    "Dict": ("{'a': 1}", "{'a': 1}"),
    "BinOp": ("2 * 3", "6"),
    "UnaryOp": ("-4", "-4"),
    "BoolOp": ("0 or 5", "5"),
    "Compare": ("1 < 2 < 3", "True"),
    "IfExp": ("'y' if 1 else 'n'", "y"),
    "Subscript": ("[1, 2, 3][1:]", "[2, 3]"),
    "Attribute": ("'ab'.upper()", "AB"),
    "Call": ("len('abc')", "3"),
    "ListComp": ("[x * 2 for x in [1, 2]]", "[2, 4]"),
    "SetComp": ("sorted({x % 2 for x in range(4)})", "[0, 1]"),
    "DictComp": ("{x: x for x in [1]}", "{1: 1}"),
    "GeneratorExp": ("sum(x for x in [1, 2])", "3"),
    "Expr": ("1 + 1", "2"),
    "Assign": ("a, b = 1, 2\nb", "2"),
    "AugAssign": ("n = 1\nn += 2\nn", "3"),
    "For": ("t = 0\nfor i in [1, 2]: t += i\nt", "3"),
    "If": ("if 1:\n    r = 'yes'\nr", "yes"),
    "Pass": ("pass", "None"),
    "Delete": ("d = {'k': 1}\ndel d['k']\nd", "{}"),
}


def test_every_node_of_the_language_is_run_or_refused_on_purpose() -> None:
    allowed = {cls.__name__ for cls in (*_EXPRESSIONS, *_STATEMENTS)}

    assert allowed == set(_RUNS)
    assert allowed <= {*_EXPRESSION_NODES, *_STATEMENT_NODES}


@pytest.mark.parametrize("name", sorted(_RUNS))
def test_an_allowed_node_runs(name: str) -> None:
    code, answer = _RUNS[name]

    assert python_repl(code) == answer


@pytest.mark.parametrize(
    "name", [name for name in _EXPRESSION_NODES if name not in {c.__name__ for c in _EXPRESSIONS}]
)
def test_a_refused_expression_node_is_refused(name: str) -> None:
    kind = getattr(ast, name)
    node = kind.__new__(kind)  # only its type matters to the evaluator

    with pytest.raises(ValueError, match=r"not supported|Unsupported expression"):
        _SafeEvaluator()._eval_expr(node)


@pytest.mark.parametrize(
    "name", [name for name in _STATEMENT_NODES if name not in {c.__name__ for c in _STATEMENTS}]
)
def test_a_refused_statement_node_is_refused(name: str) -> None:
    kind = getattr(ast, name)
    node = kind.__new__(kind)  # only its type matters to the evaluator

    with pytest.raises(ValueError, match=f"Unsupported statement: {name}"):
        _SafeEvaluator()._exec_stmt(node)


@pytest.mark.parametrize(
    ("code", "refusal"),
    [
        ("(y := 1)", "NamedExpr"),
        ("(lambda: 1)()", "Lambda"),
        ("f'{1}'", "f-strings are not supported"),
        ("print(*[1, 2])", "Starred"),
        ("while True: pass", "While"),
        ("def f(): pass", "FunctionDef"),
        ("class C: pass", "ClassDef"),
        ("import os", "Import"),
        ("from os import path", "ImportFrom"),
        ("with open('x') as f: pass", "With"),
        ("try:\n    1\nexcept Exception:\n    2", "Try"),
        ("raise ValueError('x')", "Raise"),
        ("assert 1", "Assert"),
        ("global g", "Global"),
        ("x: int = 1", "AnnAssign"),
        ("for i in [1]:\n    break", "Break"),
        ("match 1:\n    case 1:\n        pass", "Match"),
    ],
)
def test_a_program_using_a_refused_node_is_refused(code: str, refusal: str) -> None:
    assert refusal in python_repl(code)
