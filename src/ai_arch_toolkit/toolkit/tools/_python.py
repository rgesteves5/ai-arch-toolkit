"""Safe Python REPL — sandboxed code execution via AST walking.

Parses Python code into an AST (abstract syntax tree), then walks each node
recursively. Only explicitly whitelisted node types are evaluated. Any
unrecognized node is rejected — the code never reaches Python's eval/exec.

Behaves like a Jupyter cell: captures all print() output AND auto-displays
the last expression value.

Supported:
    - Literals: int, float, str, bool, None, list, tuple, dict, set
    - Arithmetic: +, -, *, /, //, %, **
    - Bitwise: &, |, ^, <<, >>
    - Boolean: and, or, not
    - Comparison: ==, !=, <, <=, >, >=, in, not in, is, is not
    - Collections: [1,2,3], (1,2), {"a": 1}, {1,2,3}
    - Indexing/slicing: items[0], items[1:3], items[-1]
    - Comprehensions: [x for x in items if x > 2]
    - Ternary: x if cond else y
    - Assignment: x = 5, a, b = b, a
    - Augmented assignment: x += 1, x -= 2
    - Multi-line: statements separated by newlines, last expression is result
    - For loops: for x in items: ...
    - If/elif/else: if cond: ... elif: ... else: ...
    - Safe string methods: .split, .join, .replace, .upper, .lower, .strip,
      .lstrip, .rstrip, .startswith, .endswith, .count, .find, .index,
      .isdigit, .isalpha, .isalnum, .title, .capitalize, .swapcase, .zfill,
      .center, .ljust, .rjust
    - Safe list methods: .append, .extend, .insert, .pop, .remove, .copy,
      .index, .count, .reverse, .sort
    - Safe dict methods: .keys, .values, .items, .get, .pop, .update, .copy
    - Whitelisted functions: len, sorted, reversed, sum, min, max, abs, round,
      range, enumerate, zip, all, any, int, float, str, bool, list, tuple,
      set, dict, type, isinstance, ord, chr, map, filter, print, hash,
      math.gcd, math.lcm, math.isqrt, math.factorial, math.comb, math.perm

Blocked:
    - Imports (import, from...import)
    - Dangerous attribute access (x.__class__, x.__dict__)
    - exec/eval/compile/open/__import__/globals/locals/getattr/setattr/delattr
    - f-strings
    - Walrus operator (:=)
    - Star expressions (*args, **kwargs)
    - While loops (infinite loop risk)
    - Class/function definitions
"""

from __future__ import annotations

import ast
import math
import operator
import re
import types
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core import tool

# ---------------------------------------------------------------------------
# Whitelisted operations
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_COMPARE_OPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    # types / constructors
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
    "type": type,
    # collections
    "len": len,
    "sorted": sorted,
    "reversed": lambda x: list(reversed(x)),
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "range": range,
    "enumerate": lambda x: list(enumerate(x)),
    "zip": lambda *a: list(zip(*a, strict=False)),
    "map": map,
    "filter": filter,
    # logic
    "all": all,
    "any": any,
    "isinstance": isinstance,
    "hash": hash,
    # string
    "ord": ord,
    "chr": chr,
    # math
    "gcd": math.gcd,
    "lcm": math.lcm,
    "isqrt": math.isqrt,
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
}

# Safe methods that can be called on objects via attribute access
_SAFE_STR_METHODS: set[str] = {
    "split",
    "rsplit",
    "join",
    "replace",
    "upper",
    "lower",
    "strip",
    "lstrip",
    "rstrip",
    "startswith",
    "endswith",
    "count",
    "find",
    "rfind",
    "index",
    "rindex",
    "isdigit",
    "isalpha",
    "isalnum",
    "isnumeric",
    "isdecimal",
    "isspace",
    "isupper",
    "islower",
    "title",
    "capitalize",
    "swapcase",
    "zfill",
    "center",
    "ljust",
    "rjust",
    "partition",
    "rpartition",
    "removeprefix",
    "removesuffix",
    "encode",
}

_SAFE_LIST_METHODS: set[str] = {
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "copy",
    "index",
    "count",
    "reverse",
    "sort",
    "clear",
}

_SAFE_DICT_METHODS: set[str] = {
    "keys",
    "values",
    "items",
    "get",
    "pop",
    "update",
    "copy",
    "clear",
    "setdefault",
}

_SAFE_SET_METHODS: set[str] = {
    "add",
    "remove",
    "discard",
    "pop",
    "clear",
    "copy",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "issubset",
    "issuperset",
    "isdisjoint",
}

# Dunder attributes that are NEVER accessible
_BLOCKED_ATTRS: set[str] = {
    "__class__",
    "__dict__",
    "__module__",
    "__bases__",
    "__subclasses__",
    "__init__",
    "__new__",
    "__del__",
    "__repr__",
    "__str__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "__mro__",
    "__reduce__",
    "__reduce_ex__",
}

# Functions that are NEVER callable
_BLOCKED_FUNCTIONS: set[str] = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "vars",
    "dir",
    "breakpoint",
    "exit",
    "quit",
    "input",
    "memoryview",
    "bytearray",
    "classmethod",
    "staticmethod",
    "property",
    "super",
}

# Max values to prevent resource exhaustion
_MAX_RANGE = 10_000
_MAX_COLLECTION = 10_000
_MAX_POWER = 1000
_MAX_STATEMENTS = 100
_MAX_FOR_ITERATIONS = 10_000


# ---------------------------------------------------------------------------
# What attribute access may reach: a safe method of a builtin value, or a re function
# ---------------------------------------------------------------------------

_METHODS: tuple[tuple[type, dict[str, Any]], ...] = tuple(
    (kind, {name: vars(kind)[name] for name in sorted(names)})
    for kind, names in (
        (str, _SAFE_STR_METHODS),
        (list, _SAFE_LIST_METHODS),
        (dict, _SAFE_DICT_METHODS),
        (set, _SAFE_SET_METHODS),
        (tuple, {"count", "index"}),
        (bytes, {"decode"}),
    )
)
_RE_FUNCTIONS: dict[str, Any] = {
    name: vars(re)[name] for name in ("findall", "match", "search", "split", "sub")
}


def _method(owner: object, name: str) -> Any:
    """``owner.name`` when it is allowed (a bound safe method or a re function), else ``None``."""
    if owner is re:
        return _RE_FUNCTIONS.get(name)
    for kind, methods in _METHODS:
        if isinstance(owner, kind) and name in methods:
            return methods[name].__get__(owner, kind)
    return None


# ---------------------------------------------------------------------------
# Scoped AST walker: a handler per allowed node (``_EXPRESSIONS``, ``_STATEMENTS``)
# ---------------------------------------------------------------------------

_SENTINEL = object()


class _SafeEvaluator:
    """Evaluate AST nodes with a local scope for variable bindings."""

    def __init__(self) -> None:
        self._print_function = self._print
        self.scope: dict[str, Any] = {
            "True": True,
            "False": False,
            "None": None,
            "print": self._print_function,
        }
        self._output: list[str] = []
        self._last_expr_value: Any = None

    def _print(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
        """Capture print output into buffer instead of real stdout."""
        text = sep.join(str(a) for a in args) + end
        self._output.append(text)

    def get_result(self) -> str:
        """Compose return value from captured prints and last expression."""
        captured = "".join(self._output).rstrip("\n")
        last = self._last_expr_value
        has_output = bool(captured)
        has_value = last is not None

        if has_output and has_value:
            return f"{captured}\n\n{last}"
        if has_output:
            return captured
        if has_value:
            return str(last)
        return "None"

    def eval_node(self, node: ast.AST) -> Any:
        """Evaluate a parsed program: a module of statements, or one expression."""
        if isinstance(node, ast.Module):
            if len(node.body) > _MAX_STATEMENTS:
                raise ValueError(f"Too many statements: {len(node.body)}")
            self._run(node.body)
            return self._last_expr_value
        if isinstance(node, ast.Expression):
            self._last_expr_value = self._eval_expr(node.body)
            return self._last_expr_value
        return self._eval_expr(node)

    def _eval_expr(self, node: ast.AST) -> Any:
        """Evaluate an expression node with its handler; a node without one is refused."""
        handler = _EXPRESSIONS.get(type(node))
        if handler is None:
            raise ValueError(
                _REFUSALS.get(type(node), f"Unsupported expression: {type(node).__name__}")
            )
        return handler(self, node)

    def _exec_stmt(self, node: ast.AST) -> None:
        """Execute a statement node with its handler; a node without one is refused."""
        handler = _STATEMENTS.get(type(node))
        if handler is None:
            raise ValueError(f"Unsupported statement: {type(node).__name__}")
        handler(self, node)

    def _run(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._exec_stmt(statement)

    # --- statements -------------------------------------------------------

    def _expression_statement(self, node: ast.Expr) -> None:
        self._last_expr_value = self._eval_expr(node.value)

    def _assign_statement(self, node: ast.Assign) -> None:
        value = self._eval_expr(node.value)
        for target in node.targets:
            self._assign(target, value)

    def _augmented_assignment(self, node: ast.AugAssign) -> None:
        current = self._eval_expr(node.target)
        value = self._eval_expr(node.value)
        self._assign(node.target, _binary(node.op, current, value))

    def _for(self, node: ast.For) -> None:
        for count, item in enumerate(self._eval_expr(node.iter), 1):
            if count > _MAX_FOR_ITERATIONS:
                raise ValueError(f"For loop exceeded {_MAX_FOR_ITERATIONS} iterations")
            self._assign(node.target, item)
            self._run(node.body)
        self._run(node.orelse)

    def _if(self, node: ast.If) -> None:
        self._run(node.body if self._eval_expr(node.test) else node.orelse)

    def _pass(self, node: ast.Pass) -> None:
        """``pass`` does nothing."""

    def _delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scope.pop(target.id, None)
            elif isinstance(target, ast.Subscript):
                del self._eval_expr(target.value)[self._eval_expr(target.slice)]
            else:
                raise ValueError(f"Unsupported delete target: {type(target).__name__}")

    def _assign(self, target: ast.AST, value: Any) -> None:
        """Assign a value to a target (name, tuple, list, subscript)."""
        if isinstance(target, ast.Name):
            if target.id in _BLOCKED_FUNCTIONS:
                raise ValueError(f"Cannot assign to blocked name: {target.id}")
            self.scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)):
                raise ValueError("Cannot unpack non-sequence")
            if len(target.elts) != len(value):
                raise ValueError(
                    f"Unpack mismatch: {len(target.elts)} targets, {len(value)} values"
                )
            for t, v in zip(target.elts, value, strict=True):
                self._assign(t, v)
        elif isinstance(target, ast.Subscript):
            obj = self._eval_expr(target.value)
            idx = self._eval_expr(target.slice)
            obj[idx] = value
        else:
            raise ValueError(f"Unsupported assignment target: {type(target).__name__}")

    # --- expressions ------------------------------------------------------

    def _constant(self, node: ast.Constant) -> Any:
        return node.value

    def _name(self, node: ast.Name) -> Any:
        if node.id in self.scope:
            return self.scope[node.id]
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        if node.id in _BLOCKED_FUNCTIONS:
            raise ValueError(f"Blocked function: {node.id}")
        raise ValueError(f"Unknown name: {node.id}")

    def _list(self, node: ast.List) -> list[Any]:
        result = [self._eval_expr(e) for e in node.elts]
        if len(result) > _MAX_COLLECTION:
            raise ValueError(f"Collection too large: {len(result)}")
        return result

    def _tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self._eval_expr(e) for e in node.elts)

    def _set(self, node: ast.Set) -> set[Any]:
        return {self._eval_expr(e) for e in node.elts}

    def _dict(self, node: ast.Dict) -> dict[Any, Any]:
        if any(key is None for key in node.keys):
            raise ValueError("Dict unpacking (**) is not supported")
        return {
            self._eval_expr(key): self._eval_expr(value)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }

    def _binary_operation(self, node: ast.BinOp) -> Any:
        return _binary(node.op, self._eval_expr(node.left), self._eval_expr(node.right))

    def _unary_operation(self, node: ast.UnaryOp) -> Any:
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_fn(self._eval_expr(node.operand))

    def _boolean_operation(self, node: ast.BoolOp) -> Any:
        """``and`` stops at the first false value, ``or`` at the first true one."""
        stop_when = not isinstance(node.op, ast.And)
        result: Any = None
        for value in node.values:
            result = self._eval_expr(value)
            if bool(result) is stop_when:
                return result
        return result

    def _compare(self, node: ast.Compare) -> bool:
        left = self._eval_expr(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            op_fn = _COMPARE_OPS.get(type(op))
            if op_fn is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            right = self._eval_expr(comparator)
            if not op_fn(left, right):
                return False
            left = right
        return True

    def _if_expression(self, node: ast.IfExp) -> Any:
        return self._eval_expr(node.body if self._eval_expr(node.test) else node.orelse)

    def _subscript(self, node: ast.Subscript) -> Any:
        obj = self._eval_expr(node.value)
        part = node.slice
        if not isinstance(part, ast.Slice):
            return obj[self._eval_expr(part)]
        bounds = (part.lower, part.upper, part.step)
        lower, upper, step = (self._eval_expr(b) if b is not None else None for b in bounds)
        return obj[lower:upper:step]

    def _attribute(self, node: ast.Attribute) -> Any:
        if node.attr in _BLOCKED_ATTRS or node.attr.startswith("__"):
            raise ValueError(f"Blocked attribute: {node.attr}")
        owner = self._eval_expr(node.value)
        method = _method(owner, node.attr)
        if method is None:
            raise ValueError(f"Attribute not allowed: {type(owner).__name__}.{node.attr}")
        return method

    def _call(self, node: ast.Call) -> Any:
        func = self._eval_expr(node.func)
        if not self._vetted(func):
            raise ValueError(f"Function not allowed: {ast.dump(node.func)}")
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError("Keyword unpacking (**) is not supported")
        args = [self._eval_expr(a) for a in node.args]
        kwargs = {kw.arg: self._eval_expr(kw.value) for kw in node.keywords if kw.arg}
        if func is range:
            _check_range(args)
        return func(*args, **kwargs)

    def _vetted(self, func: object) -> bool:
        """Whether ``func`` is one the evaluator hands out: a safe function, ``print``, a re
        function, or a safe method bound to a builtin value."""
        known = (self._print_function, *_SAFE_FUNCTIONS.values(), *_RE_FUNCTIONS.values())
        if any(func is callable_ for callable_ in known):
            return True
        return (
            isinstance(func, types.BuiltinMethodType)
            and _method(func.__self__, func.__name__) is not None
        )

    def _list_comprehension(self, node: ast.ListComp) -> list[Any]:
        return self._comprehend(node.generators, lambda: self._eval_expr(node.elt))

    def _set_comprehension(self, node: ast.SetComp) -> set[Any]:
        return set(self._comprehend(node.generators, lambda: self._eval_expr(node.elt)))

    def _dict_comprehension(self, node: ast.DictComp) -> dict[Any, Any]:
        def item() -> tuple[Any, Any]:
            return self._eval_expr(node.key), self._eval_expr(node.value)

        return dict(self._comprehend(node.generators, item))

    def _generator(self, node: ast.GeneratorExp) -> list[Any]:
        return self._comprehend(node.generators, lambda: self._eval_expr(node.elt))

    def _comprehend(
        self, generators: list[ast.comprehension], produce: Callable[[], Any]
    ) -> list[Any]:
        """What a comprehension produces, as a list; its loop names do not outlive it."""
        results: list[Any] = []
        self._generate(generators, 0, produce, results)
        if len(results) > _MAX_COLLECTION:
            raise ValueError(f"Comprehension produced too many items: {len(results)}")
        return results

    def _generate(
        self,
        generators: list[ast.comprehension],
        index: int,
        produce: Callable[[], Any],
        results: list[Any],
    ) -> None:
        if index == len(generators):
            results.append(produce())
            return
        generator = generators[index]
        saved: dict[str, Any] = {}
        for item in self._eval_expr(generator.iter):
            self._assign_comp(generator.target, item, saved)
            if all(self._eval_expr(condition) for condition in generator.ifs):
                self._generate(generators, index + 1, produce, results)
        for name, value in saved.items():
            if value is _SENTINEL:
                self.scope.pop(name, None)
            else:
                self.scope[name] = value

    def _assign_comp(self, target: ast.AST, value: Any, saved: dict[str, Any]) -> None:
        """Assign in comprehension scope, tracking previous values for restore."""
        if isinstance(target, ast.Name):
            if target.id not in saved:
                saved[target.id] = self.scope.get(target.id, _SENTINEL)
            self.scope[target.id] = value
        elif isinstance(target, ast.Tuple):
            if isinstance(value, (tuple, list)) and len(value) == len(target.elts):
                for t, v in zip(target.elts, value, strict=True):
                    self._assign_comp(t, v, saved)
        else:
            raise ValueError(f"Unsupported comp target: {type(target).__name__}")


def _binary(op: ast.operator, left: Any, right: Any) -> Any:
    """``left op right`` for a whitelisted operator; ``**`` keeps its exponent guard."""
    op_fn = _BINARY_OPS.get(type(op))
    if op_fn is None:
        raise ValueError(f"Unsupported binary op: {type(op).__name__}")
    if isinstance(op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > _MAX_POWER:
        raise ValueError(f"Exponent too large: {right}")
    return op_fn(left, right)


def _check_range(args: list[Any]) -> None:
    """Refuse a ``range`` longer than ``_MAX_RANGE``."""
    if len(args) == 1 and isinstance(args[0], int) and args[0] > _MAX_RANGE:
        raise ValueError(f"range too large: {args[0]}")
    if len(args) >= 2:
        start = args[0] if isinstance(args[0], int) else 0
        stop = args[1] if isinstance(args[1], int) else 0
        if abs(stop - start) > _MAX_RANGE:
            raise ValueError(f"range too large: {abs(stop - start)}")


# The nodes the evaluator runs: anything else is refused (see the module docstring).
_EXPRESSIONS: dict[type[ast.AST], Callable[[_SafeEvaluator, Any], Any]] = {
    ast.Constant: _SafeEvaluator._constant,
    ast.Name: _SafeEvaluator._name,
    ast.List: _SafeEvaluator._list,
    ast.Tuple: _SafeEvaluator._tuple,
    ast.Set: _SafeEvaluator._set,
    ast.Dict: _SafeEvaluator._dict,
    ast.BinOp: _SafeEvaluator._binary_operation,
    ast.UnaryOp: _SafeEvaluator._unary_operation,
    ast.BoolOp: _SafeEvaluator._boolean_operation,
    ast.Compare: _SafeEvaluator._compare,
    ast.IfExp: _SafeEvaluator._if_expression,
    ast.Subscript: _SafeEvaluator._subscript,
    ast.Attribute: _SafeEvaluator._attribute,
    ast.Call: _SafeEvaluator._call,
    ast.ListComp: _SafeEvaluator._list_comprehension,
    ast.SetComp: _SafeEvaluator._set_comprehension,
    ast.DictComp: _SafeEvaluator._dict_comprehension,
    ast.GeneratorExp: _SafeEvaluator._generator,
}
_STATEMENTS: dict[type[ast.AST], Callable[[_SafeEvaluator, Any], None]] = {
    ast.Expr: _SafeEvaluator._expression_statement,
    ast.Assign: _SafeEvaluator._assign_statement,
    ast.AugAssign: _SafeEvaluator._augmented_assignment,
    ast.For: _SafeEvaluator._for,
    ast.If: _SafeEvaluator._if,
    ast.Pass: _SafeEvaluator._pass,
    ast.Delete: _SafeEvaluator._delete,
}
_REFUSALS: dict[type[ast.AST], str] = {ast.JoinedStr: "f-strings are not supported"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@tool(
    capability="python",
    risk_level="high",
    requires_approval=True,
    approval_reason="Python execution can perform high-impact computation or resource use.",
)
def python_repl(code: str) -> str:
    """Execute Python code and return the result. Works like a Jupyter cell:
    the last expression's value is returned automatically, and print() output
    is captured and included.

    Use a bare expression on the last line to get its value — no print needed:
        python_repl("x = 2 + 2\\nx")           → "4"
        python_repl("sorted(['c','a','b'])")   → "['a', 'b', 'c']"
        python_repl("x = [1,2,3]\\nsum(x)")    → "6"

    print() also works (output is captured, not lost):
        python_repl("for x in [1,2,3]: print(x)")  → "1\\n2\\n3"

    Supports: variables, for loops, if/else, comprehensions, string/list/dict/set
    methods, regex (re), and math. No imports, no file access, no while loops.

    Args:
        code: Python code to execute. Last expression value is the result.
    """
    evaluator = _SafeEvaluator()
    evaluator.scope["re"] = re
    try:
        try:
            tree = ast.parse(code, mode="eval")
            evaluator.eval_node(tree)
        except SyntaxError:
            tree = ast.parse(code, mode="exec")
            evaluator.eval_node(tree)
        return evaluator.get_result()
    except (
        ValueError,
        TypeError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError,
        KeyError,
        IndexError,
        AttributeError,
        RuntimeError,
    ) as e:
        captured = "".join(evaluator._output).rstrip("\n")
        if captured:
            return f"{captured}\n\nError: {e}"
        return f"Error: {e}"
