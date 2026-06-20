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
# Scoped AST walker
# ---------------------------------------------------------------------------

_SENTINEL = object()


class _SafeEvaluator:
    """Evaluate AST nodes with a local scope for variable bindings."""

    def __init__(self) -> None:
        self.scope: dict[str, Any] = {
            "True": True,
            "False": False,
            "None": None,
            "print": self._print,
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
        """Recursively evaluate an AST node."""
        # --- Module (multi-line) ---
        if isinstance(node, ast.Module):
            if len(node.body) > _MAX_STATEMENTS:
                raise ValueError(f"Too many statements: {len(node.body)}")
            for stmt in node.body:
                self._exec_stmt(stmt)
            return self._last_expr_value

        # --- Expression wrapper ---
        if isinstance(node, ast.Expression):
            self._last_expr_value = self._eval_expr(node.body)
            return self._last_expr_value

        return self._eval_expr(node)

    def _exec_stmt(self, node: ast.stmt) -> None:
        """Execute a statement node."""
        # --- Expression statement (last value becomes result) ---
        if isinstance(node, ast.Expr):
            self._last_expr_value = self._eval_expr(node.value)
            return

        # --- Assignment: x = 5, a, b = 1, 2 ---
        if isinstance(node, ast.Assign):
            value = self._eval_expr(node.value)
            for target in node.targets:
                self._assign(target, value)
            return

        # --- Augmented assignment: x += 1 ---
        if isinstance(node, ast.AugAssign):
            current = self._eval_expr(node.target)
            value = self._eval_expr(node.value)
            op_fn = _BINARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported augmented op: {type(node.op).__name__}")
            result = op_fn(current, value)
            self._assign(node.target, result)
            return

        # --- For loop ---
        if isinstance(node, ast.For):
            iterable = self._eval_expr(node.iter)
            for count, item in enumerate(iterable, 1):
                if count > _MAX_FOR_ITERATIONS:
                    raise ValueError(f"For loop exceeded {_MAX_FOR_ITERATIONS} iterations")
                self._assign(node.target, item)
                for stmt in node.body:
                    self._exec_stmt(stmt)
            if node.orelse:
                for stmt in node.orelse:
                    self._exec_stmt(stmt)
            return

        # --- If/elif/else ---
        if isinstance(node, ast.If):
            if self._eval_expr(node.test):
                for stmt in node.body:
                    self._exec_stmt(stmt)
            elif node.orelse:
                for stmt in node.orelse:
                    self._exec_stmt(stmt)
            return

        # --- Pass ---
        if isinstance(node, ast.Pass):
            return

        # --- Delete ---
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.scope.pop(target.id, None)
                elif isinstance(target, ast.Subscript):
                    obj = self._eval_expr(target.value)
                    idx = self._eval_expr(target.slice)
                    del obj[idx]
            return

        raise ValueError(f"Unsupported statement: {type(node).__name__}")

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

    def _eval_expr(self, node: ast.AST) -> Any:
        """Evaluate an expression node."""
        # --- Literals ---
        if isinstance(node, ast.Constant):
            return node.value

        # --- Names ---
        if isinstance(node, ast.Name):
            if node.id in self.scope:
                return self.scope[node.id]
            if node.id in _SAFE_FUNCTIONS:
                return _SAFE_FUNCTIONS[node.id]
            if node.id in _BLOCKED_FUNCTIONS:
                raise ValueError(f"Blocked function: {node.id}")
            raise ValueError(f"Unknown name: {node.id}")

        # --- List / Tuple / Set literals ---
        if isinstance(node, ast.List):
            result = [self._eval_expr(e) for e in node.elts]
            if len(result) > _MAX_COLLECTION:
                raise ValueError(f"Collection too large: {len(result)}")
            return result

        if isinstance(node, ast.Tuple):
            return tuple(self._eval_expr(e) for e in node.elts)

        if isinstance(node, ast.Set):
            return {self._eval_expr(e) for e in node.elts}

        # --- Dict literal ---
        if isinstance(node, ast.Dict):
            keys = [self._eval_expr(k) if k is not None else None for k in node.keys]
            values = [self._eval_expr(v) for v in node.values]
            return dict(zip(keys, values, strict=False))

        # --- Binary operations ---
        if isinstance(node, ast.BinOp):
            op_fn = _BINARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            if (
                isinstance(node.op, ast.Pow)
                and isinstance(right, (int, float))
                and abs(right) > _MAX_POWER
            ):
                raise ValueError(f"Exponent too large: {right}")
            return op_fn(left, right)

        # --- Unary operations ---
        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_fn(self._eval_expr(node.operand))

        # --- Boolean: and, or ---
        if isinstance(node, ast.BoolOp):
            values = [self._eval_expr(v) for v in node.values]
            if isinstance(node.op, ast.And):
                result = values[0]
                for v in values[1:]:
                    result = result and v
                return result
            if isinstance(node.op, ast.Or):
                result = values[0]
                for v in values[1:]:
                    result = result or v
                return result
            raise ValueError(f"Unsupported boolean op: {type(node.op).__name__}")

        # --- Comparisons ---
        if isinstance(node, ast.Compare):
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

        # --- Ternary ---
        if isinstance(node, ast.IfExp):
            cond = self._eval_expr(node.test)
            return self._eval_expr(node.body) if cond else self._eval_expr(node.orelse)

        # --- Subscript ---
        if isinstance(node, ast.Subscript):
            obj = self._eval_expr(node.value)
            slc = node.slice
            if isinstance(slc, ast.Slice):
                lower = self._eval_expr(slc.lower) if slc.lower else None
                upper = self._eval_expr(slc.upper) if slc.upper else None
                step = self._eval_expr(slc.step) if slc.step else None
                return obj[lower:upper:step]
            idx = self._eval_expr(slc)
            return obj[idx]

        # --- Attribute access (safe methods only) ---
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in _BLOCKED_ATTRS or attr.startswith("__"):
                raise ValueError(f"Blocked attribute: {attr}")
            obj = self._eval_expr(node.value)
            if isinstance(obj, str) and attr in _SAFE_STR_METHODS:
                return getattr(obj, attr)
            if isinstance(obj, list) and attr in _SAFE_LIST_METHODS:
                return getattr(obj, attr)
            if isinstance(obj, dict) and attr in _SAFE_DICT_METHODS:
                return getattr(obj, attr)
            if isinstance(obj, set) and attr in _SAFE_SET_METHODS:
                return getattr(obj, attr)
            if isinstance(obj, tuple) and attr in {"count", "index"}:
                return getattr(obj, attr)
            if isinstance(obj, bytes) and attr in {"decode"}:
                return getattr(obj, attr)
            # re module access
            if obj is re and attr in {"match", "search", "findall", "sub", "split"}:
                return getattr(obj, attr)
            raise ValueError(f"Attribute not allowed: {type(obj).__name__}.{attr}")

        # --- Function calls ---
        if isinstance(node, ast.Call):
            func = self._eval_expr(node.func)
            # Check if it's a blocked function by name
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_FUNCTIONS:
                raise ValueError(f"Blocked function: {node.func.id}")
            # Allow safe functions and bound methods
            is_safe_func = func in _SAFE_FUNCTIONS.values()
            is_bound_method = callable(func) and hasattr(func, "__self__")
            is_re_func = callable(func) and getattr(func, "__module__", "") == "re"
            if not (is_safe_func or is_bound_method or is_re_func):
                raise ValueError(f"Function not allowed: {ast.dump(node.func)}")
            args = [self._eval_expr(a) for a in node.args]
            kwargs = {kw.arg: self._eval_expr(kw.value) for kw in node.keywords if kw.arg}
            # Guard range()
            if func is range:
                test_args = list(args)
                if len(test_args) == 1 and isinstance(test_args[0], int):
                    if test_args[0] > _MAX_RANGE:
                        raise ValueError(f"range too large: {test_args[0]}")
                elif len(test_args) >= 2:
                    start = test_args[0] if isinstance(test_args[0], int) else 0
                    stop = test_args[1] if isinstance(test_args[1], int) else 0
                    if abs(stop - start) > _MAX_RANGE:
                        raise ValueError(f"range too large: {abs(stop - start)}")
            return func(*args, **kwargs)

        # --- List comprehension ---
        if isinstance(node, ast.ListComp):
            return self._eval_comprehension(node.elt, node.generators)

        # --- Set comprehension ---
        if isinstance(node, ast.SetComp):
            return set(self._eval_comprehension(node.elt, node.generators))

        # --- Dict comprehension ---
        if isinstance(node, ast.DictComp):
            keys_vals = self._eval_dict_comprehension(node.key, node.value, node.generators)
            return dict(keys_vals)

        # --- Generator expression (evaluate as list) ---
        if isinstance(node, ast.GeneratorExp):
            return self._eval_comprehension(node.elt, node.generators)

        # --- Formatted string (f-string) — blocked ---
        if isinstance(node, ast.JoinedStr):
            raise ValueError("f-strings are not supported")

        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    def _eval_comprehension(self, elt: ast.AST, generators: list[ast.comprehension]) -> list[Any]:
        results: list[Any] = []
        self._eval_comp_recursive(elt, generators, 0, results)
        if len(results) > _MAX_COLLECTION:
            raise ValueError(f"Comprehension produced too many items: {len(results)}")
        return results

    def _eval_comp_recursive(
        self,
        elt: ast.AST,
        generators: list[ast.comprehension],
        gen_idx: int,
        results: list[Any],
    ) -> None:
        if gen_idx >= len(generators):
            results.append(self._eval_expr(elt))
            return

        gen = generators[gen_idx]
        iterable = self._eval_expr(gen.iter)
        saved: dict[str, Any] = {}

        for item in iterable:
            self._assign_comp(gen.target, item, saved)
            if all(self._eval_expr(cond) for cond in gen.ifs):
                self._eval_comp_recursive(elt, generators, gen_idx + 1, results)

        # Restore scope
        for name in saved:
            if saved[name] is _SENTINEL:
                self.scope.pop(name, None)
            else:
                self.scope[name] = saved[name]

    def _eval_dict_comprehension(
        self,
        key_node: ast.AST,
        value_node: ast.AST,
        generators: list[ast.comprehension],
    ) -> list[tuple[Any, Any]]:
        results: list[tuple[Any, Any]] = []
        self._eval_dict_comp_recursive(key_node, value_node, generators, 0, results)
        if len(results) > _MAX_COLLECTION:
            raise ValueError(f"Dict comprehension too large: {len(results)}")
        return results

    def _eval_dict_comp_recursive(
        self,
        key_node: ast.AST,
        value_node: ast.AST,
        generators: list[ast.comprehension],
        gen_idx: int,
        results: list[tuple[Any, Any]],
    ) -> None:
        if gen_idx >= len(generators):
            k = self._eval_expr(key_node)
            v = self._eval_expr(value_node)
            results.append((k, v))
            return

        gen = generators[gen_idx]
        iterable = self._eval_expr(gen.iter)
        saved: dict[str, Any] = {}

        for item in iterable:
            self._assign_comp(gen.target, item, saved)
            if all(self._eval_expr(cond) for cond in gen.ifs):
                self._eval_dict_comp_recursive(
                    key_node, value_node, generators, gen_idx + 1, results
                )

        for name in saved:
            if saved[name] is _SENTINEL:
                self.scope.pop(name, None)
            else:
                self.scope[name] = saved[name]

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run_eval(code: str) -> _SafeEvaluator:
    """Set up evaluator and run code, returning the evaluator for result access."""
    evaluator = _SafeEvaluator()
    evaluator.scope["re"] = re

    try:
        tree = ast.parse(code, mode="eval")
        evaluator.eval_node(tree)
    except SyntaxError:
        tree = ast.parse(code, mode="exec")
        evaluator.eval_node(tree)

    return evaluator


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
