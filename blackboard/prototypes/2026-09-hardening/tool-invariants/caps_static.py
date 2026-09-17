"""Static check: is every integer 'size' parameter of a tool clamped or range-checked locally?"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path("/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/src/ai_arch_toolkit/toolkit/tools")
SIZEY = re.compile(r"^(max_|limit$|count$|results$|rows$|days$|past_days$|forecast_days$|scan_pages$|context_segments$|last_time_periods$|timeout$)")


def dotted(n: ast.AST) -> str:
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return ".".join(reversed(parts))


def is_tool(fn: ast.FunctionDef) -> bool:
    return any(dotted(d.func if isinstance(d, ast.Call) else d) == "tool" for d in fn.decorator_list)


def uses(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


unclamped, clamped, char_caps = [], [], []
for path in sorted(ROOT.glob("_*.py")):
    tree = ast.parse(path.read_text())
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and is_tool(fn)):
            continue
        params = [a for a in [*fn.args.args, *fn.args.kwonlyargs]]
        ints = [a.arg for a in params if a.annotation is not None and ast.unparse(a.annotation) == "int"]
        if any(a.arg == "max_chars" for a in params):
            char_caps.append(f"{fn.name}")
        for p in ints:
            if not SIZEY.search(p):
                continue
            guarded = False
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    callee = dotted(node.func)
                    if callee in {"min", "max"} or re.search(r"bound|clamp|limit|cap|valid|check|normal|page_size|coerce", callee, re.I):
                        if any(uses(a, p) for a in node.args) or any(uses(k.value, p) for k in node.keywords):
                            guarded = True
                if isinstance(node, ast.Compare) and uses(node, p):
                    ops = {type(o) for o in node.ops}
                    if ops & {ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.NotIn, ast.In}:
                        # 'len(x) > max_chars' is a use of the cap, not a guard on the cap itself
                        left_is_param = isinstance(node.left, ast.Name) and node.left.id == p
                        right_is_const = all(isinstance(c, (ast.Constant, ast.Name, ast.UnaryOp)) for c in node.comparators)
                        if left_is_param and right_is_const:
                            guarded = True
            (clamped if guarded else unclamped).append(f"{path.name}:{fn.lineno} {fn.name}({p})")

print(f"size-like int params: {len(clamped) + len(unclamped)}  clamped/range-checked: {len(clamped)}  UNCLAMPED: {len(unclamped)}")
for u in unclamped:
    print("  UNCLAMPED", u)
print(f"tools exposing a max_chars param: {len(char_caps)} -> {char_caps}")
