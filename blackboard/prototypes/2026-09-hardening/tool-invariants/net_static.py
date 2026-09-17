"""Static survey of network call sites in toolkit/tools (read-only)."""

from __future__ import annotations

import ast
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(
    "/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/src/ai_arch_toolkit/toolkit/tools"
)
URLISH_PARAM = re.compile(
    r"(url|uri|host|endpoint|api|base|server|domain|site|lang|wiki|project|country|locale|"
    r"subdomain|instance|mirror|region)",
    re.I,
)


def dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def is_tool(fn: ast.FunctionDef) -> bool:
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if dotted(target) == "tool":
            return True
    return False


totals = Counter()
per_module: dict[str, dict] = {}
dyn_host: list[str] = []
urlish_params: list[str] = []
hosts_by_module: dict[str, set[str]] = defaultdict(set)
helpers_by_module: dict[str, list[str]] = defaultdict(list)
header_keys: Counter[str] = Counter()
cross_module_imports: list[str] = []

for path in sorted(ROOT.glob("_*.py")):
    if path.name == "__init__.py":
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"))
    info = {"urlopen": 0, "urlopen_no_timeout": 0, "read_unbounded": 0, "read_bounded": 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "ai_arch_toolkit.toolkit"
        ):
            cross_module_imports.append(f"{path.name}:{node.lineno} from {node.module}")
        if isinstance(node, ast.Call):
            name = dotted(node.func) or ""
            if name.endswith("urlopen"):
                info["urlopen"] += 1
                if not any(k.arg == "timeout" for k in node.keywords) and len(node.args) < 3:
                    info["urlopen_no_timeout"] += 1
                    print(f"NO TIMEOUT {path.name}:{node.lineno}")
            if name.endswith(".read") and isinstance(node.func, ast.Attribute):
                owner = dotted(node.func.value) or ""
                if owner in {"resp", "response", "r", "error", "e", "err", "exc"}:
                    if node.args or node.keywords:
                        info["read_bounded"] += 1
                    else:
                        info["read_unbounded"] += 1
            if name.endswith("Request"):
                for k in node.keywords:
                    if k.arg == "headers" and isinstance(k.value, ast.Dict):
                        for key in k.value.keys:
                            if isinstance(key, ast.Constant):
                                header_keys[str(key.value)] += 1
                    elif k.arg == "headers":
                        header_keys[f"<dynamic:{ast.unparse(k.value)}>"] += 1
            if name in {
                "urllib.request.build_opener",
                "urllib.request.install_opener",
            } or name.endswith("HTTPRedirectHandler"):
                print(f"OPENER/REDIRECT HANDLER {path.name}:{node.lineno} {name}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in re.finditer(r"https?://[^\s\"'<>{}]+", node.value):
                host = urllib.parse.urlparse(m.group(0)).netloc
                if host:
                    hosts_by_module[path.name].add(host)
        if isinstance(node, ast.JoinedStr) and node.values:
            first = node.values[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and re.match(r"^https?://[^/]*$", first.value)
                and len(node.values) > 1
            ):
                dyn_host.append(f"{path.name}:{node.lineno}  {ast.unparse(node)[:110]}")
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            calls = {dotted(c.func) or "" for c in ast.walk(fn) if isinstance(c, ast.Call)}
            if any(c.endswith("urlopen") for c in calls):
                helpers_by_module[path.name].append(f"{fn.name}@{fn.lineno}")
            if is_tool(fn):
                for a in [*fn.args.args, *fn.args.kwonlyargs]:
                    if URLISH_PARAM.search(a.arg):
                        urlish_params.append(f"{path.name}:{fn.lineno} {fn.name}({a.arg})")
    if info["urlopen"]:
        per_module[path.name] = info
        totals.update(info)

print("== totals ==", dict(totals), "modules with urlopen:", len(per_module))
print("== request header keys ==", dict(header_keys))
print("== cross-module toolkit imports inside tools/ (shared helper?) ==")
for c in cross_module_imports:
    if "core" not in c:
        print("  ", c)
print("== functions that call urlopen, per module ==")
for mod, fns in helpers_by_module.items():
    print(f"  {mod}: {fns}")
print("== f-strings with a dynamic host ==")
for d in dyn_host:
    print("  ", d)
print("== tool params with URL/host-ish names ==")
for u in urlish_params:
    print("  ", u)
print("== hosts in string constants per module ==")
for mod, hosts in sorted(hosts_by_module.items()):
    print(f"  {mod}: {sorted(hosts)}")
