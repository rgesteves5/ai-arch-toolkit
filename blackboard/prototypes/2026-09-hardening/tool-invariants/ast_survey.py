"""AST survey of toolkit.tools: capability classification per exported tool.

Read-only. Resolves every exported tool to its module + line, builds an intra-module
call graph, and classifies the transitive closure of what each tool can reach.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SAFE_NS = "ai_arch_toolkit.toolkit.tools"
DANGER_NS = "ai_arch_toolkit.toolkit.tools.dangerous"

FS_PREFIX = (
    "pathlib.", "shutil.", "tempfile.", "glob.", "sqlite3.", "zipfile.", "tarfile.",
    "fileinput.", "mmap.", "dbm.", "shelve.", "os.path.", "io.open", "gzip.open",
    "bz2.open", "lzma.open", "linecache.",
)
FS_EXACT = {
    "open", "os.listdir", "os.walk", "os.scandir", "os.remove", "os.unlink", "os.rename",
    "os.replace", "os.mkdir", "os.makedirs", "os.rmdir", "os.stat", "os.chmod", "os.chown",
    "os.getcwd", "os.chdir", "os.open", "os.fdopen", "os.link", "os.symlink", "os.readlink",
    "os.access", "os.truncate", "os.utime",
}
FS_METHODS = {
    "read_text", "read_bytes", "write_text", "write_bytes", "glob", "rglob", "iterdir",
    "unlink", "mkdir", "rmdir", "touch", "expanduser", "is_file", "is_dir", "exists",
    "resolve", "stat", "lstat", "chmod", "symlink_to", "hardlink_to", "walk",
}
PROC_PREFIX = ("subprocess.", "pty.", "multiprocessing.", "signal.", "resource.")
PROC_EXACT_RE = re.compile(r"^os\.(system|popen|exec\w*|spawn\w*|kill|fork\w*|startfile)$")
DYN_EXACT = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
DYN_PREFIX = ("importlib.", "pickle.", "marshal.", "ctypes.", "runpy.", "code.", "codeop.")
NET_PREFIX = (
    "urllib.request.", "http.client.", "socket.", "ssl.", "ftplib.", "smtplib.", "poplib.",
    "imaplib.", "xmlrpc.", "webbrowser.", "requests.", "httpx.", "aiohttp.", "telnetlib.",
    "youtube_transcript_api.",
)
ENV_EXACT = {"os.environ", "os.getenv", "os.putenv", "os.getlogin", "os.uname", "sys.argv"}
ENV_PREFIX = ("getpass.", "platform.", "os.environ.")
TIME_RE = re.compile(r"^(datetime\.(datetime\.)?(now|today|utcnow)|time\.\w+|datetime\.date\.today)$")


def dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):  # e.g. Path(x).expanduser()
        inner = dotted(node.func)
        if inner:
            parts.append(inner + "()")
            return ".".join(reversed(parts))
    return None


class ModuleInfo:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tree = ast.parse(path.read_text(encoding="utf-8"))
        self.aliases: dict[str, str] = {}
        self.funcs: dict[str, ast.AST] = {}
        self.constants: dict[str, ast.AST] = {}
        self._index()

    def _index(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.aliases[a.asname or a.name.split(".")[0]] = (
                        a.name if a.asname else a.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    self.aliases[a.asname or a.name] = f"{node.module}.{a.name}"
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.funcs[node.name] = node  # class body (methods) counts as reachable
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.funcs.setdefault(sub.name, sub)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and node.value is not None:
                        self.constants[t.id] = node.value

    def resolve(self, name: str) -> str:
        head, _, rest = name.partition(".")
        head = head.replace("()", "")
        if head in self.aliases:
            full = self.aliases[head]
            return f"{full}.{rest}" if rest else full
        if "." in name:
            return "<local>." + name  # attribute on a local object, not on an imported module
        return name

    def third_party(self) -> set[str]:
        roots = {v.split(".")[0] for v in self.aliases.values()}
        return {r for r in roots if r not in sys.stdlib_module_names and r != "ai_arch_toolkit"}

    def reach(self, root: str) -> tuple[set[str], set[str]]:
        """Return (resolved call/attr names, local function names) reachable from root."""
        seen: set[str] = set()
        names: set[str] = set()
        stack = [root]
        while stack:
            fn = stack.pop()
            if fn in seen or fn not in self.funcs:
                continue
            seen.add(fn)
            for node in ast.walk(self.funcs[fn]):
                if isinstance(node, ast.Call):
                    d = dotted(node.func)
                    if d:
                        names.add(self.resolve(d))
                elif isinstance(node, ast.Attribute):
                    d = dotted(node)
                    if d:
                        names.add(self.resolve(d))
                if isinstance(node, ast.Name) and node.id in self.funcs:
                    stack.append(node.id)
                # module-level constants referenced (e.g. lambdas / dispatch dicts)
                if isinstance(node, ast.Name) and node.id in self.constants:
                    for sub in ast.walk(self.constants[node.id]):
                        if isinstance(sub, ast.Name) and sub.id in self.funcs:
                            stack.append(sub.id)
                        if isinstance(sub, ast.Call):
                            d = dotted(sub.func)
                            if d:
                                names.add(self.resolve(d))
        return names, seen


def classify(names: set[str], mod: ModuleInfo) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    has_pathlib = any(v.startswith(("pathlib", "os")) for v in mod.aliases.values())
    for n in sorted(names):
        local = n.startswith("<local>.")
        base = n.replace("()", "").removeprefix("<local>.")
        last = base.rsplit(".", 1)[-1]
        if local:  # only pathlib-style method names count for local objects
            if has_pathlib and last in FS_METHODS:
                out["fs"].append(base)
            continue
        if base in {"ast.parse", "ast.literal_eval"}:
            out["parse_code"].append(base)
        if base in FS_EXACT or base.startswith(FS_PREFIX):
            out["fs"].append(n)
        elif has_pathlib and last in FS_METHODS and not base.startswith(("re.", "json.")):
            out["fs"].append(n)
        if base.startswith(PROC_PREFIX) or PROC_EXACT_RE.match(base):
            out["proc"].append(n)
        if base in DYN_EXACT or base.startswith(DYN_PREFIX):
            if base in dir(builtins) or base.startswith(DYN_PREFIX):
                out["dyn"].append(n)
        if base.startswith(NET_PREFIX):
            out["net"].append(n)
        if base in ENV_EXACT or base.startswith(ENV_PREFIX):
            out["env"].append(n)
        if TIME_RE.match(base) or base.startswith("zoneinfo."):
            out["time"].append(n)
    return out


def exported(ns: str) -> dict[str, object]:
    mod = importlib.import_module(ns)
    return {name: getattr(mod, name) for name in mod.__all__}


def main() -> None:
    safe = exported(SAFE_NS)
    danger = exported(DANGER_NS)
    modules: dict[str, ModuleInfo] = {}
    rows = []
    for ns_label, table in (("safe", safe), ("dangerous", danger)):
        for name, fn in sorted(table.items()):
            definition = getattr(fn, "__tool_definition__", None)
            raw = inspect.unwrap(fn)
            file = Path(inspect.getsourcefile(raw))
            line = raw.__code__.co_firstlineno
            info = modules.setdefault(str(file), ModuleInfo(file))
            names, reached = info.reach(raw.__name__)
            caps = classify(names, info)
            tp = info.third_party()
            if tp and any(f.startswith("_load_") for f in reached):
                caps["thirdparty_net"] = sorted(tp)
            policy = definition.policy if definition else None
            rows.append(
                {
                    "ns": ns_label,
                    "tool": name,
                    "schema_name": definition.schema.name if definition else None,
                    "file": file.name,
                    "line": line,
                    "caps": {k: v for k, v in caps.items()},
                    "policy": None
                    if policy is None
                    else {
                        "capability": policy.capability,
                        "risk_level": policy.risk_level,
                        "requires_approval": policy.requires_approval,
                    },
                    "params": list(definition.schema.input_schema.get("properties", {}))
                    if definition
                    else [],
                    "reached": sorted(reached),
                }
            )
    out = Path(__file__).with_name("ast_survey.json")
    out.write_text(json.dumps(rows, indent=1))

    print(f"safe exports: {len(safe)}  dangerous exports: {len(danger)}")
    no_def = [r["tool"] for r in rows if r["policy"] is None]
    print("exports without __tool_definition__:", no_def)
    mism = [(r["tool"], r["schema_name"]) for r in rows if r["schema_name"] != r["tool"]]
    print("export name != schema name:", mism)
    tally: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["ns"] != "safe":
            continue
        key = "+".join(sorted(k for k in r["caps"] if k not in {"time", "parse_code"})) or (
            "time" if "time" in r["caps"] else "pure"
        )
        tally[key] += 1
    print("safe-namespace tally by capability set:", dict(tally))
    print()
    print("== tools (both namespaces) touching fs/proc/dyn/env ==")
    for r in rows:
        hot = {k: v for k, v in r["caps"].items() if k in {"fs", "proc", "dyn", "env", "parse_code", "thirdparty_net"}}
        if hot:
            print(f"[{r['ns']}] {r['tool']}  {r['file']}:{r['line']}  policy={r['policy']}")
            for k, v in hot.items():
                print(f"      {k}: {v}")
    print()
    print("== safe tools with no net/fs/proc/dyn (pure or time) ==")
    print("  ", [r["tool"] for r in rows if r["ns"] == "safe" and not (set(r["caps"]) - {"time", "parse_code"})])
    print("== policy values across the safe namespace ==")
    from collections import Counter
    print("  ", Counter(json.dumps(r["policy"], sort_keys=True) for r in rows if r["ns"] == "safe"))
    print()
    print("== undecorated/low-policy check: dangerous namespace ==")
    for r in rows:
        if r["ns"] == "dangerous":
            print(f"  {r['tool']:16s} {r['file']}:{r['line']} {r['policy']} caps={sorted(r['caps'])}")


if __name__ == "__main__":
    sys.exit(main())
