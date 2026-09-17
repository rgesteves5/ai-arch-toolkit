"""Parent: run harness_child.py once per exported tool, hard-killed after 12 s. Aggregates."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
SANDBOX = HERE / "sandbox" / "a" / "b" / "c"
RESULTS = HERE / "results"
SKIP = {"run_command", "python_repl"}  # per instructions: process exec / code eval
HARD_TIMEOUT_S = 12


def tools() -> list[str]:
    names: list[str] = []
    for ns in ("ai_arch_toolkit.toolkit.tools", "ai_arch_toolkit.toolkit.tools.dangerous"):
        names += list(importlib.import_module(ns).__all__)
    return [n for n in names if n not in SKIP]


def run(name: str) -> tuple[str, str]:
    out_path = RESULTS / f"{name}.jsonl"
    start = time.monotonic()
    status = "ok"
    with out_path.open("w") as fh:
        try:
            proc = subprocess.run(
                [sys.executable, str(HERE / "harness_child.py"), name, str(SANDBOX)],
                stdout=fh, stderr=subprocess.PIPE, timeout=HARD_TIMEOUT_S, check=False,
            )
            if proc.returncode != 0:
                status = f"exit {proc.returncode}: {proc.stderr.decode()[-300:]}"
        except subprocess.TimeoutExpired:
            status = "HARD_TIMEOUT"
    return name, f"{status} ({time.monotonic() - start:.1f}s)"


def main() -> None:
    names = tools()
    only = sys.argv[1:]
    if only:
        names = [n for n in names if n in only]
    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = dict(pool.map(run, names))
    bad = {k: v for k, v in statuses.items() if not v.startswith("ok")}
    print(f"tools run: {len(names)}  non-ok child status: {json.dumps(bad, indent=1)}")


if __name__ == "__main__":
    main()
