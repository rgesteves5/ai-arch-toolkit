"""Aggregate harness results: raises, non-str returns, hangs, slow calls, big outputs, hosts."""

from __future__ import annotations

import glob
import json
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
rows, inflight, done = [], {}, set()
for f in sorted(glob.glob(str(HERE / "results" / "*.jsonl"))):
    last_start = None
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            last_start = r
        elif r.get("event") == "end":
            rows.append(r)
            last_start = None
        elif r.get("event") == "done":
            done.add(r["tool"])
        elif r.get("event") == "budget_exhausted":
            print("BUDGET EXHAUSTED (soft 8s):", r["tool"], r["mode"])
    if last_start is not None:
        inflight[last_start["tool"]] = last_start

print(f"calls completed: {len(rows)}; tools: {len({r['tool'] for r in rows})}; finished cleanly: {len(done)}")
print("\n== HANGS (call in flight when the 12 s hard kill hit) ==")
for t, r in inflight.items():
    print(f"  {t} mode={r['mode']} label={r['label']} args={r['args']}")

print("\n== slow calls (>1 s) ==")
for r in rows:
    if r["dt"] > 1.0:
        print(f"  {r['tool']} {r['mode']} {r['label']} dt={r['dt']}s outcome={r['outcome']}")

for mode in ("offline", "realopen", "badbody"):
    sel = [r for r in rows if r["mode"] == mode and r["outcome"] == "RAISED"]
    tools = sorted({r["tool"] for r in sel})
    print(f"\n== RAISED in mode={mode}: {len(sel)} calls, {len(tools)} tools ==")
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in sel:
        grouped[(r["tool"], r["exc_type"], r["where"])].append(r["label"])
    if "-v" in sys.argv or mode != "badbody":
        for (tool, exc, where), labels in sorted(grouped.items()):
            print(f"  {tool:28s} {exc:32s} {where:26s} {labels[:6]}{'…' if len(labels) > 6 else ''}")
    else:
        by_tool = Counter(t for (t, _, _) in grouped)
        excs = Counter(e for (_, e, _) in grouped)
        print("  tools:", ", ".join(tools))
        print("  exception types:", dict(excs))

nonstr = [r for r in rows if r["outcome"] == "NON_STR"]
print(f"\n== NON-STR returns: {len(nonstr)} ==")
for r in nonstr[:10]:
    print("  ", r["tool"], r["label"], r["out_type"])

print("\n== largest outputs (chars) per tool, offline/realopen modes ==")
best: dict[str, dict] = {}
for r in rows:
    if r["outcome"] == "str" and r["mode"] != "badbody":
        if r["out_len"] > best.get(r["tool"], {"out_len": -1})["out_len"]:
            best[r["tool"]] = r
big = sorted(best.values(), key=lambda r: -r["out_len"])
print("  tools echoing >= 10k chars back:", sum(1 for r in big if r["out_len"] >= 10_000))
for r in big[:12]:
    print(f"  {r['tool']:28s} {r['out_len']:>8d}  {r['label']}")

print("\n== network reach: hosts requested per tool (offline+realopen) ==")
hosts: dict[str, Counter] = defaultdict(Counter)
schemes: Counter = Counter()
reached = set()
for r in rows:
    for q in r.get("requests", []):
        u = urllib.parse.urlsplit(q["url"])
        hosts[r["tool"]][u.netloc] += 1
        schemes[u.scheme] += 1
        reached.add(r["tool"])
        if q["timeout"] is None:
            print("  NO TIMEOUT passed:", r["tool"], q["url"][:80])
print("  tools that reached urlopen at least once:", len(reached))
print("  schemes:", dict(schemes))
multi = {t: dict(c) for t, c in hosts.items() if len(c) > 1}
print("  tools requesting MORE THAN ONE host across hostile args:")
for t, c in sorted(multi.items()):
    print(f"    {t}: {c}")
json.dump({t: dict(c) for t, c in hosts.items()}, open(HERE / "hosts_by_tool.json", "w"), indent=1)
