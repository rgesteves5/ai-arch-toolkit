"""Experiment 13: Meta — neutralise "SDK-required but server-optional" fields with an explicit allowlist,
and check which SDK-conformant shapes the adapter could emit instead."""
from __future__ import annotations

import asyncio
import re

from capture import capture
from contract import validate_call

# Each entry documents one deliberate deviation from the openai SDK types (Meta != OpenAI).
TOLERATED = [
    r"^tools\.\d+\.function\.strict \| Field required$",
    r"\.input_image\.detail \| Field required$",
]


def tolerated(line: str) -> bool:
    return any(re.search(p, line) for p in TOLERATED)


async def main() -> None:
    remaining = 0
    for cap in await capture("meta"):
        if cap.kwargs is None:
            continue
        errs = validate_call("MetaProvider", cap.method, cap.kwargs)
        left = [e for e in errs if not tolerated(e)]
        # a union with no matching member reports every member: drop sibling noise once one
        # member's only problems were tolerated ones
        if left and any(tolerated(e) for e in errs):
            members = {e.split(".content.")[0] for e in errs if tolerated(e)}
            if any(all(tolerated(e) for e in errs if e.startswith(m)) for m in members):
                left = [e for e in left if not any(e.startswith(m.rsplit(".", 1)[0]) for m in members)]
        if left:
            remaining += 1
            print(f"  FAIL {cap.scenario}")
            for e in left[:6]:
                print("        ", e[:210])
    print("scenarios still failing after allowlist:", remaining)

    base = {"model": "muse-spark-1", "store": False}
    shapes = {
        "adapter today: content=[output_text], phase": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}], "phase": "commentary"},
        "easy message: content=str, phase": {"type": "message", "role": "assistant", "content": "hi", "phase": "commentary"},
        "easy message: content=str, no type": {"role": "assistant", "content": "hi"},
        "output message: +id/status/annotations": {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed", "content": [{"type": "output_text", "text": "hi", "annotations": []}]},
    }
    print("SDK-conformant alternatives for a rebuilt assistant turn:")
    for label, item in shapes.items():
        errs = validate_call("MetaProvider", "responses.create", {**base, "input": [item]})
        print(f"  {'conforms' if not errs else f'{len(errs)} errors':10s} {label}")
    # The allowlist must not blunt real detection:
    bad = {**base, "input": "hi", "tools": [{"type": "function", "name": "f", "parameters": {}, "strict": None}, {"type": "web_search", "max_uses": 3}]}
    print("still caught:", [e[:120] for e in validate_call("MetaProvider", "responses.create", bad) if not tolerated(e)])


asyncio.run(main())
