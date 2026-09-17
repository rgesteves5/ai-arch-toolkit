"""Experiment 5: the strict helper against the known-bad probes."""
from __future__ import annotations

import copy
import time

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from exp3_probes import PROBES
from strict import check, strict_validator

t0 = time.perf_counter()
v = strict_validator(MessageCreateParamsNonStreaming)
print(f"build strict validator: {(time.perf_counter() - t0) * 1000:.0f} ms")
for name, payload in PROBES.items():
    errs = check(v, copy.deepcopy(payload))
    print(f"  {'FAIL' if errs else 'pass'}  {name:26s} ({len(errs)} errors)")
    for e in errs[:4]:
        print("          ", e[:260])
