"""Experiment 12: differential check — strict rewrite vs plain pydantic (+drain) on every captured call.

A call that the *lenient* SDK-typed validation accepts but the strict one rejects must be explained
by extra keys only; anything else would be a bug in the core-schema rewrite.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import TypeAdapter, ValidationError

from capture import capture
from strict import check
from validate_typed import targets  # noqa: E402


def drain(value: Any) -> None:
    if isinstance(value, dict):
        for v in value.values():
            drain(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            drain(v)
    elif type(value).__name__ == "ValidatorIterator":
        drain(list(value))


async def main() -> None:
    from strict import strict_validator

    for adapter, models in (("anthropic", [None, "claude-fable-5-1"]), ("openai", [None, "gpt-5"]), ("meta", [None])):
        types = targets(adapter)
        lenient = {tp: TypeAdapter(tp) for tp in set(types.values())}
        strict = {tp: strict_validator(tp) for tp in set(types.values())}
        agree = disagree = 0
        for model in models:
            for cap in await capture(adapter, model=model):
                if cap.kwargs is None:
                    continue
                tp = types["_streaming"] if cap.kwargs.get("stream") else types[cap.method]
                try:
                    drain(lenient[tp].validate_python(cap.kwargs))
                    lenient_ok = True
                except ValidationError:
                    lenient_ok = False
                errs = check(strict[tp], cap.kwargs)
                if lenient_ok == (not errs):
                    agree += 1
                else:
                    disagree += 1
                    only_extra = all("Extra inputs" in e for e in errs)
                    print(f"  {adapter} {cap.scenario}: lenient_ok={lenient_ok} strict_errors={len(errs)} only-extra-keys={only_extra}")
                    for e in errs[:3]:
                        print("       ", e[:200])
        print(f"{adapter}: agree={agree} disagree={disagree}")


asyncio.run(main())
