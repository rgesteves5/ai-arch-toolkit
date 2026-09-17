"""Contract check for the Gemini adapter: the SDK's own (offline) request pipeline.

1. bind call kwargs to the real ``AsyncModels.generate_content`` signature
2. ``types._GenerateContentParameters(**kwargs)`` — the model the SDK itself builds first (forbid)
3. ``_GenerateContentParameters_to_mldev`` — the SDK's pure dict transformer to Gemini-API JSON;
   raises ``ValueError`` for fields the Developer API does not support
4. SDK "is not a valid <Enum>" warnings escalated to errors
"""
from __future__ import annotations

import asyncio
import inspect
import statistics
import time
import warnings
from typing import Any

from google import genai
from google.genai import models as genai_models
from google.genai import types

from capture import capture

_API_CLIENT = genai.Client(api_key="offline")._api_client  # no I/O at construction
_SIG = {
    "aio.models.generate_content": inspect.signature(genai_models.AsyncModels.generate_content),
    "aio.models.generate_content_stream": inspect.signature(genai_models.AsyncModels.generate_content_stream),
}


def check_gemini(method: str, kwargs: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    wire = None
    try:
        _SIG[method].bind(None, **kwargs)
    except TypeError as exc:
        return [f"signature | {exc}"], None
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*is not a valid.*")
        try:
            params = types._GenerateContentParameters(**kwargs)
            # Round-trip: catches post-construction mutation and model_construct() shortcuts.
            types._GenerateContentParameters.model_validate(params.model_dump())
            wire = genai_models._GenerateContentParameters_to_mldev(_API_CLIENT, params)
        except Exception as exc:  # pydantic ValidationError / ValueError / UserWarning
            errors.append(f"{type(exc).__name__} | {str(exc)[:300]}")
    return errors, wire


async def main() -> None:
    timings = []
    for model in ("gemini-2.5-flash", "gemini-3-pro-preview"):
        print(f"##### model={model}")
        for cap in await capture("gemini", model=model):
            if cap.kwargs is None:
                print(f"  skip  {cap.scenario:24s} adapter raised: {cap.error}")
                continue
            t0 = time.perf_counter()
            errs, wire = check_gemini(cap.method, cap.kwargs)
            timings.append((time.perf_counter() - t0) * 1000)
            print(f"  {'FAIL' if errs else 'pass'}  {cap.scenario:24s} wire keys={sorted(wire) if wire else None}")
            for e in errs:
                print("           ", e)
            for w in cap.warnings:
                print("            adapter-time warning:", w[:160])
    print(f"per-validation: median {statistics.median(timings):.2f} ms, max {max(timings):.2f} ms")

    # Known-bad probes: what does this layer catch?
    print("##### probes")
    good = {"model": "gemini-2.5-flash", "contents": [types.Content(role="user", parts=[types.Part(text="hi")])]}
    probes = {
        "config as dict with unknown key": {**good, "config": {"max_tokens": 5}},
        "config as dict, valid": {**good, "config": {"max_output_tokens": 5}},
        "unknown call kwarg": {**good, "messages": []},
        "contents as plain dicts (valid)": {"model": "m", "contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        "contents dict with bad part key": {"model": "m", "contents": [{"role": "user", "parts": [{"txt": "hi"}]}]},
        "mutated after construction": None,
        "vertex-only field on Developer API": {**good, "config": types.GenerateContentConfig(labels={"a": "b"})},
        "role='tool' (free-form str)": {"model": "m", "contents": [types.Content(role="tool", parts=[types.Part(text="x")])]},
    }
    cfg = types.GenerateContentConfig(temperature=0.0)
    cfg.temperature = "hot"  # type: ignore[assignment]  # validate_assignment is off
    probes["mutated after construction"] = {**good, "config": cfg}
    for name, kw in probes.items():
        errs, _ = check_gemini("aio.models.generate_content", kw)
        print(f"  {'FAIL' if errs else 'pass'}  {name}")
        for e in errs:
            print("           ", e[:230].replace("\n", " "))


asyncio.run(main())
