from __future__ import annotations

import importlib
import importlib.metadata as md

for dist in ("anthropic", "openai", "google-genai", "xai-sdk", "httpx", "httpx2", "httpcore", "grpcio", "aiohttp", "pydantic", "h2", "anyio"):
    try:
        print(f"{dist:14s} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist:14s} NOT INSTALLED")

for mod in ("httpx", "httpx2", "aiohttp"):
    try:
        m = importlib.import_module(mod)
        print(mod, "importable ->", m.__file__)
    except ImportError as exc:
        print(mod, "NOT importable:", exc)
