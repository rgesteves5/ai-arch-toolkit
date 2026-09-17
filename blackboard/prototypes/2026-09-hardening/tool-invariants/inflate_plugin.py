"""Scratch pytest plugin: reuse the repo's own tool tests as fixtures for an output-bound probe.

INFLATE_MODE=none|str|list. In 'str' mode every JSON string leaf a tool parses is padded to
K_STR chars; in 'list' mode every non-empty JSON list is repeated up to N_LIST items. Each tool
call's output length is appended to $INFLATE_OUT. Test assertions are irrelevant here.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
import pkgutil
import types
from typing import Any

MODE = os.environ.get("INFLATE_MODE", "none")
OUT = os.environ["INFLATE_OUT"]
K_STR = 5_000
N_LIST = 200
_REAL_LOADS = json.loads


def _inflate(x: Any, depth: int = 0) -> Any:
    if isinstance(x, str):
        if MODE == "str" and len(x) < K_STR:
            return x + " " + "X" * (K_STR - len(x))
        return x
    if isinstance(x, list):
        items = [_inflate(i, depth + 1) for i in x]
        if MODE == "list" and items and depth <= 3:
            return (items * (N_LIST // len(items) + 1))[:N_LIST]
        return items
    if isinstance(x, dict):
        return {k: _inflate(v, depth + 1) for k, v in x.items()}
    return x


class _JsonShim(types.SimpleNamespace):
    def __getattr__(self, name: str) -> Any:
        return getattr(json, name)


def _loads(*a: Any, **k: Any) -> Any:
    return _inflate(_REAL_LOADS(*a, **k))


def _wrap(fn: Any, name: str) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        value = fn(*args, **kwargs)
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"tool": name, "mode": MODE, "len": len(value) if isinstance(value, str) else -1}) + "\n")
        return value

    return wrapper


def pytest_configure(config: Any) -> None:
    pkg = importlib.import_module("ai_arch_toolkit.toolkit.tools")
    names = set(pkg.__all__) | set(importlib.import_module(pkg.__name__ + ".dangerous").__all__)
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
        if MODE != "none" and hasattr(mod, "json"):
            mod.json = _JsonShim(loads=_loads)
        for name in names:
            fn = mod.__dict__.get(name)
            if callable(fn) and getattr(fn, "__module__", None) == mod.__name__:
                setattr(mod, name, _wrap(fn, name))
