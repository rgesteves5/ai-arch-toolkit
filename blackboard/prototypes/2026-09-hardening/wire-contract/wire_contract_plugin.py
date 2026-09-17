"""pytest plugin (report-only): validate every call existing tests make on a mocked SDK client.

Loaded from outside the repo with ``-p wire_contract_plugin``. It records each client assigned via
``provider._client = <mock>`` and, at test teardown, checks the recorded calls of the known SDK
entry points. Nothing in the test is changed: validation is post-hoc over ``call_args_list``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any
from unittest.mock import Mock, NonCallableMock

import pytest

from contract import ENTRY_POINTS, validate_call

_P = "ai_arch_toolkit.core._providers."
_BUILDERS = (
    (_P + "_anthropic", "AnthropicProvider", "_build_sdk_kwargs", "messages.create"),
    (_P + "_openai", "OpenAIProvider", "_build_sdk_kwargs", "chat.completions.create"),
    (_P + "_meta", "MetaProvider", "_build_request", "responses.create"),
    (_P + "_gemini", "GeminiProvider", "_build_config", "aio.models.generate_content"),
    (_P + "_xai", "XAIProvider", "_build_create_kwargs", "chat.create"),
)
_REPORT: list[dict[str, Any]] = []
_CALLS: Counter[str] = Counter()


def _resolve(client: Any, dotted: str) -> Any:
    node = client
    for part in dotted.split("."):
        # only follow children the test/adapter already touched; never create new mock attributes
        children = getattr(node, "_mock_children", {})
        if part not in children:
            return None
        node = children[part]
    return node


@pytest.fixture(autouse=True)
def _wire_contract(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    from ai_arch_toolkit.core._providers._base import LoopAwareClientCache

    seen: list[tuple[str, Any]] = []
    first = len(_REPORT)
    prop = LoopAwareClientCache.__dict__["_client"]

    def _set(self: Any, value: Any) -> None:
        prop.fset(self, value)
        seen.append((type(self).__name__, value))

    monkeypatch.setattr(LoopAwareClientCache, "_client", property(prop.fget, _set))

    # Tests that call the private request builders directly never reach the mocked client:
    # wrap the builders too, so their return value is checked as the call it would become.
    built: list[tuple[str, str, dict[str, Any]]] = []
    for mod, cls_name, builder, method in _BUILDERS:
        try:
            cls = getattr(__import__(mod, fromlist=[cls_name]), cls_name)
        except ImportError:
            continue
        original = getattr(cls, builder)

        def _wrapped(self: Any, *a: Any, __o: Any = original, __c: str = cls_name, __m: str = method, **k: Any) -> Any:
            result = __o(self, *a, **k)
            if __c == "GeminiProvider":
                built.append((__c, __m, {"model": "m", "contents": "x", "config": result}))
            else:
                built.append((__c, __m, dict(result)))
            return result

        monkeypatch.setattr(cls, builder, _wrapped)
    yield
    for provider, method, kwargs in built:
        _CALLS[f"builder:{provider}"] += 1
        errors = validate_call(provider, method, kwargs)
        if errors:
            _REPORT.append({"test": request.node.nodeid, "call": f"builder:{provider}", "errors": errors})
    if os.environ.get("WIRE_CONTRACT_ENFORCE") == "1":
        import re

        marker = request.node.get_closest_marker("wire_contract")
        tolerate = marker.kwargs.get("tolerate", []) if marker else []
        lines = sorted(
            {
                f"{v['call'].removeprefix('builder:')}: {e}"
                for v in _REPORT[first:]
                for e in v["errors"]
                if not any(re.search(p, e) for p in tolerate)
            }
        )
        if lines:
            pytest.fail("SDK wire-contract violations:\n  " + "\n  ".join(lines), pytrace=False)
    for provider, client in seen:
        if not isinstance(client, NonCallableMock) or provider not in ENTRY_POINTS:
            continue
        for method in ENTRY_POINTS[provider]:
            mock = _resolve(client, method)
            if not isinstance(mock, Mock):
                continue
            for call in mock.call_args_list:
                _CALLS[f"{provider}.{method}"] += 1
                if call.args:
                    errors = ["positional args passed to SDK entry point"]
                else:
                    errors = validate_call(provider, method, dict(call.kwargs))
                if errors:
                    _REPORT.append({"test": request.node.nodeid, "call": f"{provider}.{method}", "errors": errors})


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "wire_contract(tolerate): deliberate SDK-contract deviations")


def pytest_sessionfinish(session: pytest.Session) -> None:
    out = os.environ.get("WIRE_CONTRACT_REPORT")
    if out:
        with open(out, "w") as fh:
            json.dump({"calls": dict(_CALLS), "violations": _REPORT}, fh, indent=1)
