"""Each adapter's prepared request, checked against its SDK's own request contract (R02 step 4).

The static types hold the adapters' builders; this net catches what they cannot see: values of
type ``Any`` forwarded as given, casts, unions pyright accepts too widely, and the checks the SDKs
only make at run time. It runs on every request the suite prepares (the autouse ``wire_log``
fixture in ``tests/conftest.py``); ``tests/test_wire_contract.py`` holds its canaries.

- ``openai`` and ``anthropic`` (Stainless): the request ``TypedDict`` compiled into a strict
  validator. Extra keys are forbidden, scalars are not coerced (the SDK sends them as they are),
  iterables are validated at once, an SDK model is accepted only as an instance, and a union with
  a literal ``type``/``role`` is validated against the one member that tag names, for one precise
  error instead of one per member.
- ``google-genai``: the request through pydantic again (what was assigned after construction is
  not validated) and through the SDK's offline conversion for the Developer API, which refuses the
  fields only Vertex takes; the SDK's lenient enums warn, and a warning counts as an error.
- ``xai-sdk``: the SDK's ``Chat``, whose proto takes and serializes an enum value it does not
  define, so the net walks the proto for those.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from google.protobuf.message import Message as ProtoMessage
from pydantic import TypeAdapter
from pydantic_core import SchemaValidator, ValidationError

from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._providers._xai import XAIProvider

# The adapters whose ``prepare`` the ``wire_log`` fixture checks.
ADAPTERS = (OpenAIProvider, XAIProvider, GeminiProvider, MetaProvider, AnthropicProvider)

_UNTAGGED = "<untagged dict>"
_DISCRIMINATOR_KEYS = ("type", "role")
# Union members that the Python type of the input alone tells apart.
_KIND_TAGS = {"str": "<str>", "list": "<list>", "none": "<none>"}
_SCALARS = frozenset({"bool", "int", "float", "str"})


# ---------------------------------------------------------------------------
# The strict validator of a Stainless request TypedDict
# ---------------------------------------------------------------------------


def _resolve(schema: dict[str, Any], defs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    while schema.get("type") == "definition-ref":
        schema = defs[schema["schema_ref"]]
    return schema


def _literal_tags(node: dict[str, Any], key: str) -> list[str] | None:
    """The tags a typed dict answers to under ``key``: its literal values, and untagged when the
    key is optional; ``None`` when ``key`` is not a literal field of it."""
    found = node["fields"].get(key)
    if found is None:
        return None
    inner, optional = found["schema"], not found.get("required", True)
    while inner.get("type") in ("nullable", "default"):
        optional = True
        inner = inner["schema"]
    if inner.get("type") != "literal":
        return None
    return [*map(str, inner["expected"]), *([_UNTAGGED] if optional else [])]


def _tag_of(key: str) -> Callable[[Any], str]:
    def tag(value: Any) -> str:
        if isinstance(value, Mapping):
            found = value.get(key)
            return _UNTAGGED if found is None else str(found)
        if isinstance(value, str):
            return "<str>"
        if isinstance(value, list | tuple):
            return "<list>"
        return "<none>" if value is None else "<object>"

    return tag


def _classes(schema: dict[str, Any], defs: dict[str, dict[str, Any]]) -> tuple[type, ...] | None:
    """The classes of an ``is-instance`` member, or of a union made only of such members."""
    node = _resolve(schema, defs)
    kind = node.get("type")
    if kind == "is-instance":
        return node["cls"] if isinstance(node["cls"], tuple) else (node["cls"],)
    if kind not in ("union", "tagged-union"):
        return None
    found: list[type] = []
    members = node["choices"].values() if kind == "tagged-union" else node["choices"]
    for member in members:
        inner = _classes(member[0] if isinstance(member, tuple) else member, defs)
        if inner is None:
            return None
        found += inner
    return tuple(found)


def _tagged(node: dict[str, Any], defs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A union as a tagged union on its members' literal ``type`` or ``role``, when every member
    can be told apart by it or by its Python kind; otherwise the union as it is."""
    choices = [c[0] if isinstance(c, tuple) else c for c in node["choices"]]
    dicts = [c for c in choices if _resolve(c, defs).get("type") == "typed-dict"]
    classes: list[type] = []
    by_tag: dict[str, list[Any]] = {}
    for choice in choices:
        kind = _resolve(choice, defs).get("type")
        if choice in dicts:
            continue
        if (found := _classes(choice, defs)) is not None:
            classes += found
        elif kind in _KIND_TAGS:
            by_tag.setdefault(_KIND_TAGS[kind], []).append(choice)
        else:
            return node  # a member nothing tells apart (dict[str, object], Any, ...)
    key = next(
        (
            k
            for k in _DISCRIMINATOR_KEYS
            if all(_literal_tags(_resolve(c, defs), k) for c in dicts)
        ),
        None,
    )
    if dicts and key is None:
        return node
    for choice in dicts:
        for tag in _literal_tags(_resolve(choice, defs), key or "type") or ():
            by_tag.setdefault(tag, []).append(choice)
    members: dict[str, Any] = {
        tag: found[0] if len(found) == 1 else {"type": "union", "choices": found}
        for tag, found in by_tag.items()
    }
    if classes:  # every accepted SDK class in one isinstance check
        members["<object>"] = {"type": "is-instance", "cls": tuple(dict.fromkeys(classes))}
    if len(members) < 2:
        return node
    return {"type": "tagged-union", "choices": members, "discriminator": _tag_of(key or "type")}


def _walk(node: Any, visit: Callable[[dict[str, Any]], dict[str, Any]]) -> Any:
    if isinstance(node, list):
        return [_walk(v, visit) for v in node]
    if isinstance(node, tuple):
        return tuple(_walk(v, visit) for v in node)
    if not isinstance(node, dict):
        return node
    return visit({k: _walk(v, visit) for k, v in node.items()})


def _strict(node: dict[str, Any]) -> dict[str, Any]:
    kind = node.get("type")  # a dict when ``node`` maps field names and one is named "type"
    if not isinstance(kind, str):
        return node
    if kind == "typed-dict" and node.get("extra_behavior", "ignore") == "ignore":
        node["extra_behavior"] = "forbid"
    elif kind == "generator":  # an Iterable field: validate its items now
        node["type"] = "list"
    elif kind in _SCALARS:  # sent as it is: "yes" is no boolean on the wire
        node["strict"] = True
    elif kind == "model":  # an SDK response model: only an instance of it
        ref = {"ref": node["ref"]} if "ref" in node else {}
        return {"type": "is-instance", "cls": node["cls"], **ref}
    return node


@functools.cache
def strict_validator(tp: Any) -> SchemaValidator:
    """A strict, eager validator of ``tp``, a request ``TypedDict``."""
    fixed = _walk(TypeAdapter(tp).core_schema, _strict)
    defs: dict[str, dict[str, Any]] = {}
    if fixed.get("type") == "definitions":
        defs.update({d["ref"]: d for d in fixed["definitions"]})
    return SchemaValidator(
        _walk(fixed, lambda n: _tagged(n, defs) if n.get("type") == "union" else n)
    )


def _check(tp: Any, payload: Mapping[str, Any], prefix: str = "") -> list[str]:
    try:
        strict_validator(tp).validate_python(dict(payload))
    except ValidationError as exc:
        return [f"{prefix}{'.'.join(map(str, e['loc']))} | {e['msg']}" for e in exc.errors()]
    return []


# ---------------------------------------------------------------------------
# Per SDK
# ---------------------------------------------------------------------------


def _openai(params: Mapping[str, Any]) -> list[str]:
    from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

    return _check(CompletionCreateParamsNonStreaming, params)


class _Sampling(TypedDict, total=False):
    """The sampling fields of the Messages API (https://platform.claude.com/docs/en/api/messages),
    sent in the body since ``anthropic`` 1.x took them out of its params."""

    temperature: float
    top_p: float
    top_k: int


def _anthropic(params: Mapping[str, Any]) -> list[str]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    payload = dict(params)
    body = payload.pop("extra_body", {})  # a request option of the SDK, not a params field
    return _check(_Sampling, body, "extra_body.") + _check(
        MessageCreateParamsNonStreaming, payload
    )


def _is(value: Any, kind: str) -> bool:
    return isinstance(value, dict) and value.get("type") == kind


def _meta_item(item: Any) -> Any:
    if not isinstance(item, dict) or not isinstance(item.get("content"), list):
        return item
    if item.get("role") == "assistant" and item.get("type") == "message":  # deviation 1
        content = [
            {"annotations": [], **p} if _is(p, "output_text") else p for p in item["content"]
        ]
        return {"id": "msg_wire", "status": "completed", **item, "content": content}
    content = [{"detail": "auto", **p} if _is(p, "input_image") else p for p in item["content"]]
    return {**item, "content": content}  # deviation 3


def _meta(params: Mapping[str, Any]) -> list[str]:
    """The Responses request with the three deviations the Meta adapter lists (at the top of
    ``_meta.py``, each with its live proof) filled in, so the rest is checked strictly. Only a
    missing field is filled: a wrong value in it is still caught."""
    from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming

    payload = dict(params)
    if isinstance(payload.get("input"), list):
        payload["input"] = [_meta_item(item) for item in payload["input"]]
    if isinstance(payload.get("tools"), list):  # deviation 2
        payload["tools"] = [
            {"strict": None, **tool} if _is(tool, "function") else tool
            for tool in payload["tools"]
        ]
    return _check(ResponseCreateParamsNonStreaming, payload)


@functools.cache
def _developer_api() -> Any:
    from google import genai

    return genai.Client(api_key="offline")._api_client


def _gemini(params: Mapping[str, Any]) -> list[str]:
    from google.genai import models, types

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*is not a valid")  # the SDK's lenient enums
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        try:
            request = types._GenerateContentParameters(**params)
            request = types._GenerateContentParameters.model_validate(request.model_dump())
            models._GenerateContentParameters_to_mldev(_developer_api(), request)
        except (ValueError, TypeError, UserWarning) as exc:  # ValidationError is a ValueError
            return [f"{type(exc).__name__} | {' '.join(str(exc).split())[:300]}"]
    return []


def _undefined_enums(message: ProtoMessage, path: str = "") -> list[str]:
    found: list[str] = []
    for descriptor, value in message.ListFields():
        where = f"{path}{descriptor.name}"
        values = list(value) if descriptor.is_repeated else [value]
        if descriptor.enum_type is not None:
            defined = descriptor.enum_type.values_by_number
            found += [
                f"{where} | {v} is not a {descriptor.enum_type.name}"
                for v in values
                if v not in defined
            ]
        elif descriptor.message_type is not None:
            items = value.values() if descriptor.message_type.GetOptions().map_entry else values
            for i, item in enumerate(items):
                if isinstance(item, ProtoMessage):
                    at = f"{where}.{i}." if descriptor.is_repeated else f"{where}."
                    found += _undefined_enums(item, at)
    return found


def _xai(params: Any) -> list[str]:
    from xai_sdk.chat import BaseChat

    if not isinstance(params, BaseChat):
        return [f"request | {type(params).__name__} is not an xai_sdk Chat"]
    return _undefined_enums(params.proto)


VALIDATORS: dict[str, Callable[[Any], list[str]]] = {
    "OpenAIProvider": _openai,
    "XAIProvider": _xai,
    "GeminiProvider": _gemini,
    "MetaProvider": _meta,
    "AnthropicProvider": _anthropic,
}


def violations(adapter: str, params: Any) -> list[str]:
    """Where ``params``, what ``adapter``'s ``prepare`` built, leaves its SDK's contract, as
    ``"<where> | <what>"`` lines; empty when it conforms."""
    return VALIDATORS[adapter](params)


@dataclass(frozen=True, slots=True)
class WireLog:
    """What one test prepared: each adapter that built a request, and each violation as
    ``"<adapter>: <where> | <what>"``."""

    checked: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def record(self, adapter: str, params: Any) -> None:
        self.checked.append(adapter)
        self.violations.extend(f"{adapter}: {line}" for line in violations(adapter, params))

    def unexpected(self, tolerate: Iterable[str]) -> list[str]:
        """The violations no pattern in ``tolerate`` (regexes a test declares) matches."""
        patterns = [re.compile(p) for p in tolerate]
        return [line for line in self.violations if not any(p.search(line) for p in patterns)]
