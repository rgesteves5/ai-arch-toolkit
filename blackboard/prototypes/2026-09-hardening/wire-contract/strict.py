"""Strict, offline validation of SDK request kwargs against the SDK's own TypedDict params.

``strict_validator(tp)`` compiles the SDK type with pydantic, then rewrites the *core schema*:

* ``typed-dict`` nodes: ``extra_behavior`` ``ignore`` -> ``forbid`` (an explicit ``allow`` is kept)
* ``generator`` nodes (``Iterable[...]``) -> ``list``: eager validation with full error paths
* ``model`` nodes (SDK response models allowed as input) -> ``is-instance``: a plain dict must
  match a *Param* TypedDict, it cannot sneak through a lenient ``extra="allow"`` response model
* ``union`` nodes -> ``tagged-union`` keyed on the members' literal ``type``/``role`` field and on
  the python kind of the input: one precise error instead of one error per union member
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import SchemaValidator, ValidationError

__all__ = ["check", "strict_validator"]

_UNTAGGED = "<untagged dict>"
_DISCRIMINATOR_KEYS = ("type", "role")
# Non-dict union members that can be told apart by the python type of the input alone.
_KIND_TAGS = {"str": "<str>", "list": "<list>", "is-instance": "<object>", "none": "<none>"}


def _resolve(schema: dict[str, Any], defs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    while schema.get("type") == "definition-ref":
        schema = defs[schema["schema_ref"]]
    return schema


def _literal_tags(node: dict[str, Any], key: str) -> list[str] | None:
    """Tags a typed-dict answers to under ``key``: its literal values (+ untagged if optional)."""
    field = node["fields"].get(key)
    if field is None:
        return None
    inner, optional = field["schema"], not field.get("required", True)
    while inner.get("type") in ("nullable", "default"):
        optional = True
        inner = inner["schema"]
    if inner.get("type") != "literal":
        return None
    return [*map(str, inner["expected"]), *([_UNTAGGED] if optional else [])]


def _make_discriminator(key: str) -> Callable[[Any], str]:
    def discriminator(value: Any) -> str:
        if isinstance(value, Mapping):
            tag = value.get(key)
            return _UNTAGGED if tag is None else str(tag)
        if isinstance(value, str):
            return "<str>"
        if isinstance(value, list | tuple):
            return "<list>"
        return "<none>" if value is None else "<object>"

    discriminator.__qualname__ = f"by[{key!r}]"
    return discriminator


def _classes(schema: dict[str, Any], defs: dict[str, dict[str, Any]]) -> tuple[type, ...] | None:
    """Classes of an ``is-instance`` member, or of a (tagged) union made only of such members."""
    node = _resolve(schema, defs)
    kind = node.get("type")
    if kind == "is-instance":
        cls = node["cls"]
        return cls if isinstance(cls, tuple) else (cls,)
    if kind in ("union", "tagged-union"):
        members = node["choices"].values() if kind == "tagged-union" else node["choices"]
        found: list[type] = []
        for member in members:
            inner = _classes(member[0] if isinstance(member, tuple) else member, defs)
            if inner is None:
                return None
            found += inner
        return tuple(found)
    return None


def _tag_union(node: dict[str, Any], defs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    choices = [c[0] if isinstance(c, tuple) else c for c in node["choices"]]
    dicts = [c for c in choices if _resolve(c, defs).get("type") == "typed-dict"]
    others = [c for c in choices if _resolve(c, defs).get("type") != "typed-dict"]
    classes: list[type] = []
    simple: list[Any] = []
    for choice in others:
        found = _classes(choice, defs)
        if found is not None:
            classes += found
        elif _resolve(choice, defs).get("type") in _KIND_TAGS:
            simple.append(choice)
        else:
            return node  # a member we cannot dispatch on (dict[str, object], Any, ...): keep as is
    key = next(
        (k for k in _DISCRIMINATOR_KEYS if all(_literal_tags(_resolve(c, defs), k) for c in dicts)),
        None,
    )
    if dicts and key is None:
        return node
    tagged: dict[str, list[Any]] = {}
    for choice in dicts:
        for tag in _literal_tags(_resolve(choice, defs), key or "type") or ():
            tagged.setdefault(tag, []).append(choice)
    for choice in simple:
        tagged.setdefault(_KIND_TAGS[_resolve(choice, defs)["type"]], []).append(choice)
    by_tag: dict[str, Any] = {
        tag: members[0] if len(members) == 1 else {"type": "union", "choices": members}
        for tag, members in tagged.items()
    }
    if classes:  # fold every accepted SDK class into one isinstance check
        by_tag["<object>"] = {
            "type": "is-instance",
            "cls": tuple(dict.fromkeys(classes)),
            "cls_repr": "an SDK object",
        }
    if len(by_tag) < 2:
        return node
    return {
        "type": "tagged-union",
        "choices": by_tag,
        "discriminator": _make_discriminator(key or "type"),
    }


def strict_validator(tp: Any) -> SchemaValidator:
    """Compile ``tp`` (an SDK params TypedDict) into a strict, eager validator."""
    defs: dict[str, dict[str, Any]] = {}

    def walk(node: Any, visit: Callable[[dict[str, Any]], dict[str, Any]]) -> Any:
        if isinstance(node, list):
            return [walk(v, visit) for v in node]
        if isinstance(node, tuple):
            return tuple(walk(v, visit) for v in node)
        if not isinstance(node, dict):
            return node
        return visit({k: walk(v, visit) for k, v in node.items()})

    def fix(node: dict[str, Any]) -> dict[str, Any]:
        kind = node.get("type")
        if kind == "typed-dict" and node.get("extra_behavior", "ignore") == "ignore":
            node["extra_behavior"] = "forbid"
        elif kind == "generator":
            node["type"] = "list"
        elif kind == "model":
            ref = {"ref": node["ref"]} if "ref" in node else {}
            return {"type": "is-instance", "cls": node["cls"], **ref}
        return node

    fixed = walk(TypeAdapter(tp).core_schema, fix)
    if fixed.get("type") == "definitions":
        defs.update({d["ref"]: d for d in fixed["definitions"]})
    tagged = walk(fixed, lambda n: _tag_union(n, defs) if n.get("type") == "union" else n)
    return SchemaValidator(tagged)


def check(validator: SchemaValidator, kwargs: dict[str, Any]) -> list[str]:
    """Validate; return short ``loc | message`` lines (empty list = conforms)."""
    try:
        validator.validate_python(kwargs)
    except ValidationError as exc:
        return [f"{'.'.join(map(str, e['loc']))} | {e['msg']}" for e in exc.errors()]
    return []
