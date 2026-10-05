"""The YAML, JSON and TOML of the manifests and resources the toolkit loads, kept bounded (G-34,
D59).

Agent manifests and the resource codecs (which prompt manifests and knowledge use) parse here, so
an imported file costs time and memory in proportion to its size:

* YAML aliases are limited, not forbidden. A document's aliases may add at most
  :data:`MAX_ALIAS_NODES` nodes and :data:`MAX_ALIAS_CHARS` characters, or as many as the document
  holds if that is more. Anchors and merge keys keep working; an alias bomb, of many values or of
  one long one, is refused before it is built, and so is an alias inside its own anchor.
* No document nests deeper than :data:`MAX_DEPTH` levels of mappings and lists, in any format,
  merge keys included, so no recursive walk of the toolkit nears Python's recursion limit. YAML is
  refused as it is composed; a JSON or TOML parser that runs out of stack is the same refusal.

A refusal is an :class:`UnsafeDataError`; each caller turns it into its own error. A syntax error
is the parser's own (``yaml.YAMLError``, ``json.JSONDecodeError``, ``tomllib.TOMLDecodeError``).
"""

from __future__ import annotations

import contextlib
import functools
import json
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from yaml.nodes import Node

__all__ = [
    "MAX_ALIAS_CHARS",
    "MAX_ALIAS_NODES",
    "MAX_DEPTH",
    "UnsafeDataError",
    "check_depth",
    "load_json",
    "load_toml",
    "load_yaml",
]

MAX_ALIAS_NODES: Final = 10_000
"""The nodes a YAML document's aliases may add, when the document holds fewer than this itself."""

MAX_ALIAS_CHARS: Final = 1_000_000
"""The characters a YAML document's aliases may add, when it holds fewer than this itself."""

MAX_DEPTH: Final = 100
"""The levels of mappings and lists a document may nest (and the links a chain of YAML merge keys
may have)."""

_MERGE: Final = "tag:yaml.org,2002:merge"
_TREE: Final = (dict, list)  # what the JSON and TOML parsers build


class UnsafeDataError(ValueError):
    """A document that would take more memory or stack than its size warrants."""


def _too_deep() -> UnsafeDataError:
    return UnsafeDataError(f"the document nests deeper than {MAX_DEPTH} levels")


def _parser_too_deep() -> UnsafeDataError:
    return UnsafeDataError(
        f"the document nests deeper than the parser can follow (the limit is {MAX_DEPTH} levels)"
    )


def load_yaml(text: str) -> Any:
    """Safe-load one YAML document, its aliases and its nesting checked before it is built.

    Needs PyYAML (the ``yaml`` extra); the caller says how to install it.
    """
    loader = _bounded_loader()(text)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _check_nodes(node)
        return loader.construct_document(node)
    finally:
        loader.dispose()


def load_json(text: str) -> Any:
    """Parse JSON, its nesting checked."""
    try:
        data = json.loads(text)
    except RecursionError:
        raise _parser_too_deep() from None
    _check_tree(data)
    return data


def load_toml(text: str) -> dict[str, Any]:
    """Parse TOML, its nesting checked."""
    try:
        data = tomllib.loads(text)
    except RecursionError:
        raise _parser_too_deep() from None
    _check_tree(data)
    return data


def _check_tree(data: Any) -> None:
    """:func:`check_depth` for what a JSON or TOML parser built: a tree of dicts and lists, with
    no value shared or inside itself, so it is walked a level at a time, with no bookkeeping."""
    level = [data] if type(data) in _TREE else []
    depth = 0
    while level:
        depth += 1
        if depth > MAX_DEPTH:
            raise _too_deep()
        below: list[Any] = []
        for value in level:
            children = value.values() if type(value) is dict else value
            below += [child for child in children if type(child) in _TREE]
        level = below


def check_depth(data: Any, *, above: int = 0) -> None:
    """Refuse ``data`` if its mappings and lists nest deeper than :data:`MAX_DEPTH` levels, with
    ``above`` levels already taken by where it goes (an override's dotted path).

    Walks without recursion, each container once however often it is shared. A container inside
    itself is not followed: refusing the cycle is the caller's.
    """
    if not _is_container(data):
        if above > MAX_DEPTH:
            raise _too_deep()
        return
    height: dict[int, int] = {}  # id(container) -> levels in it
    on_path: set[int] = set()
    stack: list[tuple[Any, bool]] = [(data, False)]
    while stack:
        value, finished = stack.pop()
        key = id(value)
        if finished:
            on_path.discard(key)
            inner = (height.get(id(kid), 0) for kid in _containers_in(value))
            height[key] = 1 + max(inner, default=0)
            if above + height[key] > MAX_DEPTH:
                raise _too_deep()
            continue
        if key in height or key in on_path:
            continue
        on_path.add(key)
        stack.append((value, True))
        stack.extend((kid, False) for kid in _containers_in(value) if id(kid) not in height)


def _is_container(value: Any) -> bool:
    return isinstance(value, Mapping | list | tuple)


def _containers_in(value: Any) -> Iterator[Any]:
    children = value.values() if isinstance(value, Mapping) else value
    return (child for child in children if _is_container(child))


@functools.cache
def _bounded_loader() -> type[Any]:
    """PyYAML's ``SafeLoader``, refusing a mapping or a list deeper than :data:`MAX_DEPTH` as it
    composes it: its composer recurses once per level, and would go on to Python's limit."""
    import yaml

    class BoundedLoader(yaml.SafeLoader):
        levels = 0

        # ``anchor`` is the anchor's name or ``None`` (types-PyYAML calls it a dict).
        def compose_sequence_node(self, anchor: Any) -> Any:
            with self._level():
                return super().compose_sequence_node(anchor)

        def compose_mapping_node(self, anchor: Any) -> Any:
            with self._level():
                return super().compose_mapping_node(anchor)

        @contextlib.contextmanager
        def _level(self) -> Iterator[None]:
            self.levels += 1
            try:
                if self.levels > MAX_DEPTH:
                    raise _too_deep()
                yield
            finally:
                self.levels -= 1

    return BoundedLoader


@dataclass(frozen=True, slots=True)
class _Built:
    """What a composed node becomes once its aliases are expanded and its merge keys flattened."""

    nodes: int
    chars: int  # the characters of its scalars
    depth: int  # levels of mappings and lists
    merges: int  # the longest chain of merge keys from it (PyYAML recurses once per link)


def _check_nodes(root: Node) -> None:
    """Walk the composed node graph once, without recursion (an alias is the node it names, so the
    graph shares nodes), building each node's :class:`_Built` from its children's."""
    from yaml.nodes import ScalarNode

    built: dict[int, _Built] = {}
    on_path: set[int] = set()
    own_nodes = own_chars = 0
    stack: list[tuple[Node, bool]] = [(root, False)]
    while stack:
        node, finished = stack.pop()
        key = id(node)
        if finished:
            on_path.discard(key)
            result = _build(node, built)
            if result.depth > MAX_DEPTH:
                raise _too_deep()
            if result.merges > MAX_DEPTH:
                raise UnsafeDataError(
                    f"the document's merge keys (<<) chain more than {MAX_DEPTH} mappings"
                )
            built[key] = result
            own_nodes += 1
            own_chars += len(node.value) if isinstance(node, ScalarNode) else 0
            continue
        if key in built:
            continue
        if key in on_path:
            raise UnsafeDataError("a YAML alias refers to itself: its anchor contains it")
        on_path.add(key)
        stack.append((node, True))
        stack.extend((kid, False) for kid in _children(node) if id(kid) not in built)
    whole = built[id(root)]
    _within_budget("nodes", whole.nodes - own_nodes, own_nodes, MAX_ALIAS_NODES)
    _within_budget("characters", whole.chars - own_chars, own_chars, MAX_ALIAS_CHARS)


def _within_budget(what: str, added: int, own: int, floor: int) -> None:
    limit = max(floor, own)
    if added > limit:
        raise UnsafeDataError(
            f"the document's aliases expand its {own} {what} by {added}, more than {limit}"
        )


def _build(node: Node, built: dict[int, _Built]) -> _Built:
    """``node`` built from its children's results, which ``built`` already holds."""
    from yaml.nodes import MappingNode, ScalarNode, SequenceNode

    if isinstance(node, ScalarNode):
        return _Built(nodes=1, chars=len(node.value), depth=0, merges=0)
    if isinstance(node, SequenceNode):
        items = [built[id(item)] for item in node.value]
        return _Built(
            nodes=1 + sum(item.nodes for item in items),
            chars=sum(item.chars for item in items),
            depth=1 + max((item.depth for item in items), default=0),
            merges=0,
        )
    nodes, chars, inner, merges = 1, 0, 0, 0
    for key, value in node.value:
        if key.tag == _MERGE:
            # The merged mappings' pairs land beside this mapping's own (``flatten_mapping``).
            for merged in _merged(value):
                result = built[id(merged)]
                itself = 1 if isinstance(merged, MappingNode) else 0
                nodes += result.nodes - itself
                chars += result.chars
                inner = max(inner, result.depth - itself)
                merges = max(merges, 1 + result.merges)
            continue
        for part in (built[id(key)], built[id(value)]):
            nodes += part.nodes
            chars += part.chars
            inner = max(inner, part.depth)
    return _Built(nodes=nodes, chars=chars, depth=1 + inner, merges=merges)


def _merged(value: Node) -> list[Node]:
    """The nodes a merge key merges: one mapping, or a list of them (anything else, PyYAML
    refuses when it builds the data)."""
    from yaml.nodes import SequenceNode

    return list(value.value) if isinstance(value, SequenceNode) else [value]


def _children(node: Node) -> Iterator[Node]:
    from yaml.nodes import MappingNode, SequenceNode

    if isinstance(node, SequenceNode):
        yield from node.value
    elif isinstance(node, MappingNode):
        for key, value in node.value:
            yield key
            yield value
