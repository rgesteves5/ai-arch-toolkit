"""The technical model catalog: what a model takes through its adapter, each fact with its source.

Apart from ``pricing``, which knows rates only. A fact is ``None`` when unknown, never "no"
(D63). The shipped seed (``_default_catalog.toml``) holds what the providers publish, limits and
modalities, each with its page and the day it was read. What the adapter does with a model
(tools, tool choices, reasoning, server tools) comes from the adapter's own tables, through its
pure ``model_facts`` classmethod; no adapter reads the catalog. An app overrides any fact:
``load()`` over the seed and the adapter, ``register()`` over everything, field by field.

Ids match as everywhere in ``core`` (``_model_id.lookup``): an entry's id, an alias, or a dated
snapshot of either; never a family prefix, since a variant is another model. ``register``,
``load`` and ``unregister`` name an entry the same way.

A catalog is safe to share between threads: a change builds the next version of it, under a
lock, and a read takes the current one, so what a read builds belongs to the version it read.
"""

from __future__ import annotations

import dataclasses
import threading
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import date
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._providers import _match_provider
from ai_arch_toolkit.core._providers._base import (
    EFFORT_ORDER,
    SERVER_TOOL_TYPES,
    AdapterFacts,
    InputModality,
    OutputModality,
    ThinkingMode,
    ToolChoiceMode,
    ordered_efforts,
)

if TYPE_CHECKING:
    from ai_arch_toolkit.core._providers._base import BaseProvider

__all__ = ["ModelCapabilities", "ModelCatalog", "Provenance", "model_catalog"]

type SourceKind = Literal["docs", "probe", "api", "adapter", "override"]

CATALOG_VERSION = 1
_SEED = Path(__file__).with_name("_default_catalog.toml")


def _vocabulary(alias: Any) -> tuple[str, ...]:
    return get_args(alias.__value__)


_SOURCE_KINDS = _vocabulary(SourceKind)


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    """Where a fact comes from.

    Attributes:
        kind: ``"docs"`` (the provider's page), ``"probe"`` (a live run), ``"api"`` (the
            provider's models endpoint), ``"adapter"`` (the adapter's own tables) or
            ``"override"`` (the app's word).
        ref: The page, run or code it was read from.
        verified_at: The day it was read; ``None`` for an adapter's fact, as current as the
            installed code, or for an override given without a day.
    """

    kind: SourceKind
    ref: str
    verified_at: date | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise ValueError(f"kind must be one of {', '.join(_SOURCE_KINDS)}, got {self.kind!r}")
        if not isinstance(self.ref, str) or not self.ref:
            raise ValueError(f"ref must name the page, run or code, got {self.ref!r}")
        if self.verified_at is not None and type(self.verified_at) is not date:
            raise ValueError(
                f"verified_at must be a date (a datetime is not), got {self.verified_at!r}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilities:
    """What one model takes through its adapter, each fact with its source (D63).

    A fact is ``None`` when unknown, and ``None`` never means "no": a filter that needs a fact
    excludes a model that lacks it. A set lists what its sources state: a modality a provider's
    page does not name is not stated, while one the adapter does not send (``kind="adapter"``)
    does not reach the model. The three limits are kept as the provider publishes them: OpenAI
    gives a context window shared by input and output, Gemini an input and an output limit.

    Attributes:
        provider: The adapter, for the provider's own host (``"anthropic"``, ``"openai"``,
            ``"gemini"``, ``"xai"``, ``"meta"``), or a namespace of the app's (``"ollama"``).
        model: The id the provider gives the model.
        aliases: Other ids of the same model.
        context_window: Tokens of input and output together.
        input_token_limit: Tokens of input.
        output_token_limit: Tokens of output, reasoning included.
        input_modalities: What the model reads, through the adapter.
        output_modalities: What it writes: ``"text"``, ``"image"``.
        tools: It calls function tools.
        tool_choice_modes: The ``tool_choice`` forms it takes (``"named"``: a tool's name).
        parallel_tool_calls: It calls several tools in one turn.
        structured_output: It takes ``output_schema``.
        json_mode: It takes ``json_mode``.
        streaming: Its answers stream.
        thinking_mode: ``"none"``: it does not reason; ``"optional"``: it reasons when asked;
            ``"always"``: it reasons unasked (``"none"`` among its efforts stops it).
        thinking_efforts: The ``thinking_effort`` values it takes, weakest first.
        thinking_budget: It takes a ``thinking_budget`` in tokens.
        server_tools: The server tools it runs, by ``ServerTool.type``.
        sources: Each set fact's :class:`Provenance`, by field name.

    Raises:
        ValueError: A fact of the wrong type or outside its vocabulary, as
            :meth:`ModelCatalog.load` refuses it. A set may be given as any collection, and is
            kept as a ``frozenset`` (the efforts as a tuple, weakest first, each once).
    """

    provider: str
    model: str
    aliases: tuple[str, ...] = ()
    context_window: int | None = None
    input_token_limit: int | None = None
    output_token_limit: int | None = None
    input_modalities: frozenset[InputModality] | None = None
    output_modalities: frozenset[OutputModality] | None = None
    tools: bool | None = None
    tool_choice_modes: frozenset[ToolChoiceMode] | None = None
    parallel_tool_calls: bool | None = None
    structured_output: bool | None = None
    json_mode: bool | None = None
    streaming: bool | None = None
    thinking_mode: ThinkingMode | None = None
    thinking_efforts: tuple[str, ...] | None = None
    thinking_budget: bool | None = None
    server_tools: frozenset[str] | None = None
    sources: tuple[tuple[str, Provenance], ...] = ()

    def __post_init__(self) -> None:
        named = isinstance(self.provider, str) and isinstance(self.model, str)
        if not named or not self.provider or not self.model:
            raise ValueError(
                f"an entry needs a provider and a model, got {self.provider!r}, {self.model!r}"
            )
        where = f"{self.provider} {self.model!r}"
        try:
            aliases = _strings(self.aliases)
        except ValueError as refused:
            raise ValueError(f"{where}: aliases {refused}") from None
        if not isinstance(self.aliases, tuple):
            object.__setattr__(self, "aliases", tuple(aliases))
        for name in _FACTS:
            if (value := getattr(self, name)) is not None:
                try:
                    object.__setattr__(self, name, _PARSERS[name](value))
                except ValueError as refused:
                    raise ValueError(f"{where}: {name} {refused}") from None
        for name, source in self.sources:
            _check_fact(name)
            if not isinstance(source, Provenance):
                raise ValueError(f"{where}: sources.{name} is not a Provenance, got {source!r}")

    def provenance(self, name: str) -> Provenance | None:
        """Where the fact ``name`` comes from; ``None`` when it is not set."""
        _check_fact(name)
        return dict(self.sources).get(name)

    def to_dict(self) -> dict[str, Any]:
        """The entry as plain data, ready for JSON: sets as sorted lists, days as ISO dates."""
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "aliases": list(self.aliases),
        }
        for name in _FACTS:
            value = getattr(self, name)
            if isinstance(value, frozenset):
                value = sorted(value)
            elif isinstance(value, tuple):
                value = list(value)
            data[name] = value
        data["sources"] = {
            name: {
                "kind": source.kind,
                "ref": source.ref,
                "verified_at": source.verified_at.isoformat() if source.verified_at else None,
            }
            for name, source in self.sources
        }
        return data


_FACTS = tuple(
    f.name
    for f in fields(ModelCapabilities)
    if f.name not in {"provider", "model", "aliases", "sources"}
)
# The adapter's modalities narrow a published list (C06.1); its other facts fill in what no
# layer sets. The provider's word is its page or its models endpoint.
_NARROWED = ("input_modalities", "output_modalities")
_ADAPTER_FACTS = tuple(f.name for f in fields(AdapterFacts) if f.name not in _NARROWED)
_PUBLISHED: frozenset[str] = frozenset({"docs", "api"})
_OVERRIDE = Provenance(kind="override", ref="ModelCatalog.register")


def _check_fact(name: str) -> None:
    if name not in _FACTS:
        raise ValueError(f"{name!r} is not a fact of a model; the facts: {', '.join(_FACTS)}")


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def _sorted_sources(sources: dict[str, Provenance]) -> tuple[tuple[str, Provenance], ...]:
    return tuple(sorted(sources.items()))


def _sourced(entry: ModelCapabilities, default: Provenance) -> ModelCapabilities:
    """``entry`` with a source for each fact it sets: its own, else ``default``."""
    given = dict(entry.sources)
    sources = {
        name: given.get(name, default) for name in _FACTS if getattr(entry, name) is not None
    }
    return dataclasses.replace(entry, sources=_sorted_sources(sources))


def _merged(base: ModelCapabilities, over: ModelCapabilities) -> ModelCapabilities:
    """``base`` with each fact ``over`` sets, and its source; their aliases joined."""
    changes = {name: value for name in _FACTS if (value := getattr(over, name)) is not None}
    over_sources = dict(over.sources)
    sources = dict(base.sources) | {name: over_sources[name] for name in changes}
    return dataclasses.replace(
        base,
        aliases=tuple(dict.fromkeys((*base.aliases, *over.aliases))),
        sources=_sorted_sources(sources),
        **changes,
    )


def _with_adapter(
    entry: ModelCapabilities, adapter: type[BaseProvider[Any, Any]]
) -> ModelCapabilities:
    """``entry`` with the adapter's facts where no layer set one, and a published modality list
    narrowed to what the adapter carries (C06.1): an app's observation is never narrowed."""
    facts = adapter.model_facts(entry.model)
    source = Provenance(
        kind="adapter", ref=f"{adapter.__module__}.{adapter.__qualname__}.model_facts"
    )
    changes: dict[str, Any] = {
        name: value
        for name in _ADAPTER_FACTS
        if (value := getattr(facts, name)) is not None and getattr(entry, name) is None
    }
    for name in _NARROWED:
        if (narrowed := _narrowed(entry, name, getattr(facts, name))) is not None:
            changes[name] = narrowed
    sources = dict(entry.sources) | dict.fromkeys(changes, source)
    return dataclasses.replace(entry, sources=_sorted_sources(sources), **changes)


def _narrowed(
    entry: ModelCapabilities, name: str, carried: frozenset[str] | None
) -> frozenset[str] | None:
    """The published modalities ``name`` without those the adapter does not carry, when it
    drops some. A list of any other kind (a probe's, an override) is the app's word, left as
    it is."""
    published: frozenset[str] | None = getattr(entry, name)
    stated = entry.provenance(name)
    if published is None or carried is None or stated is None or stated.kind not in _PUBLISHED:
        return None
    return None if published <= carried else published & carried


def _adapter_of(provider: str) -> type[BaseProvider[Any, Any]] | None:
    """The adapter of a provider's own host; ``None`` for an app's namespace, or for an adapter
    that does not import: its SDK extra is not installed, or the installed SDK is broken (a
    protobuf built for another version raises ``TypeError``). Its facts are then unknown."""
    try:
        return _import_adapter(provider)
    except Exception:  # whatever an SDK raises on import, its facts are unknown
        return None


def _import_adapter(provider: str) -> type[BaseProvider[Any, Any]] | None:
    if provider == "anthropic":
        from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider

        return AnthropicProvider
    if provider == "openai":
        from ai_arch_toolkit.core._providers._openai import OpenAIProvider

        return OpenAIProvider
    if provider == "gemini":
        from ai_arch_toolkit.core._providers._gemini import GeminiProvider

        return GeminiProvider
    if provider == "xai":
        from ai_arch_toolkit.core._providers._xai import XAIProvider

        return XAIProvider
    if provider == "meta":
        from ai_arch_toolkit.core._providers._meta import MetaProvider

        return MetaProvider
    return None


# ---------------------------------------------------------------------------
# The strict loader
# ---------------------------------------------------------------------------

_ENTRY_KEYS = frozenset({"aliases", "source", "sources"})
_SOURCE_KEYS = frozenset({"kind", "ref", "verified_at"})


def _positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"must be a positive integer, got {value!r}")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"must be true or false, got {value!r}")
    return value


def _strings(value: object, vocabulary: tuple[str, ...] | None = None) -> list[str]:
    """A list of names (a TOML array, or any collection but a string), each in ``vocabulary``."""
    names = list(value) if isinstance(value, list | tuple | set | frozenset) else None
    if names is None or not all(isinstance(item, str) and item for item in names):
        raise ValueError(f"must be a list of names, got {value!r}")
    if vocabulary is not None and (unknown := sorted(set(names) - set(vocabulary))):
        raise ValueError(f"takes {', '.join(vocabulary)}; not {', '.join(unknown)}")
    return names


def _choice(vocabulary: tuple[str, ...]) -> Callable[[object], object]:
    def parse(value: object) -> object:
        if value not in vocabulary:
            raise ValueError(f"must be one of {', '.join(vocabulary)}, got {value!r}")
        return value

    return parse


def _set_of(vocabulary: tuple[str, ...]) -> Callable[[object], object]:
    return lambda value: frozenset(_strings(value, vocabulary))


_PARSERS: dict[str, Callable[[object], object]] = {
    "context_window": _positive_int,
    "input_token_limit": _positive_int,
    "output_token_limit": _positive_int,
    "input_modalities": _set_of(_vocabulary(InputModality)),
    "output_modalities": _set_of(_vocabulary(OutputModality)),
    "tools": _boolean,
    "tool_choice_modes": _set_of(_vocabulary(ToolChoiceMode)),
    "parallel_tool_calls": _boolean,
    "structured_output": _boolean,
    "json_mode": _boolean,
    "streaming": _boolean,
    "thinking_mode": _choice(_vocabulary(ThinkingMode)),
    "thinking_efforts": lambda value: ordered_efforts(_strings(value, EFFORT_ORDER)),
    "thinking_budget": _boolean,
    "server_tools": _set_of(SERVER_TOOL_TYPES),
}


def _provenance(raw: object, where: str) -> Provenance:
    """A source table: its kind, its ref, and the day it was read, a TOML date."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a source is a table (kind, ref, verified_at), got {raw!r}")
    if unknown := sorted(set(raw) - _SOURCE_KEYS):
        raise ValueError(
            f"{where}: unknown key {unknown[0]!r} (a source has kind, ref, verified_at)"
        )
    day = raw.get("verified_at")
    if type(day) is not date:  # a TOML datetime is a date too, and is refused
        raise ValueError(
            f"{where}: verified_at must be a TOML date such as 2026-10-08, got {day!r}"
        )
    kind, ref = raw.get("kind"), raw.get("ref")
    try:  # Provenance checks the kind and the ref
        return Provenance(kind=kind, ref=ref, verified_at=day)  # type: ignore[arg-type]
    except ValueError as refused:
        raise ValueError(f"{where}: {refused}") from None


def _sources(values: dict[str, Any], facts: Iterable[str], where: str) -> dict[str, Provenance]:
    """Each fact's source: its own under ``sources``, else the entry's ``source``."""
    default = values.get("source")
    entry_source = _provenance(default, f"{where} source") if default is not None else None
    own = values.get("sources", {})
    if not isinstance(own, dict):
        raise ValueError(f"{where}: sources is a table of facts, got {own!r}")
    facts = list(facts)
    if stray := sorted(set(own) - set(facts)):
        raise ValueError(f"{where}: sources.{stray[0]} names no fact this entry sets")
    parsed = {name: _provenance(raw, f"{where} sources.{name}") for name, raw in own.items()}
    sources: dict[str, Provenance] = {}
    for name in facts:
        found = parsed.get(name, entry_source)
        if found is None:
            raise ValueError(f"{where}: {name} has no source; give `source` or `sources.{name}`")
        sources[name] = found
    return sources


def _entry(provider: str, model: str, values: object, where: str) -> ModelCapabilities:
    if not isinstance(values, dict):
        raise ValueError(f"{where}: an entry is a table of facts, got {values!r}")
    if unknown := sorted(set(values) - set(_FACTS) - _ENTRY_KEYS):
        raise ValueError(f"{where}: unknown key {unknown[0]!r}; the facts: {', '.join(_FACTS)}")
    facts: dict[str, Any] = {}
    for name in _FACTS:
        if name in values:
            try:
                facts[name] = _PARSERS[name](values[name])
            except ValueError as refused:
                raise ValueError(f"{where}: {name} {refused}") from None
    try:
        aliases = tuple(_strings(values.get("aliases", [])))
    except ValueError as refused:
        raise ValueError(f"{where}: aliases {refused}") from None
    return ModelCapabilities(
        provider=provider,
        model=model,
        aliases=aliases,
        sources=_sorted_sources(_sources(values, facts, where)),
        **facts,
    )


def _parse(path: Path) -> list[ModelCapabilities]:
    """Every entry of a catalog file, or ``ValueError`` naming the file, the entry and the key."""
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as refused:
        raise ValueError(f"{path}: not a TOML file: {refused}") from refused
    version = data.pop("catalog_version", None)
    if type(version) is not int or version != CATALOG_VERSION:  # true == 1, and is refused
        raise ValueError(f"{path}: catalog_version must be {CATALOG_VERSION}, got {version!r}")
    entries: list[ModelCapabilities] = []
    for provider, models in data.items():
        if not isinstance(models, dict):
            raise ValueError(f"{path}: [{provider}] is a table of models, got {models!r}")
        entries.extend(
            _entry(provider, model, values, f'{path}: [{provider}."{model}"]')
            for model, values in models.items()
        )
    return entries


@cache
def _seed() -> tuple[ModelCapabilities, ...]:
    return tuple(_parse(_SEED))


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

type _Key = tuple[str, str]  # (provider, the canonical id)
type _Names = Mapping[str, Mapping[str, str]]  # provider -> an id or an alias -> the entry's id


@dataclass(frozen=True, slots=True, kw_only=True)
class _State:
    """One version of a catalog. Its layers never change once it is built: a change builds the
    next version, so an entry built from this one is kept with it (``built``) and never read
    after the change, and a read never waits for a writer."""

    seeded: Mapping[_Key, ModelCapabilities]
    loaded: Mapping[_Key, ModelCapabilities]
    registered: Mapping[_Key, ModelCapabilities]
    replaced: frozenset[_Key]
    names: _Names
    built: dict[_Key, ModelCapabilities] = field(default_factory=dict)


def _state(
    seeded: Mapping[_Key, ModelCapabilities],
    loaded: Mapping[_Key, ModelCapabilities],
    registered: Mapping[_Key, ModelCapabilities],
    replaced: frozenset[_Key] = frozenset(),
) -> _State:
    """The version with these layers, and the index of every id and alias they name."""
    names: dict[str, dict[str, str]] = {}
    for layer in (seeded, loaded, registered):
        for (provider, model), entry in layer.items():
            names.setdefault(provider, {}).update(dict.fromkeys((model, *entry.aliases), model))
    return _State(
        seeded=seeded, loaded=loaded, registered=registered, replaced=replaced, names=names
    )


def _canonical(names: _Names, entries: Iterable[ModelCapabilities]) -> list[ModelCapabilities]:
    """Each entry under the id of the one its id names, as :meth:`ModelCatalog.get` finds it
    (an id, an alias, or a dated snapshot of either), so a fact about an alias or a snapshot
    lands on its model; an alias that names another model of the provider is refused."""
    claimed = {provider: dict(own) for provider, own in names.items()}
    canonical: list[ModelCapabilities] = []
    for entry in entries:
        own = claimed.setdefault(entry.provider, {})
        found = lookup(entry.model, own)
        model = found.value if found is not None else entry.model
        own.setdefault(model, model)
        for alias in entry.aliases:
            owner = lookup(alias, own)
            if owner is not None and owner.value != model:
                raise ValueError(
                    f"{entry.provider}: {alias!r} already names {owner.value!r}, not {model!r}"
                )
            own.setdefault(alias, model)
        canonical.append(
            entry if model == entry.model else dataclasses.replace(entry, model=model)
        )
    return canonical


def _joined_aliases(entries: Iterable[ModelCapabilities]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(alias for entry in entries for alias in entry.aliases))


class ModelCatalog:
    """What each model takes through its adapter, by ``(provider, model)``, with its sources.

    It ships with the seed and the adapters' facts (``defaults=False`` starts empty, with
    neither). Layers, field by field: the seed and the adapters, then what ``load()`` read, then
    what ``register()`` said. Nothing in the toolkit reads the catalog: the adapters keep their
    own rules, and the catalog only tells them (D63). It is safe to share between threads.

    Usage::

        from ai_arch_toolkit.core import ModelCapabilities, model_catalog

        caps = model_catalog.get("claude-opus-5-5")
        fits = [c for c in model_catalog.entries() if c.tools and (c.context_window or 0) > n]
        model_catalog.register(ModelCapabilities(provider="ollama", model="llama3", tools=True))
        model_catalog.load("./my_catalog.toml")
        model_catalog.reset()
    """

    def __init__(self, *, defaults: bool = True) -> None:
        self._defaults = defaults
        self._writing = threading.Lock()  # one change at a time; a read takes the current state
        self._state = self._shipped()

    # ── Query ──

    def get(self, model: str, *, provider: str | None = None) -> ModelCapabilities | None:
        """The entry of ``model``, by its id, an alias or a dated snapshot of either; ``None``
        when unknown. ``provider`` defaults to the one the model id routes to, so an app's
        namespace is asked by name (``provider="ollama"``)."""
        provider = provider or _match_provider(model)
        if provider is None:
            return None
        state = self._state
        found = lookup(model, state.names.get(provider, {}))
        return self._resolve(state, (provider, found.value)) if found is not None else None

    def entries(self, provider: str | None = None) -> list[ModelCapabilities]:
        """Every entry, or ``provider``'s, sorted by provider and id."""
        state = self._state
        keys = {*state.seeded, *state.loaded, *state.registered}
        return [
            self._resolve(state, key)
            for key in sorted(keys)
            if provider is None or key[0] == provider
        ]

    # ── Overrides ──

    def register(self, capabilities: ModelCapabilities, *, replace: bool = False) -> None:
        """Say what a model takes, over the seed, the adapter and ``load()``, field by field: a
        fact left ``None`` keeps the one below it. A fact without a source is an override.

        Args:
            capabilities: The entry, by the model's id, an alias or a dated snapshot of either
                (the facts land on the entry ``get`` finds for it); its aliases must name no
                other model of its provider.
            replace: The entry is the whole of what is known: no fact below it shows through.
        """
        given = _sourced(capabilities, _OVERRIDE)
        with self._writing:
            state = self._state
            (entry,) = _canonical(state.names, [given])
            key = (entry.provider, entry.model)
            registered = dict(state.registered)
            current = registered.get(key)
            replaced = state.replaced
            if replace:
                registered[key] = entry
                replaced = replaced | {key}
            else:
                registered[key] = _merged(current, entry) if current else entry
            self._state = _state(state.seeded, state.loaded, registered, replaced)

    def load(self, path: str | Path) -> None:
        """Read a catalog file over the seed and the adapters, field by field. An entry may be
        keyed by the model's id, an alias or a dated snapshot of either.

        Raises:
            ValueError: The file is not TOML, or a key, a type, a vocabulary word or a source is
                wrong; the message names the file, the entry and the key, and nothing of the
                file is kept.
        """
        entries = _parse(Path(path))
        with self._writing:
            state = self._state
            loaded = dict(state.loaded)
            for entry in _canonical(state.names, entries):
                key = (entry.provider, entry.model)
                current = loaded.get(key)
                loaded[key] = _merged(current, entry) if current else entry
            self._state = _state(state.seeded, loaded, state.registered, state.replaced)

    def unregister(self, model: str, *, provider: str | None = None) -> None:
        """Forget ``model`` (its id, an alias or a dated snapshot of either) in every layer,
        until :meth:`reset`."""
        provider = provider or _match_provider(model)
        if provider is None:
            return
        with self._writing:
            state = self._state
            found = lookup(model, state.names.get(provider, {}))
            if found is None:
                return
            key = (provider, found.value)
            seeded, loaded, registered = (
                {k: entry for k, entry in layer.items() if k != key}
                for layer in (state.seeded, state.loaded, state.registered)
            )
            self._state = _state(seeded, loaded, registered, state.replaced - {key})

    def reset(self) -> None:
        """Back to the shipped seed and the adapters, discarding what was loaded or said."""
        with self._writing:
            self._state = self._shipped()

    # ── Internals ──

    def _shipped(self) -> _State:
        seeded = {(e.provider, e.model): e for e in _seed()} if self._defaults else {}
        return _state(seeded, {}, {})

    def _resolve(self, state: _State, key: _Key) -> ModelCapabilities:
        found = state.built.get(key)
        if found is None:
            found = state.built.setdefault(key, self._build(state, key))
        return found

    def _build(self, state: _State, key: _Key) -> ModelCapabilities:
        layers = [
            layer[key] for layer in (state.seeded, state.loaded, state.registered) if key in layer
        ]
        if key in state.replaced:  # no fact below shows through; every id that names it does
            return dataclasses.replace(state.registered[key], aliases=_joined_aliases(layers))
        provider, model = key
        entry = ModelCapabilities(provider=provider, model=model)
        for layer_entry in layers:
            entry = _merged(entry, layer_entry)
        adapter = _adapter_of(provider) if self._defaults else None
        return _with_adapter(entry, adapter) if adapter is not None else entry


# ── Module-level singleton ──
model_catalog = ModelCatalog()
