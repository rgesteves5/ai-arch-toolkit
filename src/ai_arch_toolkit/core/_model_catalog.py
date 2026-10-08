"""The technical model catalog: what a model takes through its adapter, each fact with its source.

Apart from ``pricing``, which knows rates only. A fact is ``None`` when unknown, never "no"
(D63). The shipped seed (``_default_catalog.toml``) holds what the providers publish, limits and
modalities, each with its page and the day it was read. What the adapter does with a model
(tools, tool choices, reasoning, server tools) comes from the adapter's own tables, through its
pure ``model_facts`` classmethod; no adapter reads the catalog. An app overrides any fact:
``load()`` over the seed and the adapter, ``register()`` over everything, field by field.

Ids match as everywhere in ``core`` (``_model_id.lookup``): an entry's id, an alias, or a dated
snapshot of either; never a family prefix, since a variant is another model.
"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from datetime import date
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._providers import _match_provider
from ai_arch_toolkit.core._providers._base import (
    EFFORT_ORDER,
    AdapterFacts,
    InputModality,
    ThinkingMode,
    ToolChoiceMode,
    ordered_efforts,
)

if TYPE_CHECKING:
    from ai_arch_toolkit.core._providers._base import BaseProvider

__all__ = ["ModelCapabilities", "ModelCatalog", "Provenance", "model_catalog"]

type SourceKind = Literal["docs", "probe", "api", "adapter", "override"]
type OutputModality = Literal["text", "image"]

CATALOG_VERSION = 1
_SEED = Path(__file__).with_name("_default_catalog.toml")


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
        if not self.provider or not self.model:
            raise ValueError(
                f"an entry needs a provider and a model, got {self.provider!r}, {self.model!r}"
            )
        for name, _ in self.sources:
            _check_fact(name)

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
# The adapter's facts a layer may leave unset; its input modalities narrow a published list.
_ADAPTER_FACTS = tuple(f.name for f in fields(AdapterFacts) if f.name != "input_modalities")
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
    narrowed to what the adapter sends (C06.1): an app's observation is never narrowed."""
    facts = adapter.model_facts(entry.model)
    source = Provenance(
        kind="adapter", ref=f"{adapter.__module__}.{adapter.__qualname__}.model_facts"
    )
    changes: dict[str, Any] = {
        name: value
        for name in _ADAPTER_FACTS
        if (value := getattr(facts, name)) is not None and getattr(entry, name) is None
    }
    if (narrowed := _narrowed(entry, facts.input_modalities)) is not None:
        changes["input_modalities"] = narrowed
    sources = dict(entry.sources) | dict.fromkeys(changes, source)
    return dataclasses.replace(entry, sources=_sorted_sources(sources), **changes)


def _narrowed(
    entry: ModelCapabilities, sent: frozenset[InputModality] | None
) -> frozenset[InputModality] | None:
    """The published inputs without those the adapter does not send, when it drops some; an
    input list of any other kind (a probe's, an override) is the app's word, left as it is."""
    published = entry.input_modalities
    stated = entry.provenance("input_modalities")
    if published is None or sent is None or stated is None or stated.kind != "docs":
        return None
    return None if published <= sent else published & sent


def _adapter_of(provider: str) -> type[BaseProvider[Any, Any]] | None:
    """The adapter of a provider's own host; ``None`` for an app's namespace, or for an adapter
    whose SDK extra is not installed (its facts are then unknown)."""
    try:
        return _import_adapter(provider)
    except ImportError:
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
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"must be a list of names, got {value!r}")
    if vocabulary is not None and (unknown := sorted(set(value) - set(vocabulary))):
        raise ValueError(f"takes {', '.join(vocabulary)}; not {', '.join(unknown)}")
    return value


def _choice(vocabulary: tuple[str, ...]) -> Callable[[object], object]:
    def parse(value: object) -> object:
        if value not in vocabulary:
            raise ValueError(f"must be one of {', '.join(vocabulary)}, got {value!r}")
        return value

    return parse


def _set_of(vocabulary: tuple[str, ...] | None) -> Callable[[object], object]:
    return lambda value: frozenset(_strings(value, vocabulary))


def _vocabulary(alias: Any) -> tuple[str, ...]:
    return get_args(alias.__value__)


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
    "server_tools": _set_of(None),
}


def _provenance(raw: object, where: str) -> Provenance:
    """A source table: its kind, its ref, and the day it was read, a TOML date."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a source is a table (kind, ref, verified_at), got {raw!r}")
    if unknown := sorted(set(raw) - _SOURCE_KEYS):
        raise ValueError(
            f"{where}: unknown key {unknown[0]!r} (a source has kind, ref, verified_at)"
        )
    kinds = _vocabulary(SourceKind)
    if raw.get("kind") not in kinds:
        raise ValueError(
            f"{where}: kind must be one of {', '.join(kinds)}, got {raw.get('kind')!r}"
        )
    ref = raw.get("ref")
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"{where}: ref must name the page, run or code, got {ref!r}")
    day = raw.get("verified_at")
    if type(day) is not date:  # a TOML datetime is a date too, and is refused
        raise ValueError(
            f"{where}: verified_at must be a TOML date such as 2026-10-08, got {day!r}"
        )
    return Provenance(kind=raw["kind"], ref=ref, verified_at=day)


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
    """Every entry of a catalog file, or ``ValueError`` naming the entry and the key."""
    with path.open("rb") as file:
        data = tomllib.load(file)
    version = data.pop("catalog_version", None)
    if version != CATALOG_VERSION:
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


class ModelCatalog:
    """What each model takes through its adapter, by ``(provider, model)``, with its sources.

    It ships with the seed and the adapters' facts (``defaults=False`` starts empty, with
    neither). Layers, field by field: the seed and the adapters, then what ``load()`` read, then
    what ``register()`` said. Nothing in the toolkit reads the catalog: the adapters keep their
    own rules, and the catalog only tells them (D63).

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
        self._seeded: dict[_Key, ModelCapabilities] = {}
        self._loaded: dict[_Key, ModelCapabilities] = {}
        self._registered: dict[_Key, ModelCapabilities] = {}
        self._replaced: set[_Key] = set()
        self._cache: dict[_Key, ModelCapabilities] = {}
        self._names: dict[str, dict[str, str]] | None = None  # provider -> id or alias -> id
        self.reset()

    # ── Query ──

    def get(self, model: str, *, provider: str | None = None) -> ModelCapabilities | None:
        """The entry of ``model``, by its id, an alias or a dated snapshot of either; ``None``
        when unknown. ``provider`` defaults to the one the model id routes to, so an app's
        namespace is asked by name (``provider="ollama"``)."""
        provider = provider or _match_provider(model)
        if provider is None:
            return None
        found = lookup(model, self._index().get(provider, {}))
        return self._resolve((provider, found.value)) if found is not None else None

    def entries(self, provider: str | None = None) -> list[ModelCapabilities]:
        """Every entry, or ``provider``'s, sorted by provider and id."""
        keys = {*self._seeded, *self._loaded, *self._registered}
        return [
            self._resolve(key) for key in sorted(keys) if provider is None or key[0] == provider
        ]

    # ── Overrides ──

    def register(self, capabilities: ModelCapabilities, *, replace: bool = False) -> None:
        """Say what a model takes, over the seed, the adapter and ``load()``, field by field: a
        fact left ``None`` keeps the one below it. A fact without a source is an override.

        Args:
            capabilities: The entry; its id and aliases must name no other model of its
                provider.
            replace: The entry is the whole of what is known: nothing below it shows through.
        """
        entry = _sourced(capabilities, _OVERRIDE)
        self._claim([entry])
        key = (entry.provider, entry.model)
        current = self._registered.get(key)
        if replace:
            self._registered[key] = entry
            self._replaced.add(key)
        else:
            self._registered[key] = _merged(current, entry) if current else entry
        self._changed()

    def load(self, path: str | Path) -> None:
        """Read a catalog file over the seed and the adapters, field by field.

        Raises:
            ValueError: A key, a type, a vocabulary word or a source is wrong; the message
                names the entry and the key, and nothing of the file is kept.
        """
        entries = _parse(Path(path))
        self._claim(entries)
        for entry in entries:
            key = (entry.provider, entry.model)
            current = self._loaded.get(key)
            self._loaded[key] = _merged(current, entry) if current else entry
        self._changed()

    def unregister(self, model: str, *, provider: str | None = None) -> None:
        """Forget ``model`` (its id or an alias) in every layer, until :meth:`reset`."""
        provider = provider or _match_provider(model)
        canonical = self._index().get(provider or "", {}).get(model)
        if provider is None or canonical is None:
            return
        key = (provider, canonical)
        for layer in (self._seeded, self._loaded, self._registered):
            layer.pop(key, None)
        self._replaced.discard(key)
        self._changed()

    def reset(self) -> None:
        """Back to the shipped seed and the adapters, discarding what was loaded or said."""
        self._seeded = {(e.provider, e.model): e for e in _seed()} if self._defaults else {}
        self._loaded.clear()
        self._registered.clear()
        self._replaced.clear()
        self._changed()

    # ── Internals ──

    def _changed(self) -> None:
        self._cache.clear()
        self._names = None

    def _index(self) -> dict[str, dict[str, str]]:
        if self._names is None:
            names: dict[str, dict[str, str]] = {}
            for layer in (self._seeded, self._loaded, self._registered):
                for (provider, model), entry in layer.items():
                    own = names.setdefault(provider, {})
                    own.update(dict.fromkeys((model, *entry.aliases), model))
            self._names = names
        return self._names

    def _claim(self, entries: Iterable[ModelCapabilities]) -> None:
        """Refuse an id or an alias that already names another model of the provider."""
        names = {provider: dict(own) for provider, own in self._index().items()}
        for entry in entries:
            own = names.setdefault(entry.provider, {})
            for name in (entry.model, *entry.aliases):
                owner = own.setdefault(name, entry.model)
                if owner != entry.model:
                    raise ValueError(
                        f"{entry.provider}: {name!r} already names {owner!r}; "
                        f"register {entry.model!r} under its own id"
                    )

    def _resolve(self, key: _Key) -> ModelCapabilities:
        if key not in self._cache:
            self._cache[key] = self._build(key)
        return self._cache[key]

    def _build(self, key: _Key) -> ModelCapabilities:
        if key in self._replaced:
            return self._registered[key]
        provider, model = key
        entry = ModelCapabilities(provider=provider, model=model)
        for layer in (self._seeded, self._loaded, self._registered):
            if key in layer:
                entry = _merged(entry, layer[key])
        adapter = _adapter_of(provider) if self._defaults else None
        return _with_adapter(entry, adapter) if adapter is not None else entry


# ── Module-level singleton ──
model_catalog = ModelCatalog()
