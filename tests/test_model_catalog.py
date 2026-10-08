"""The technical model catalog (C06, D63): matching, layers, the strict loader, and the contract
between each adapter and the facts the catalog takes from it."""

from __future__ import annotations

import ast
import json
import re
import sys
import threading
import types
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from ai_arch_toolkit.core import (
    ModelCapabilities,
    ModelCatalog,
    OutputSchema,
    Provenance,
    RequestError,
    code_execution,
    document,
    image_generation,
    model_catalog,
    prepare_tools,
    tool,
    user,
    web_search,
)
from ai_arch_toolkit.core._providers import create_provider
from ai_arch_toolkit.core._providers._base import SERVER_TOOL_TYPES
from tests.provider_calls import prepare

DOCS = Provenance(kind="docs", ref="https://example.com/models", verified_at=date(2026, 10, 8))
SRC = Path(__file__).resolve().parents[1] / "src/ai_arch_toolkit/core"


def _catalog(*entries: ModelCapabilities) -> ModelCatalog:
    catalog = ModelCatalog(defaults=False)
    for entry in entries:
        catalog.register(entry)
    return catalog


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "catalog.toml"
    path.write_text("catalog_version = 1\n" + body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Matching (C06.2: _model_id.lookup, never a family)
# ---------------------------------------------------------------------------


def test_an_id_finds_its_entry_and_its_dated_snapshots_but_never_a_familys() -> None:
    catalog = _catalog(
        ModelCapabilities(provider="anthropic", model="claude-fable-5", context_window=1_000)
    )

    assert catalog.get("claude-fable-5") is not None
    found = catalog.get("claude-fable-5-20260101")
    assert found is not None and found.model == "claude-fable-5"
    assert catalog.get("claude-fable-5-1") is None  # another model, unknown here


def test_an_alias_gives_the_canonical_entry() -> None:
    catalog = _catalog(
        ModelCapabilities(
            provider="xai",
            model="grok-4.20-reasoning",
            aliases=("grok-4.20-0309-reasoning",),
            context_window=2_000,
        )
    )

    found = catalog.get("grok-4.20-0309-reasoning")

    assert found is not None
    assert found.model == "grok-4.20-reasoning"
    assert found.aliases == ("grok-4.20-0309-reasoning",)


def test_an_app_namespace_never_shadows_the_providers_own_entry() -> None:
    local = ModelCapabilities(provider="ollama", model="gpt-4o", context_window=8_192)
    catalog = _catalog(local)

    assert catalog.get("gpt-4o") is None  # inferred as openai: the app's entry is elsewhere
    catalog.register(ModelCapabilities(provider="openai", model="gpt-4o", context_window=128_000))

    hosted = catalog.get("gpt-4o")
    assert hosted is not None and hosted.provider == "openai" and hosted.context_window == 128_000
    mine = catalog.get("gpt-4o", provider="ollama")
    assert mine is not None and mine.context_window == 8_192


def test_an_unknown_model_or_provider_is_none() -> None:
    catalog = _catalog(ModelCapabilities(provider="openai", model="gpt-5", context_window=1))

    assert catalog.get("gpt-7") is None
    assert catalog.get("llama3") is None  # no provider claims it
    assert catalog.get("gpt-5", provider="ollama") is None


def test_an_alias_that_names_another_entry_is_refused() -> None:
    catalog = _catalog(ModelCapabilities(provider="xai", model="grok-a", context_window=1))

    with pytest.raises(ValueError, match="grok-a"):
        catalog.register(
            ModelCapabilities(provider="xai", model="grok-b", aliases=("grok-a",), tools=True)
        )


def test_register_load_and_unregister_name_the_entry_get_finds(tmp_path: Path) -> None:
    """An id names the entry ``get`` finds for it: its own, an alias's or a dated snapshot's.
    xAI's inventory ids are aliases, and Anthropic's snapshots are dated."""
    catalog = ModelCatalog()
    catalog.register(
        ModelCapabilities(provider="xai", model="grok-4.20-reasoning", output_token_limit=7)
    )
    catalog.load(
        _write(tmp_path, f'[xai."grok-4.20-reasoning"]\nparallel_tool_calls = true\n{_SOURCE}\n')
    )

    found = catalog.get("grok-4.20-0309-reasoning")
    assert found is not None and found.model == "grok-4.20-0309-reasoning"
    assert (found.output_token_limit, found.parallel_tool_calls) == (7, True)
    assert found.context_window == 1_000_000  # the seed's, kept
    assert found.aliases == ("grok-4.20-reasoning",)
    assert "grok-4.20-reasoning" not in {entry.model for entry in catalog.entries("xai")}

    catalog.register(
        ModelCapabilities(
            provider="anthropic", model="claude-haiku-4-5-20251001", parallel_tool_calls=True
        )
    )
    haiku = catalog.get("claude-haiku-4-5")
    assert haiku is not None and haiku.parallel_tool_calls is True
    assert haiku.context_window == 200_000

    catalog.unregister("claude-haiku-4-5-20251001")
    assert catalog.get("claude-haiku-4-5") is None


def test_replace_through_an_alias_keeps_the_ids_that_name_the_entry() -> None:
    catalog = ModelCatalog()

    catalog.register(
        ModelCapabilities(provider="xai", model="grok-4.20-reasoning", tools=False), replace=True
    )

    found = catalog.get("grok-4.20-reasoning")
    assert found is not None and found.model == "grok-4.20-0309-reasoning"
    assert found.tools is False and found.context_window is None
    assert found.aliases == ("grok-4.20-reasoning",)


# ---------------------------------------------------------------------------
# Layers: seed < load() < register(), field by field
# ---------------------------------------------------------------------------


def test_an_override_keeps_the_provenance_of_the_other_fields(tmp_path: Path) -> None:
    catalog = ModelCatalog(defaults=False)
    catalog.load(
        _write(
            tmp_path,
            """
[openai."gpt-x"]
context_window = 400_000
output_token_limit = 128_000
source = { kind = "docs", ref = "https://example.com/gpt-x", verified_at = 2026-10-08 }
""",
        )
    )
    mine = Provenance(kind="override", ref="our eval", verified_at=date(2026, 10, 8))

    catalog.register(
        ModelCapabilities(
            provider="openai",
            model="gpt-x",
            output_token_limit=64_000,
            sources=(("output_token_limit", mine),),
        )
    )

    found = catalog.get("gpt-x")
    assert found is not None
    assert (found.context_window, found.output_token_limit) == (400_000, 64_000)
    assert found.provenance("output_token_limit") == mine
    context = found.provenance("context_window")
    assert context is not None and context.kind == "docs"


def test_a_registered_fact_without_a_source_is_an_override() -> None:
    catalog = _catalog(ModelCapabilities(provider="ollama", model="llama3", tools=True))

    found = catalog.get("llama3", provider="ollama")
    assert found is not None
    source = found.provenance("tools")
    assert source is not None and source.kind == "override"


def test_load_wins_over_the_seed_register_over_load_and_reset_restores(tmp_path: Path) -> None:
    catalog = ModelCatalog()
    seeded = catalog.get("gpt-5")
    assert seeded is not None and seeded.context_window is not None
    path = _write(
        tmp_path,
        """
[openai."gpt-5"]
context_window = 1_234
output_token_limit = 99
source = { kind = "probe", ref = "run 1", verified_at = 2026-10-08 }
""",
    )

    catalog.load(path)
    loaded = catalog.get("gpt-5")
    assert loaded is not None and (loaded.context_window, loaded.output_token_limit) == (1_234, 99)
    assert loaded.tools == seeded.tools  # the facts it does not name stay

    catalog.register(ModelCapabilities(provider="openai", model="gpt-5", context_window=5))
    registered = catalog.get("gpt-5")
    assert registered is not None and (
        registered.context_window,
        registered.output_token_limit,
    ) == (
        5,
        99,
    )

    catalog.reset()
    assert catalog.get("gpt-5") == seeded


def test_replace_drops_every_fact_below_it() -> None:
    catalog = ModelCatalog()
    assert (seeded := catalog.get("gpt-5")) is not None and seeded.context_window is not None

    catalog.register(
        ModelCapabilities(provider="openai", model="gpt-5", tools=False), replace=True
    )

    found = catalog.get("gpt-5")
    assert found is not None
    assert found.tools is False
    assert found.context_window is None and found.thinking_efforts is None


def test_unregister_removes_an_entry_until_reset() -> None:
    catalog = ModelCatalog()
    assert catalog.get("gpt-5") is not None

    catalog.unregister("gpt-5")
    assert catalog.get("gpt-5") is None
    assert all(entry.model != "gpt-5" for entry in catalog.entries())

    catalog.reset()
    assert catalog.get("gpt-5") is not None


def test_a_catalog_without_defaults_starts_empty() -> None:
    assert ModelCatalog(defaults=False).entries() == []
    assert ModelCatalog(defaults=False).get("gpt-5") is None


def test_entries_filter_by_provider_and_come_sorted() -> None:
    catalog = _catalog(
        ModelCapabilities(provider="openai", model="gpt-b", tools=True),
        ModelCapabilities(provider="openai", model="gpt-a", tools=True),
        ModelCapabilities(provider="ollama", model="llama3", tools=True),
    )

    assert [entry.model for entry in catalog.entries(provider="openai")] == ["gpt-a", "gpt-b"]
    assert [entry.provider for entry in catalog.entries()] == ["ollama", "openai", "openai"]


def test_the_auto_filter_excludes_an_unknown_fact() -> None:
    catalog = _catalog(
        ModelCapabilities(
            provider="openai",
            model="gpt-known",
            tools=True,
            structured_output=True,
            input_token_limit=200_000,
        ),
        ModelCapabilities(provider="openai", model="gpt-unknown", tools=True, input_token_limit=1),
        ModelCapabilities(
            provider="openai", model="gpt-small", tools=True, structured_output=True
        ),
    )

    picked = [
        c.model
        for c in catalog.entries()
        if c.tools and c.structured_output and (c.input_token_limit or 0) >= 100_000
    ]

    assert picked == ["gpt-known"]  # None never reads as yes


def test_to_dict_is_plain_data_with_each_fields_source() -> None:
    entry = ModelCapabilities(
        provider="openai",
        model="gpt-x",
        input_modalities=frozenset({"text", "image"}),
        thinking_efforts=("low", "high"),
        sources=(("input_modalities", DOCS), ("thinking_efforts", DOCS)),
    )

    data = entry.to_dict()

    assert data["input_modalities"] == ["image", "text"]
    assert data["thinking_efforts"] == ["low", "high"]
    assert data["tools"] is None
    assert data["sources"] == {
        name: {"kind": "docs", "ref": "https://example.com/models", "verified_at": "2026-10-08"}
        for name in ("input_modalities", "thinking_efforts")
    }


def test_an_entry_names_its_provider_and_its_model() -> None:
    with pytest.raises(ValueError, match="provider and a model"):
        ModelCapabilities(provider="", model="gpt-x")
    with pytest.raises(ValueError, match="provider and a model"):
        ModelCapabilities(provider="openai", model="")


def test_a_source_must_name_a_fact() -> None:
    with pytest.raises(ValueError, match="tool"):
        ModelCapabilities(provider="openai", model="gpt-x", sources=(("tool", DOCS),))
    with pytest.raises(ValueError, match="tool"):
        ModelCapabilities(provider="openai", model="gpt-x").provenance("tool")


@pytest.mark.parametrize(
    ("fact", "value"),
    [
        ("tools", "yes"),
        ("tools", 1),
        ("context_window", -1),
        ("context_window", True),
        ("input_modalities", {"txt"}),
        ("input_modalities", "text"),
        ("output_modalities", {"audio"}),
        ("tool_choice_modes", {"forced"}),
        ("thinking_mode", "sometimes"),
        ("thinking_efforts", ["hgh"]),
        ("server_tools", {"web_serch"}),
    ],
)
def test_register_refuses_what_load_refuses(fact: str, value: object) -> None:
    catalog = ModelCatalog(defaults=False)

    with pytest.raises(ValueError, match=fact):
        catalog.register(ModelCapabilities(provider="openai", model="gpt-x", **{fact: value}))
    assert catalog.entries() == []


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "rumour", "ref": "x"},
        {"kind": "docs", "ref": ""},
        {"kind": "docs", "ref": "x", "verified_at": datetime(2026, 10, 8, 10, 0)},
    ],
)
def test_a_source_is_checked_as_the_loader_checks_it(source: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Provenance(**source)


def test_an_entry_keeps_its_sets_and_efforts_as_the_loader_does() -> None:
    entry = ModelCapabilities(
        provider="openai",
        model="gpt-x",
        aliases=["gpt-y"],  # type: ignore[arg-type]
        input_modalities={"text", "image"},  # type: ignore[arg-type]
        thinking_efforts=["high", "low", "low"],  # type: ignore[arg-type]
        sources=(("input_modalities", DOCS),),
    )

    assert entry.aliases == ("gpt-y",)
    assert isinstance(entry.input_modalities, frozenset)
    assert entry.thinking_efforts == ("low", "high")
    assert json.loads(json.dumps(entry.to_dict()))["input_modalities"] == ["image", "text"]


# ---------------------------------------------------------------------------
# Threads: a read that overlaps a change never keeps what it built before it
# ---------------------------------------------------------------------------


def test_a_read_that_overlaps_a_register_never_keeps_the_entry_it_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader building an entry is in the adapter's ``model_facts``, Python code where a
    thread may switch; here it waits there, made certain, while the writer registers."""
    from ai_arch_toolkit.core._providers._openai import OpenAIProvider

    catalog = ModelCatalog()
    building, written = threading.Event(), threading.Event()
    real = OpenAIProvider.model_facts.__func__  # type: ignore[attr-defined]

    def paused(cls: type[OpenAIProvider], model: str) -> Any:
        if threading.current_thread().name == "reader":
            building.set()
            written.wait(timeout=5)
        return real(cls, model)

    monkeypatch.setattr(OpenAIProvider, "model_facts", classmethod(paused))
    reader = threading.Thread(target=catalog.get, args=("gpt-5.4-mini",), name="reader")
    reader.start()
    assert building.wait(timeout=5)

    catalog.register(ModelCapabilities(provider="openai", model="gpt-5.4-mini", context_window=7))
    written.set()
    reader.join(timeout=5)

    found = catalog.get("gpt-5.4-mini")
    assert found is not None and found.context_window == 7


def test_a_read_that_overlaps_a_load_never_hides_the_model_it_added(tmp_path: Path) -> None:
    """A reader walking an entry's aliases waits there, made certain, while the writer loads a
    new model: the model is found after."""
    reading, written = threading.Event(), threading.Event()

    class PausedAliases(tuple[str, ...]):
        def __iter__(self) -> Any:
            if threading.current_thread().name == "reader" and not reading.is_set():
                reading.set()
                written.wait(timeout=5)
            return super().__iter__()

    catalog = ModelCatalog(defaults=False)
    catalog.register(
        ModelCapabilities(
            provider="xai", model="grok-a", aliases=PausedAliases(("grok-b",)), context_window=1
        )
    )
    reader = threading.Thread(target=catalog.get, args=("grok-a",), name="reader")
    reader.start()
    assert reading.wait(timeout=5)

    catalog.load(_write(tmp_path, f'[ollama."llama3"]\ntools = true\n{_SOURCE}\n'))
    written.set()
    reader.join(timeout=5)

    assert catalog.get("llama3", provider="ollama") is not None
    assert catalog.get("grok-b") is not None


# ---------------------------------------------------------------------------
# The strict loader
# ---------------------------------------------------------------------------

_SOURCE = 'source = { kind = "docs", ref = "https://example.com", verified_at = 2026-10-08 }'


@pytest.mark.parametrize(
    ("body", "names"),
    [
        (f'[openai."gpt-x"]\ntool = true\n{_SOURCE}\n', ("gpt-x", "tool")),
        (
            '[openai."gpt-x"]\ntools = true\n'
            'source = { kind = "docs", ref = "https://example.com" }\n',
            ("gpt-x", "verified_at"),
        ),
        (f'[openai."gpt-x"]\ncontext_window = "1M"\n{_SOURCE}\n', ("gpt-x", "context_window")),
        (f'[openai."gpt-x"]\ncontext_window = 0\n{_SOURCE}\n', ("gpt-x", "context_window")),
        (f'[openai."gpt-x"]\ntools = 1\n{_SOURCE}\n', ("gpt-x", "tools")),
        (
            f'[openai."gpt-x"]\ninput_modalities = ["text", "smell"]\n{_SOURCE}\n',
            ("gpt-x", "input_modalities", "smell"),
        ),
        (
            f'[openai."gpt-x"]\nthinking_mode = "sometimes"\n{_SOURCE}\n',
            ("gpt-x", "thinking_mode"),
        ),
        (
            f'[openai."gpt-x"]\nthinking_efforts = ["low", "turbo"]\n{_SOURCE}\n',
            ("gpt-x", "thinking_efforts"),
        ),
        (
            f'[openai."gpt-x"]\nserver_tools = ["web_serch"]\n{_SOURCE}\n',
            ("gpt-x", "server_tools", "web_serch"),
        ),
        ('[openai."gpt-x"]\ntools = true\n', ("gpt-x", "tools", "source")),
        (
            '[openai."gpt-x"]\ntools = true\n'
            'source = { kind = "rumour", ref = "x", verified_at = 2026-10-08 }\n',
            ("gpt-x", "kind"),
        ),
        (
            '[openai."gpt-x"]\ntools = true\n'
            'source = { kind = "docs", ref = "x", verified_at = 2026-10-08T10:00:00 }\n',
            ("gpt-x", "verified_at"),
        ),
        (
            f'[openai."gpt-x"]\ntools = true\n{_SOURCE}\n'
            'sources.context_window = { kind = "docs", ref = "x", verified_at = 2026-10-08 }\n',
            ("gpt-x", "sources", "context_window"),
        ),
        (f'[openai."gpt-x"]\naliases = "gpt-y"\ntools = true\n{_SOURCE}\n', ("gpt-x", "aliases")),
        ("[openai]\nmodel = 1\n", ("openai", "model")),
    ],
)
def test_the_loader_names_the_entry_and_the_key_it_refuses(
    tmp_path: Path, body: str, names: tuple[str, ...]
) -> None:
    catalog = ModelCatalog(defaults=False)

    with pytest.raises(ValueError) as refused:
        catalog.load(_write(tmp_path, body))

    for name in names:
        assert name in str(refused.value)
    assert catalog.entries() == []  # nothing of a refused file is kept


@pytest.mark.parametrize(
    "version", ["", "catalog_version = true\n", "catalog_version = 2\n", 'catalog_version = "1"\n']
)
def test_the_loader_needs_its_version(tmp_path: Path, version: str) -> None:
    path = tmp_path / "catalog.toml"
    path.write_text(f'{version}[openai."gpt-x"]\ntools = true\n{_SOURCE}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="catalog_version"):
        ModelCatalog(defaults=False).load(path)


def test_a_file_that_is_not_toml_is_refused_by_its_path(tmp_path: Path) -> None:
    path = _write(tmp_path, '[openai."gpt-x"\ntools = true\n')

    with pytest.raises(ValueError, match=re.escape(str(path))):
        ModelCatalog(defaults=False).load(path)


def test_the_loader_lists_each_effort_once(tmp_path: Path) -> None:
    catalog = ModelCatalog(defaults=False)
    catalog.load(
        _write(
            tmp_path, f'[openai."gpt-x"]\nthinking_efforts = ["high", "low", "low"]\n{_SOURCE}\n'
        )
    )

    found = catalog.get("gpt-x")
    assert found is not None and found.thinking_efforts == ("low", "high")


def test_the_loader_reads_every_fact_with_its_source(tmp_path: Path) -> None:
    catalog = ModelCatalog(defaults=False)
    catalog.load(
        _write(
            tmp_path,
            f"""
[ollama."llama3"]
aliases = ["llama3:8b"]
context_window = 8_192
input_token_limit = 8_000
output_token_limit = 2_048
input_modalities = ["text"]
output_modalities = ["text"]
tools = true
tool_choice_modes = ["auto", "none"]
parallel_tool_calls = false
structured_output = true
json_mode = true
streaming = true
thinking_mode = "none"
thinking_efforts = []
thinking_budget = false
server_tools = []
{_SOURCE}
sources.tools = {{ kind = "probe", ref = "run 7", verified_at = 2026-10-01 }}
""",
        )
    )

    found = catalog.get("llama3:8b", provider="ollama")

    assert found is not None
    every_fact = set(found.to_dict()) - {"provider", "model", "aliases", "sources"}
    assert all(getattr(found, name) is not None for name in every_fact)  # a new fact needs a line
    assert found.model == "llama3"
    assert (found.context_window, found.input_token_limit, found.output_token_limit) == (
        8_192,
        8_000,
        2_048,
    )
    assert found.input_modalities == frozenset({"text"})
    assert found.tool_choice_modes == frozenset({"auto", "none"})
    assert found.thinking_mode == "none" and found.thinking_efforts == ()
    assert found.server_tools == frozenset()
    assert found.provenance("tools") == Provenance(
        kind="probe", ref="run 7", verified_at=date(2026, 10, 1)
    )
    context = found.provenance("context_window")
    assert context is not None and context.kind == "docs"


# ---------------------------------------------------------------------------
# The adapter's facts (C06.4): from its own tables, kind="adapter"
# ---------------------------------------------------------------------------


def test_the_adapters_facts_come_with_their_provenance() -> None:
    found = model_catalog.get("gpt-6-astra")

    assert found is not None
    source = found.provenance("thinking_efforts")
    assert source is not None and source.kind == "adapter"
    assert "OpenAIProvider" in source.ref


def test_astras_efforts_are_its_row_in_the_adapters_table() -> None:
    from ai_arch_toolkit.core._providers import _openai

    found = model_catalog.get("gpt-6-astra")

    assert found is not None and found.thinking_efforts is not None
    assert set(found.thinking_efforts) == _openai._MODELS["gpt-6-astra"].efforts
    assert found.thinking_mode == "always"  # no "none": Astra always reasons


def test_meta_takes_only_auto_and_none() -> None:
    found = model_catalog.get("muse-spark-1.3")

    assert found is not None
    assert found.tool_choice_modes == frozenset({"auto", "none"})


def test_without_the_sdk_extra_the_adapters_facts_are_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ai_arch_toolkit.core._providers._anthropic", None)
    catalog = ModelCatalog()

    found = catalog.get("claude-opus-5-5")

    assert found is not None
    assert found.context_window is not None  # published, from the seed
    assert found.tools is None and found.thinking_efforts is None and found.server_tools is None


def test_a_published_modality_the_adapter_does_not_send_is_left_out() -> None:
    """Meta publishes video and audio input; no content part of the toolkit carries them
    (C06.1)."""
    found = model_catalog.get("muse-spark-1.3")

    assert found is not None
    assert found.input_modalities == frozenset({"text", "image", "pdf"})
    source = found.provenance("input_modalities")
    assert source is not None and source.kind == "adapter"


def test_the_adapter_narrows_what_a_page_publishes_not_what_a_run_saw(tmp_path: Path) -> None:
    """The xAI adapter drops documents: a published PDF input does not reach the model, while a
    probe's observation is the app's word."""
    catalog = ModelCatalog()
    catalog.load(
        _write(
            tmp_path,
            f"""
[xai."grok-9"]
input_modalities = ["text", "image", "pdf"]
{_SOURCE}

[xai."grok-9-seen"]
input_modalities = ["text", "image", "pdf"]
source = {{ kind = "probe", ref = "run 3", verified_at = 2026-10-08 }}
""",
        )
    )

    published = catalog.get("grok-9")
    seen = catalog.get("grok-9-seen")

    assert published is not None and published.input_modalities == frozenset({"text", "image"})
    narrowed = published.provenance("input_modalities")
    assert narrowed is not None and narrowed.kind == "adapter"
    assert seen is not None and seen.input_modalities == frozenset({"text", "image", "pdf"})


def test_the_adapter_narrows_a_models_endpoint_list_too(tmp_path: Path) -> None:
    """A list from the provider's models endpoint (``kind="api"``) is the provider's word, as
    its page is."""
    catalog = ModelCatalog()
    catalog.load(
        _write(
            tmp_path,
            """
[xai."grok-9"]
input_modalities = ["text", "image", "pdf"]
source = { kind = "api", ref = "GET /v1/models/grok-9", verified_at = 2026-10-08 }
""",
        )
    )

    found = catalog.get("grok-9")

    assert found is not None and found.input_modalities == frozenset({"text", "image"})
    narrowed = found.provenance("input_modalities")
    assert narrowed is not None and narrowed.kind == "adapter"


def test_a_gemini_image_model_reads_the_pdfs_its_completions_carry() -> None:
    """Gemini's image models complete: a PDF reaches them, video does not (C06.1)."""
    found = model_catalog.get("gemini-3.1-flash-image")

    assert found is not None
    assert found.input_modalities == frozenset({"text", "image", "pdf"})


def test_the_images_api_narrows_a_published_text_output() -> None:
    """GPT Image 1.5's page says it writes images and text; the Images API, the adapter's only
    way to it, answers with images (C06.1)."""
    found = model_catalog.get("gpt-image-1.5")

    assert found is not None and found.output_modalities == frozenset({"image"})
    source = found.provenance("output_modalities")
    assert source is not None and source.kind == "adapter"


def test_a_broken_sdk_leaves_the_adapters_facts_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed SDK that fails to import (a protobuf built for another version raises
    ``TypeError``) is as good as a missing one: the adapter's facts are unknown."""

    class Broken(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            raise TypeError("Descriptors cannot be created directly")

    monkeypatch.setitem(sys.modules, "ai_arch_toolkit.core._providers._xai", Broken("_xai"))
    catalog = ModelCatalog()

    found = catalog.get("grok-4.7")

    assert found is not None and found.context_window == 500_000
    assert found.tools is None and found.server_tools is None
    assert all(entry.tools is None for entry in catalog.entries(provider="xai"))


def test_without_the_sdk_extra_the_published_modalities_stand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ai_arch_toolkit.core._providers._meta", None)

    found = ModelCatalog().get("muse-spark-1.3")

    assert found is not None and found.input_modalities is not None
    assert {"video", "audio"} <= found.input_modalities
    assert found.tool_choice_modes is None


def test_no_adapter_reads_the_catalog() -> None:
    """The adapters never read the catalog (D63): none of them imports it."""
    offenders = {
        path.name for path in (SRC / "_providers").glob("*.py") if _reads_the_catalog(path)
    }

    assert offenders == set()


def test_the_catalog_reader_detector_sees_every_way_in(tmp_path: Path) -> None:
    canaries = [
        "from ai_arch_toolkit.core._model_catalog import model_catalog\n",
        "import ai_arch_toolkit.core._model_catalog as catalog\n",
        "from ai_arch_toolkit.core import ModelCatalog\n",
        "def facts():\n    from ai_arch_toolkit.core import model_catalog\n",
    ]
    for index, source in enumerate(canaries):
        path = tmp_path / f"canary_{index}.py"
        path.write_text(source)
        assert _reads_the_catalog(path), source
    clean = tmp_path / "clean.py"
    clean.write_text("from ai_arch_toolkit.core._model_id import lookup\n")
    assert not _reads_the_catalog(clean)


def _reads_the_catalog(path: Path) -> bool:
    """Whether a module imports the catalog's module or one of its names, anywhere in it."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names.extend(alias.name for alias in node.names)
    return any("model_catalog" in name or "ModelCatalog" in name for name in names)


# ---------------------------------------------------------------------------
# The contract (C06c): what the catalog says, the adapter does, model by model
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First.
        b: Second.
    """
    return a + b


_SCHEMA = OutputSchema(
    name="answer",
    schema={
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
)
_SERVER_TOOLS = {
    "web_search": web_search(),
    "code_execution": code_execution(),
    "image_generation": image_generation(model="gpt-image-2.5-flare"),
}
_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_SEEDED = [(entry.provider, entry.model) for entry in model_catalog.entries()]
# The entries that state each reasoning fact; where the adapter's tables are silent, so is the
# catalog, and there is nothing to check. A model no completion reaches takes no budget either.
_WITH_EFFORTS = [
    (e.provider, e.model) for e in model_catalog.entries() if e.thinking_efforts is not None
]
_WITH_MODE = [
    (e.provider, e.model) for e in model_catalog.entries() if e.thinking_mode is not None
]
_WITH_BUDGET = [
    (e.provider, e.model)
    for e in model_catalog.entries()
    if e.thinking_budget is not None and e.streaming is not False
]


def _facts(provider: str, model: str) -> ModelCapabilities:
    found = model_catalog.get(model, provider=provider)
    assert found is not None
    return found


def _accepts(provider: str, model: str, **kwargs: Any) -> bool:
    """Whether the adapter prepares the call; ``RequestError`` is its refusal."""
    adapter = create_provider(model, provider=provider, api_key="test-key")
    kwargs.setdefault("max_tokens", 1024)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            prepare(adapter, [user("hi")], **kwargs)
        except RequestError:
            return False
    return True


@pytest.mark.parametrize(("provider", "model"), _SEEDED)
def test_tools_are_what_the_adapter_sends(provider: str, model: str) -> None:
    facts = _facts(provider, model)
    assert facts.tools is not None

    assert _accepts(provider, model, tools=prepare_tools([add])) is facts.tools


@pytest.mark.parametrize(("provider", "model"), _SEEDED)
def test_each_tool_choice_the_catalog_lists_is_taken_and_no_other(
    provider: str, model: str
) -> None:
    facts = _facts(provider, model)
    assert facts.tool_choice_modes is not None
    if not facts.tools:
        assert facts.tool_choice_modes == frozenset()
        return
    wire = {"auto": "auto", "none": "none", "required": "required", "named": "add"}

    taken = {
        mode
        for mode, choice in wire.items()
        if _accepts(provider, model, tools=prepare_tools([add]), tool_choice=choice)
    }

    assert taken == facts.tool_choice_modes


@pytest.mark.parametrize(("provider", "model"), _SEEDED)
def test_the_server_tools_are_the_types_that_reach_the_wire(provider: str, model: str) -> None:
    facts = _facts(provider, model)
    assert facts.server_tools is not None

    sent = {
        kind
        for kind, server in _SERVER_TOOLS.items()
        if _accepts(provider, model, tools=prepare_tools([server]))
    }

    assert sent == facts.server_tools


@pytest.mark.parametrize(("provider", "model"), _WITH_EFFORTS)
def test_each_effort_the_catalog_lists_is_taken_and_no_other(provider: str, model: str) -> None:
    facts = _facts(provider, model)

    taken = tuple(e for e in _EFFORTS if _accepts(provider, model, thinking_effort=e))

    assert taken == facts.thinking_efforts


@pytest.mark.parametrize(("provider", "model"), _WITH_MODE)
def test_a_model_that_does_not_reason_refuses_thinking(provider: str, model: str) -> None:
    facts = _facts(provider, model)

    assert _accepts(provider, model, thinking=True) is (facts.thinking_mode != "none")


@pytest.mark.parametrize(("provider", "model"), _WITH_BUDGET)
def test_a_thinking_budget_is_taken_where_the_catalog_says_so(provider: str, model: str) -> None:
    facts = _facts(provider, model)
    adapter = create_provider(model, provider=provider, api_key="test-key")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare(adapter, [user("hi")], max_tokens=4096, thinking_budget=2048)

    ignored = any("thinking_budget" in str(warning.message) for warning in caught)
    assert ignored is not facts.thinking_budget


@pytest.mark.parametrize(("provider", "model"), _SEEDED)
def test_structured_output_and_json_mode_are_what_the_adapter_sends(
    provider: str, model: str
) -> None:
    facts = _facts(provider, model)

    for fact, kwargs in (
        (facts.structured_output, {"output_schema": _SCHEMA}),
        (facts.json_mode, {"json_mode": True}),
        (facts.streaming, {}),
    ):
        if fact is not None:
            assert _accepts(provider, model, **kwargs) is fact


_PDF = document(b"%PDF-1.4 catalog contract", name="contract.pdf")


@pytest.mark.parametrize(("provider", "model"), _SEEDED)
def test_a_pdf_reaches_the_wire_where_the_catalog_says_the_model_reads_one(
    provider: str, model: str
) -> None:
    """C06.1 for inputs: the adapter's own list is what its completions carry, and a PDF the
    catalog lists reaches the wire."""
    adapter = create_provider(model, provider=provider, api_key="test-key")
    facts = type(adapter).model_facts(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the xAI adapter warns as it drops the document
        try:
            prepared = prepare(adapter, [user(["Read it.", _PDF])], max_tokens=1024)
        except RequestError:  # an image model no completion reaches
            prepared = None
    sent = prepared is not None and "application/pdf" in repr(prepared.params)

    assert facts.input_modalities is not None
    assert ("pdf" in facts.input_modalities) is sent
    stated = _facts(provider, model).input_modalities
    if stated is not None and "pdf" in stated:
        assert sent


@pytest.mark.parametrize(("provider", "model"), _SEEDED)
def test_a_model_no_completion_reaches_writes_only_images(provider: str, model: str) -> None:
    """C06.1 for outputs: what only draws answers with images, whatever its page says."""
    adapter = type(create_provider(model, provider=provider, api_key="test-key"))
    draws_only = not _accepts(provider, model)

    assert (adapter.model_facts(model).output_modalities == frozenset({"image"})) is draws_only
    if draws_only:
        assert _facts(provider, model).output_modalities == frozenset({"image"})


def test_the_server_tool_vocabulary_is_the_types_core_makes() -> None:
    """The loader's words for ``server_tools`` are the types ``core/_server_tools.py`` builds,
    and the wire contract above tries each of them."""
    tree = ast.parse((SRC / "_server_tools.py").read_text())
    made = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ServerTool"
        for keyword in node.keywords
        if keyword.arg == "type" and isinstance(keyword.value, ast.Constant)
    }

    assert set(SERVER_TOOL_TYPES) == made == set(_SERVER_TOOLS)
