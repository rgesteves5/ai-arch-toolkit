# Model Catalog

`model_catalog` answers, for a model, what it takes through the toolkit: its limits, what it
reads and writes, tools and tool choices, structured output, reasoning and its efforts, and the
provider-hosted tools. Every fact carries its source and the day it was read. The catalog
describes; it does not choose a model, and nothing in the toolkit reads it.

```python
from ai_arch_toolkit import model_catalog

caps = model_catalog.get("claude-opus-5-5")
caps.context_window          # 1000000
caps.tool_choice_modes       # frozenset({"auto", "none"}): Opus 5.5 refuses a forced choice
caps.thinking_efforts        # ("low", "medium", "high", "xhigh", "max")
caps.provenance("context_window")
# Provenance(kind="docs", ref="https://platform.claude.com/docs/en/models/opus-5-5/overview",
#            verified_at=datetime.date(2026, 10, 8))
caps.provenance("tool_choice_modes").kind   # "adapter"
```

## Unknown is `None`, never "no"

Each fact is `None` when nothing says it. `None` never means the model lacks it, so a filter
that needs a fact must exclude a model that does not have it:

```python
picks = [
    c
    for c in model_catalog.entries()
    if c.tools and c.structured_output and (c.input_token_limit or c.context_window or 0) >= 200_000
]
```

`c.tools` is falsy for `None` and `False` alike: the filter keeps only models known to take
tools. Whether to ask, probe, or drop a model with an unknown fact is the app's call.

A set (`input_modalities`, `output_modalities`, `tool_choice_modes`, `server_tools`) lists what
its source states. For a published modality list, a modality the provider's page does not name
is not stated (Gemini 2.5 Flash's page lists no PDF, for example). A modality the adapter does
not carry is left out with `kind="adapter"` (see below): it does not reach the model, or does
not come back.

The three limits are kept as each provider publishes them. OpenAI gives a context window shared
by input and output, and for some models a maximum input; Gemini gives an input and an output
limit and no context window; Anthropic, xAI and Meta give a context window, and Anthropic a
maximum output.

## The facts

| Field | Meaning | From |
|---|---|---|
| `context_window` | Input and output tokens together | the provider's page |
| `input_token_limit` | Input tokens | the provider's page |
| `output_token_limit` | Output tokens, reasoning included | the provider's page |
| `input_modalities` | What the model reads through the adapter: `text`, `image`, `pdf`, `audio`, `video` | the page, narrowed by the adapter |
| `output_modalities` | What it writes through the adapter: `text`, `image` | the page, narrowed by the adapter |
| `tools` | It calls function tools | the adapter |
| `tool_choice_modes` | `auto`, `none`, `required`, `named` (a tool's name) | the adapter |
| `parallel_tool_calls` | It calls several tools in one turn | the app (no shipped value) |
| `structured_output` | It takes `output_schema` | the adapter |
| `json_mode` | It takes `json_mode` | the adapter |
| `streaming` | Its answers stream | the adapter |
| `thinking_mode` | `none` (it does not reason), `optional` (it reasons when asked), `always` (it reasons unasked; `"none"` among its efforts stops it) | the adapter |
| `thinking_efforts` | The `thinking_effort` values it takes, weakest first | the adapter |
| `thinking_budget` | It takes `thinking_budget` in tokens | the adapter |
| `server_tools` | The server tools it runs, by `ServerTool.type` (`web_search`, `code_execution`, `image_generation`) | the adapter |

Each entry also has `provider`, `model` (the provider's own id), `aliases`, and `sources`.
`provenance(field)` gives one fact's `Provenance(kind, ref, verified_at)`; `to_dict()` gives the
entry as plain data, ready for JSON.

## What works through the adapter

A fact says what works through the toolkit's adapter, not what the model can do elsewhere
(D63). Where the limit is the adapter's, the source says so with `kind="adapter"`:

- the xAI adapter sends no server tool yet, so every Grok model has `server_tools ==
  frozenset()`, though xAI runs web search;
- Meta's Muse Spark reads video and audio, but no content part of the toolkit carries them:
  its `input_modalities` is `{"text", "image", "pdf"}`, with `kind="adapter"`;
- the image models take no tools (`tools` is `False`). Gemini's and Meta's also complete, so a
  PDF reaches them (Gemini 3.1 Flash Image: `{"text", "image", "pdf"}`, video left out);
  OpenAI's and xAI's only draw, from a prompt and input images;
- GPT Image 1.5's page says it writes images and text, but the Images API, the adapter's only
  way to it, answers with images: its `output_modalities` is `{"image"}`, with
  `kind="adapter"`.

A list the provider states, on its page (`kind="docs"`) or from its models endpoint
(`kind="api"`), is narrowed this way; a list the app observed (a probe's, an override) never
is.

These facts come from each adapter's own tables, through a pure classmethod,
`BaseProvider.model_facts(model)`. The adapters never read the catalog: what limits a call is
the adapter, with the catalog or without it, and the catalog only tells it and whose limit it
is. Fixing an adapter's table fixes the catalog; overriding the catalog changes nothing on the
wire. Without the adapter's SDK extra (`pip install ai-arch-toolkit[xai]`, …), or with an
installed SDK that fails to import, its facts are `None`, and the published modalities are not
narrowed.

Some facts the adapter's tables do not hold stay `None`: whether an adaptive Claude model
thinks unasked (`thinking_mode`), and whether a Gemini 2.5 model that can stop thinking does
so by default.

## Which models

The shipped seed (`core/_default_catalog.toml`) is the current line with an official page: the
30 models of the live probe inventory still served, and the image models (GPT Image, Gemini
image, Grok Imagine, Muse Image). It holds only what the providers publish, limits and
modalities, each with its page and the day it was read; the rest comes from the adapters.

A model the catalog does not know gives `None`, also when the adapter would run it with its
provider's current rules. Add it with `register()` or `load()`; the adapter's facts join it.

## Matching ids

An id finds its entry by the entry's own id, an alias, or a dated snapshot of either
(`claude-haiku-4-5-20251001` → `claude-haiku-4-5`), as everywhere in `core`. There is no
family prefix: `claude-fable-5-1` is not `claude-fable-5`. A `-latest` pointer, which moves
between models, is never an alias. The keys are the provider's own ids: xAI calls
`grok-4.20-reasoning` `grok-4.20-0309-reasoning`, and `get("grok-4.20-reasoning")` returns
that entry. `register()`, `load()` and `unregister()` name an entry the same way: a fact
registered for `grok-4.20-reasoning`, or for `claude-haiku-4-5-20251001`, lands on the entry
`get()` finds for it.

`get(model)` asks the provider the id routes to (`claude-` → `anthropic`, as `LLM` routes).
Models on a local or compatible server go in a namespace of the app's, asked by name, which
never shadows the provider's own entry:

```python
from ai_arch_toolkit import ModelCapabilities, model_catalog

model_catalog.register(
    ModelCapabilities(provider="ollama", model="llama3.1", context_window=131_072, tools=True)
)
model_catalog.get("llama3.1", provider="ollama")
```

## Overrides

Layers, field by field, each fact from the highest layer that sets it:

1. the seed and the adapters' facts;
2. what `load(path)` read;
3. what `register(capabilities)` said.

```python
from datetime import date
from ai_arch_toolkit import ModelCapabilities, Provenance, model_catalog

ours = Provenance(kind="override", ref="eval run 41", verified_at=date(2026, 10, 8))
model_catalog.register(
    ModelCapabilities(
        provider="openai",
        model="gpt-5.4-mini",
        parallel_tool_calls=True,
        sources=(("parallel_tool_calls", ours),),
    )
)
```

A fact registered without a source gets `Provenance(kind="override", ref=
"ModelCatalog.register")`. `register(..., replace=True)` makes the entry the whole of what is
known: nothing below it shows through. `unregister(model)` forgets a model in every layer and
`reset()` returns to the seed. `ModelCatalog(defaults=False)` starts empty, with no seed and no
adapter facts. An alias that already names another model of the provider raises `ValueError`.
`ModelCapabilities` and `Provenance` check their facts as `load()` does: a wrong type or a word
outside a field's vocabulary raises `ValueError`, and a set may be given as any collection.

A catalog is safe to share between threads. A change builds the next version of the catalog,
one change at a time; a read takes the current version and never waits, and what it builds
belongs to that version, so a read that overlaps a change never keeps a stale entry.

## The file format

`load()` reads TOML in the seed's shape, and is strict: a file that is not TOML, an unknown key,
a wrong type, a word outside a field's vocabulary (`server_tools` takes the `ServerTool.type`
values the core builds), or a fact without a dated source raises `ValueError` naming the file,
the entry and the key, and nothing of the file is kept. `catalog_version` is the integer `1`.
An effort listed twice is kept once.

```toml
catalog_version = 1

[ollama."llama3.1"]
aliases = ["llama3.1:8b"]
context_window = 131_072
input_modalities = ["text"]
tools = true
tool_choice_modes = ["auto", "none"]
source = { kind = "probe", ref = "local run 3", verified_at = 2026-10-08 }
sources.context_window = { kind = "docs", ref = "https://ollama.com/library/llama3.1", verified_at = 2026-10-01 }
```

`source` is the default for every fact of the entry; `sources.<field>` gives one its own. A
source has a `kind` (`docs`, `probe`, `api`, `adapter`, `override`), a `ref`, and a
`verified_at` TOML date.

## Probe runs

`scripts/probe_models.py` writes, next to each run's report, `<run_id>.catalog.toml`: what the
run proved, as `kind = "probe"` facts dated the day of the run. A passed scenario states its
fact true, one the adapter refused (its `prepare`, before sending) states it false, and any
other outcome (a rate limit, a timeout) states nothing. A provider's error that only reads as
unsupported may be a framework bug as well, so it states nothing either: the fragment lists it
in a comment, for review against the report. The fragment is keyed by the inventory's ids,
aliases included, and is local: compare it with the adapter's facts, or load it with
`model_catalog.load()`. It never goes into the seed.

## Next to pricing and the compatibility matrix

- [Pricing](pricing.md) knows rates only; the catalog knows no prices. Both match ids the same
  way, but a price may also cover a family of local models (`match="prefix"`), and the catalog
  never does.
- [Model Compatibility](model-compatibility.md) records the live probe runs, model by model,
  with their dates. The catalog's adapter facts are the adapter's rules today; a probe run
  checks them.
- The catalog does not validate requests, choose models, or route: an app that picks a model
  per task (an `Auto` mode) filters `entries()` as above and decides what an unknown fact
  means.
