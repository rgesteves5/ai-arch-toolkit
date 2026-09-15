# C06 · Catálogo técnico de modelos no core

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (coordena `server_tools` com C05)
- **Origem:** `docs/internal/agentes-app-toolkit-review.md` L7 (`:332-350`), "Dividido" (`:38`),
  prioridade 6 (`:516`), matriz de probes não é catálogo (`:536`); fora do plano de correcção
  (`docs/internal/toolkit-fix-plan.md:648-652`).
- **Decisões:** por fixar (ver abaixo)

## Problema

- `ModelPricing` só tem tarifas (`core/_pricing.py:25-56`) e casa pelo prefixo mais longo
  (`:107-119`). Nada em `src/` guarda limites ou capacidades: `core/_tokens.py:11-25` só escolhe o
  tokenizer; `max_input_tokens`/`max_output_tokens` são tectos de orçamento (`toolkit/budget/_policy.py:36-37`).
- Os factos são regras por prefixo nos adaptadores, sem fonte nem data: Astra sem tools
  (`_openai.py:447-461`), sampling proibido (`_anthropic.py:54-62`), `thinking_level` só em `gemini-3*`
  (`_gemini.py:256-258`), Grok ignora thinking e recusa server tools (`_xai.py:362-367`, `:408-416`),
  Meta só `tool_choice="auto"` e `web_search` (`_meta.py:260-270`, `:538-542`).
- Matriz velha: corrida completa de 2026-04-28, thinking Anthropic "Not probed"
  (`docs/model-compatibility.md:13-20`, `:120-128`); o inventário não tem Claude 5, GPT-5.6, Gemini 3.5+.
- O prefixo já erra: `pricing.get("claude-fable-5-1").cache_read == 1.0` herda `[claude-fable-5]`
  (`_default_pricing.toml:18-26`); a tarifa oficial é $0.25. E sem factos passam bugs: `thinking=True`
  no `claude-opus-5` envia `{"type": "enabled", ...}` (`_anthropic.py:238-252`), recusado com 400
  desde o Claude 4.7 (https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting).

Metadados em runtime (verificados a 2026-09-15). Anthropic `GET /v1/models/{id}`: `max_input_tokens`,
`max_tokens`, `capabilities` (thinking, effort, `image_input`, `pdf_input`, `structured_outputs`) —
https://platform.claude.com/docs/en/api/models/retrieve, e `ModelInfo` no `anthropic` 0.116.0. Gemini
`models.get`: `inputTokenLimit`, `outputTokenLimit`, `thinking` — https://ai.google.dev/api/models. xAI:
`context_length`, `input_modalities`, `aliases` — https://docs.x.ai/developers/rest-api-reference/inference/models.
OpenAI e Meta: sem limites nem capacidades — https://developers.openai.com/api/reference/resources/models,
https://dev.meta.ai/docs/api-reference/models/schemas.md. Ollama `/api/show`: `capabilities`,
`<arch>.context_length` — https://docs.ollama.com/api-reference/show-model-details. Tools e schema só
nas páginas de modelo e em probes, e variam com a API (Astra: tools só na Responses,
https://developers.openai.com/api/docs/guides/latest-model).

## Objectivo

Um registo no `core`, separado de `pricing`, que responde por `(provider, model)` com os factos de
que o cliente precisa — limites, tools, `tool_choice`, schema, JSON mode, streaming, thinking e
esforços, modalidades, server tools — cada um com `source` e `verified_at`, `None` quando desconhecido,
e overrides da app. Descreve o que funciona através do adaptador. Não escolhe modelos.

## API proposta

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilities:        # factos `T | None = None`; None = desconhecido, nunca "não"
    provider: str               # adaptador ("anthropic"…) ou namespace da app ("ollama")
    model: str
    source: Provenance          # (kind: docs|probe|api|adapter|override, ref, verified_at: date)
    sources: tuple[tuple[str, Provenance], ...] = ()      # por campo; provenance(field), to_dict()
    aliases: tuple[str, ...] = ()
    context_window: int | None = None     # + input_token_limit, output_token_limit
    tools: bool | None = None             # + parallel_tool_calls, structured_output, json_mode,
                                          #   streaming, thinking_budget
    tool_choice_modes: frozenset[Literal["auto", "none", "required", "named"]] | None = None
    thinking_mode: Literal["none", "optional", "always"] | None = None
    thinking_efforts: tuple[str, ...] | None = None
    input_modalities: frozenset[Literal["text", "image", "pdf", "audio", "video"]] | None = None
    server_tools: frozenset[str] | None = None            # valores de ServerTool.type (C05)

class ModelCatalog:  # + entries(provider=None), register(caps, replace=False), unregister, load, reset
    def get(self, model: str, *, provider: str | None = None) -> ModelCapabilities | None: ...
model_catalog = ModelCatalog()  # ModelCatalog(defaults=False) começa vazio
```

```toml
catalog_version = 1             # core/_default_catalog.toml
[openai."gpt-6-astra"]
source = { kind = "docs", ref = "https://developers.openai.com/api/docs/models/gpt-6-astra", verified_at = 2026-09-15 }
context_window = 1_050_000
output_token_limit = 128_000
tools = false
sources.tools = { kind = "adapter", ref = "core/_providers/_openai.py:456-460", verified_at = 2026-09-15 }
```

- `get` sem `provider` infere-o com `_match_provider`; id exacto ou alias; desconhecido → `None`. A
  semente cobre só hosts próprios; a app regista locais num namespace seu, que não colide com `openai`.
- Semente < `load()` < `register()`, campo a campo (só não-`None`); `replace=True` substitui. Loader
  estrito (chave, tipo, vocabulário, `verified_at`) → `ValueError` com entrada e chave. `LLM` e
  adaptadores não o consultam. `thinking_mode="always"`: raciocina sem `thinking=True` (Muse Spark, D13).
- Na app: `[c for c in model_catalog.entries() if c.tools and c.structured_output and
  (c.input_token_limit or 0) >= n]` — `None` exclui; a app decide se pergunta, testa ou descarta.

## Decisões a fixar antes de codificar

1. **O que um facto afirma:** o modelo publicado, ou o que funciona via adaptador. Recomendo: via
   adaptador, com `kind="adapter"` quando o limite é dele, porque é o que o cliente usa — o Astra tem
   tools na Responses mas não aqui; o adaptador xAI descarta imagens (`_xai.py:126-130`).
2. **Casamento de ids:** prefixo; exacto + `aliases`; exacto sem datas. Recomendo: exacto + `aliases`,
   porque capacidades mudam entre snapshots e o prefixo já erra o `claude-fable-5-1`; aliases que
   trocam de modelo (`-latest`) ficam fora da semente.
3. **Estados e limites:** `bool` e fonte por entrada, ou `bool | None` e `sources` por campo; um só
   limite, ou janela, entrada e saída. Recomendo: `None`, `sources` e os três limites como publicados,
   porque as fontes têm datas diferentes e a OpenAI publica janela partilhada (Gemini e Anthropic não).
4. **Fonte de verdade e deriva:** adaptadores lêem o catálogo, ou mantêm as regras e um teste liga os
   dois; probes reescrevem o TOML, ou geram um fragmento revisível. Recomendo: regras + C06c e
   fragmento só local, porque um override ou um TOML errado não pode mudar o wire e probes custam dinheiro.
5. **Descoberta:** na app, ou no core por modelo. Recomendo: `LLM.describe_model()` + `_sync`,
   adiável, para Anthropic, Gemini e xAI (OpenAI e Meta não publicam limites), porque reaproveita
   cliente, chave e regra de hosts (F23); não escreve no singleton; listar modelos e Ollama: app.
6. **Semente:** tudo o que tem preço, ou só a linha actual com página oficial. Recomendo: a linha
   actual, sem retirados nem modelos que o adaptador não chama (`docs/model-compatibility.md:105`),
   porque as 77 entradas de preço já derivam (Fable 5.1 e Gemini 3.8 Flash em falta).

## Sub-tarefas, por ordem

- **C06a** `Provenance`, `ModelCapabilities`, `ModelCatalog`, loader, `model_catalog`, exports.
- **C06b** Semente (decisão 6) com as fontes consultadas; factos de probes com a data do run.
- **C06c** Contrato adaptador → catálogo sobre `_build_sdk_kwargs`; thinking Anthropic 4.7+ como
  `xfail(strict=True)` ligado ao achado. **C06d** Fragmento de catálogo em `scripts/probe_models.py`.
- **C06e** (adiável) `BaseProvider.describe_model()` (`NotImplementedError` por omissão), Anthropic,
  Gemini, xAI, `LLM.describe_model()`; `scripts/check_model_catalog.py` compara a semente com as APIs.
- **C06f** Docs: página nova, diferença face ao `pricing`, relação com a matriz, exemplo do `Auto`.

## Ficheiros

- `src/ai_arch_toolkit/core/_model_catalog.py`, `core/_default_catalog.toml` (novos; `server_tools`
  partilhado com C05); `core/__init__.py`, `ai_arch_toolkit/__init__.py` (partilhados com C01–C05,
  C07, C08). C06e: `core/_providers/_base.py`, `_anthropic.py`, `_gemini.py`, `_xai.py` (partilhados
  com C05), `core/_llm.py` (partilhado com C01 e C05, se tocarem no facade).
- `scripts/probe_models.py`, `scripts/model_probe_notes.md`, `scripts/check_model_catalog.py` (novo).
- `tests/test_model_catalog.py`, `tests/test_model_catalog_data.py` (novos; o segundo partilhado com
  C05), `tests/test_probe_models.py`, `tests/test_core_exports.py`; C06e: `tests/test_llm.py`,
  `tests/test_{anthropic,gemini,xai,openai,meta}_provider.py`.
- `docs/model-catalog.md` (novo), `mkdocs.yml`, `docs/index.md`, `docs/api.md` (partilhado), `docs/llm.md`,
  `docs/pricing.md`, `docs/model-compatibility.md`, `docs/framework-overview.md`, `AGENTS.md`, `README.md`.

## Prova

- C06a: com `claude-fable-5` registado, `get("claude-fable-5-1")` é `None` (o `pricing` herda); alias
  dá a entrada canónica; `provider="ollama", model="gpt-4o"` não muda `get("gpt-4o")`; sobrepor mantém
  a proveniência dos outros campos; `load()` ganha e `reset()` repõe; TOML com `tool = true` ou sem
  `verified_at` → `ValueError` com entrada e chave; o filtro do `Auto` exclui `structured_output=None`.
- C06b: `_match_provider` concorda com cada `provider` em ids e aliases; `verified_at` ≤ hoje; nenhum
  teste falha pela idade dos dados.
- C06c: `tools = false` ⇒ o adaptador levanta com tools; esforços do Astra = `_openai.py:452`; Meta
  `tool_choice_modes == {"auto", "none"}`; `server_tools` = tipos que chegam ao wire.
- C06d: `ProbeResult` sintéticos: `ok` → `true` com o run; `unsupported_capability` → `false`;
  `rate_limit` e `timeout` não afirmam nada. C06e: SDK falso (`SimpleNamespace`) mapeia limites e
  effort; xAI `IMAGE` cortado a `{"text"}`; OpenAI e Meta levantam `NotImplementedError`.
- Só local: `describe_model_sync()` contra as APIs (grátis, com chave); probes pagos a pedido do dono.

## Fora do âmbito

- `Auto`, pontuações, latência, região, modelos permitidos (app); router em `toolkit.routing`; `LLM`
  a validar pedidos pelo catálogo; listar modelos; descoberta local; preços; Bedrock/Vertex; restrições
  condicionais; corrigir o thinking Anthropic, o preço do Fable 5.1 e a matriz; server tools (C05).

## Riscos

- Dados envelhecem como a matriz (`verified_at` obrigatório; a app decide o que é velho); docs
  contraditórias (o exemplo da Models API dá `enabled` no `claude-opus-5`, a tabela diz 400).
- "Via adaptador" muda com C05 (C06c força o catálogo no mesmo diff); `None` lido como `False` ou
  `True`, e prefixos esperados por quem conhece o `pricing` — ambos explicados na doc.

## Registo do dono

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
