# C06 · Catálogo técnico de modelos no core

- **Dono:** Claude (um agente: C06a a C06d e C06f, 2026-10-08) · **Estado:** done · **Depende de:** nada (coordena `server_tools` com C05)
- **Origem:** `docs/internal/agentes-app-toolkit-review.md` L7 (`:332-350`), "Dividido" (`:38`),
  prioridade 6 (`:516`), matriz de probes não é catálogo (`:536`); fora do plano de correcção
  (`docs/internal/toolkit-fix-plan.md:648-652`).
- **Decisões:** fixadas pelo dono a 2026-10-08, D63

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

**Fixadas a 2026-10-08 (D63), todas na opção recomendada; a lista abaixo fica como estava.**

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
- **C06c** Contrato adaptador → catálogo sobre o `prepare()` de cada adaptador e as tabelas de
  perfis (`core/_model_id.py`); o thinking da Anthropic 4.7+ já segue as regras de cada modelo
  desde a R02, sem `xfail`. **C06d** Fragmento de catálogo em `scripts/probe_models.py`.
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

- **Estado:** done. Feito por agentes em worktrees, juntado e revisto pelo coordenador; uma revisão independente por parte, e as correcções dela (secção "Correcções da revisão").
- **Gate final, no checkout principal com todas as fichas da vaga:** 8195 passed, 118 skipped (os ao vivo e os 59 do nanope à espera); ruff, formatação, pyright e `uv lock --check` limpos (2026-10-08).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed, Fixed).

- **Worktree:** `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-a790d40e15e9f3fb0`
  (base `main` @ `601e38c`). Nada commitado.
- **Estado:** review. A C06e (`describe_model`) fica fora, como pedido.

### Nota de desenho

**Interface pública** (`ai_arch_toolkit` e `ai_arch_toolkit.core`): `Provenance`,
`ModelCapabilities`, `ModelCatalog`, `model_catalog`.

- `Provenance(kind, ref, verified_at)`, com `kind` em `docs | probe | api | adapter | override`.
  O `verified_at` é `date | None`: `None` só para um facto do adaptador (vale o código instalado)
  ou um override sem dia.
- `ModelCapabilities` (frozen, slots, kw_only): `provider`, `model`, `aliases` e os factos, todos
  `T | None = None`:
  - os três limites: `context_window`, `input_token_limit`, `output_token_limit`;
  - as modalidades: `input_modalities` e `output_modalities` (C06.7);
  - `tools`, `tool_choice_modes`, `parallel_tool_calls`, `structured_output`, `json_mode`,
    `streaming`;
  - `thinking_mode`, `thinking_efforts`, `thinking_budget`, `server_tools`;
  - `sources`, a proveniência por campo (C06.3), com `provenance(field)` e `to_dict()`.
- `ModelCatalog(defaults=True)` tem `get(model, provider=)`, `entries(provider=)`,
  `register(caps, replace=)`, `unregister`, `load` e `reset`.

**Casamento (C06.2):** `get` infere o fornecedor com `_match_provider` e procura por
`_model_id.lookup`: o id da entrada, um alias, ou um snapshot datado de um deles. Nunca casa por
família.

**Camadas, campo a campo:**

1. a semente e os factos do adaptador;
2. o que o `load()` leu;
3. o que o `register()` disse.

O adaptador só preenche o que nenhuma camada pôs. O `replace=True` esconde tudo o que está
abaixo, o adaptador incluído. O `ModelCatalog(defaults=False)` começa vazio, sem semente e sem
adaptador. Um facto registado sem fonte fica `kind="override"`.

**Factos do adaptador (C06.4):** cada adaptador ganhou um classmethod puro,
`model_facts(model) -> AdapterFacts`, que lê só as suas tabelas.

- `AdapterFacts` e o vocabulário (`ThinkingMode`, `ToolChoiceMode`, `InputModality`,
  `EFFORT_ORDER`) vivem em `_providers/_base.py`. O catálogo importa-os de lá e nenhum
  adaptador importa o catálogo (teste AST).
- O catálogo importa o adaptador só quando precisa de um facto. Se o import falha (falta o extra
  do SDK), os factos do adaptador ficam `None`. O `import ai_arch_toolkit` continua sem carregar
  nenhum SDK; o catálogo custa ~6 ms de import.
- Pequenas extracções, sem mudar o fio:
  - `_profile_of`, `_rules_of`, `_efforts_of` e `_image_profile_of`, para o método de instância e
    o classmethod lerem a mesma tabela;
  - no Anthropic, o if de `_server_tool` passou a tabela `_SERVER_TOOLS`, com os mesmos dois
    dicts, para os `server_tools` saírem dela.
- `ResponsesProfile.facts()` dá o que a OpenAI e a Meta partilham. Cada adaptador junta o
  raciocínio do seu modelo.

**C06.1, "através do adaptador":**

- todos os factos vindos do adaptador levam `kind="adapter"`;
- as `input_modalities` publicadas (só as de `kind="docs"`, venham da semente ou de um `load()`)
  são cortadas ao que o adaptador envia, com `kind="adapter"` quando o corte muda algo:
  - nenhuma parte de conteúdo do toolkit leva áudio nem vídeo;
  - o xAI larga documentos;
  - uma geração de imagem só aceita texto e imagens (`image_prompt`).
- Um facto observado (probe, override) nunca é cortado.
- Exemplo real: a Muse Spark 1.3 publica vídeo e áudio, e o catálogo diz `{"text", "image",
  "pdf"}` com `kind="adapter"`.

**`thinking_mode`:**

- `"none"`: não raciocina, e `thinking` é recusado;
- `"optional"`: raciocina quando se pede;
- `"always"`: raciocina sem pedir, e um `"none"` nos esforços pára-o.

Onde a tabela do adaptador não diz se o modelo pensa sem pedir, o facto fica `None`: os Claude
adaptativos e os Gemini 2.5 que podem desligar o pensamento.

**Semente (C06.6, C06.7):**

- `core/_default_catalog.toml`, 43 entradas: os 30 modelos do inventário que ainda servem e os
  13 modelos de imagem. Ficaram de fora:
  - `grok-4-1-fast-reasoning` e `grok-4-1-fast-non-reasoning`, retirados a 2026-05-15;
  - `gemini-3.1-flash-lite-preview`, desligado a 2026-05-25;
  - `gemini-3.1-flash-live-preview`, só pela Live API, que nenhum adaptador chama;
  - `claude-sonnet-4-0`, retirado a 2026-06-15.
- O TOML guarda só limites e modalidades publicados (um teste impõe as chaves), cada facto com
  página e `verified_at = 2026-10-08`.
- No xAI as chaves são os nomes do fornecedor (`grok-4.20-0309-reasoning`, …), e os ids do
  inventário são aliases. Os ponteiros `-latest` ficam fora.
- O `chatgpt-image-latest` também fica fora: é um ponteiro, e está deprecado.

**Loader estrito:**

- `catalog_version = 1`;
- chaves conhecidas, tipos (int positivo, bool exacto) e vocabulário;
- `verified_at` tem de ser uma data TOML (uma datetime é recusada);
- cada facto tem fonte (`source` da entrada, ou `sources.<campo>`);
- `ValueError` com `ficheiro: [provider."model"] campo`, e nada do ficheiro fica se uma entrada
  falha;
- um id ou alias que já nomeia outro modelo do fornecedor é recusado.

**C06d, decisão: mantém-se, como fragmento local.**

- O `scripts/probe_models.py` escreve `<run_id>.catalog.toml` ao lado do JSONL e do Markdown,
  com factos `kind = "probe"` datados do dia do run:
  - um cenário que passou dá `true`;
  - um `unsupported_capability` dá `false`;
  - qualquer outro desfecho não afirma nada (rate limit, timeout, resposta errada).
- Cenários mapeados: `tools_loop` → `tools`, `structured` → `structured_output`,
  `json_mode` → `json_mode`, `stream` → `streaming`.
- Porque cabe na D63:
  - nunca entra na semente, que só guarda factos publicados;
  - é uma camada `load()` da app, ou serve para o dono comparar com os factos do adaptador;
  - uma discordância aponta uma tabela do adaptador a corrigir, e corrigi-la corrige o catálogo.
- A alternativa rejeitada pela D63 era outra: o TOML a repetir os factos do adaptador, ligado por
  um teste.

### Fontes consultadas: URL e facto

Todas lidas a 2026-10-08, só páginas oficiais, sem chamadas a APIs.

**Anthropic**

- Os limites vêm de cada página de modelo:
  - https://platform.claude.com/docs/en/models/opus-5-5/overview, opus-4-7, opus-4-6 e
    sonnet-4-6: janela de 1M e saída de 128K;
  - opus-4-5, sonnet-4-5 e haiku-4-5: janela de 200K e saída de 64K.
  - Todas dizem "Text and images → text".
- https://platform.claude.com/docs/en/build-with-claude/pdf-support: "All active models support
  PDF processing".
- https://platform.claude.com/docs/en/about-claude/model-deprecations: a Sonnet 4.5 foi deprecada
  a 2026-09-30 e retira-se a 2026-11-30. Por isso o PDF não fica afirmado para ela.
- https://platform.claude.com/docs/en/models/overview: a Opus 5.5 e a Fable 5.1 têm "Adaptive
  (always on)". Existem ainda a Sonnet 5.5 e a Haiku 5.5, fora do inventário.
- https://platform.claude.com/docs/en/build-with-claude/thinking e .../effort: os esforços por
  modelo (a 4.6 sem `xhigh`). Batem com `_PROFILES`.

**OpenAI** (https://developers.openai.com/api/docs/models/<id>)

- `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`: janela de 1.050.000, entrada máxima de
  922.000, saída de 128.000, "Input modalities: text, image".
- `gpt-5.5` e `gpt-5.4`: janela de 1.050.000 e saída de 128.000. Não dão entrada máxima.
- `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`: janela de 400.000,
  entrada de 272.000, saída de 128.000.
- `gpt-4.1`: janela de 1.047.576 e saída de 32.768. `o3`: janela de 200.000 e saída de 100.000.
- GPT Image (`gpt-image-2.5-sunburst`, `-2.5-flare`, `-2`, `-1.5`, `-1`, `-1-mini`): sem limites
  de tokens publicados. Entrada "text, image"; saída "image" (o 1.5 dá "image, text").
- https://developers.openai.com/api/docs/deprecations, datas de fim:
  - o gpt-image-1 a 2026-10-23;
  - o gpt-image-1.5, o gpt-image-1-mini e o chatgpt-image-latest a 2026-12-01;
  - os snapshots do gpt-5, do gpt-5-mini, do gpt-5-nano e do o3 a 2026-12-11;
  - o gpt-5.4-nano a 2027-04-01.
- Os esforços das páginas batem com `_MODELS`. O Astra lista low–max, e o 6.1 Sol diz que não
  aceita `none` nem `minimal`.

**Gemini** (https://ai.google.dev/gemini-api/docs/models/<id>)

- `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`,
  `gemini-2.5-flash-lite`: entrada de 1.048.576 e saída de 65.536.
- As entradas são texto, imagem, vídeo, áudio e PDF, menos no 2.5 Flash, que não lista PDF.
- `gemini-3.1-flash-image`: entrada de 131.072, saída de 32.768; lê texto, imagem, vídeo e PDF;
  escreve imagem e texto.
- `gemini-3.1-flash-lite-image`: entrada de 65.536 e saída de 4.096.
- `gemini-3-pro-image`: entrada de 65.536 e saída de 32.768; lê só imagem e texto.
- Os modelos de imagem não aceitam function calling.
- https://ai.google.dev/gemini-api/docs/deprecations: os ids preview não têm data de fim, e não
  há id estável do 3.1 Pro nem do 3 Flash.

**xAI** (https://docs.x.ai/developers/models/<id>)

- `grok-4.7`: janela de 500.000; "text, image → text"; esforços low–xhigh; "Reasoning cannot
  be disabled".
- `grok-4.20-reasoning`, `-non-reasoning` e `-multi-agent`: janela de 1.000.000. O nome do modelo
  é o `-0309` respectivo, e os ids do inventário são aliases.
- Nenhuma página publica a saída máxima.
- https://docs.x.ai/developers/model-capabilities/text/multi-agent: "Client-side tools (function
  calling) … are not currently supported" e "max_tokens … not currently supported".
- Grok Imagine (`-2.0`, `grok-imagine-image`, `-quality`): "text, image → image". O `-quality`
  retira-se a 2026-11-02
  (https://docs.x.ai/developers/migration/imagine-image-quality-nov-2).

**Meta** (https://dev.meta.ai/docs/models)

- `muse-spark-1.3`: janela de 1.048.576; lê "Text, image, video, audio*, PDF" (o áudio "not fully
  supported" na 1.3); escreve texto. Não publica saída máxima.
- `muse-image-1.0`: lê texto e imagem, escreve imagem.
- https://dev.meta.ai/docs/tool-calling: só `auto` na Responses.
  https://dev.meta.ai/docs/reasoning: o `none` dá 400, e o `max` só existe na 1.3 standard.
  Batem com o adaptador.

### Mudanças em ficheiros partilhados

**Novos**

- `src/ai_arch_toolkit/core/_model_catalog.py` (584 linhas);
- `src/ai_arch_toolkit/core/_default_catalog.toml` (364 linhas, 43 entradas);
- `docs/model-catalog.md`;
- `tests/test_model_catalog.py` e `tests/test_model_catalog_data.py`.

**Exports**

- `core/__init__.py` e `ai_arch_toolkit/__init__.py` ganham `ModelCapabilities`, `ModelCatalog`,
  `Provenance` e `model_catalog` (import e `__all__`). A C01–C05, a C07 e a C08 também tocam
  aqui: juntar por ordem alfabética.

**Adaptadores** (só o classmethod e extracções; o fio não muda)

- `_providers/_base.py`: `AdapterFacts`, o vocabulário, `ordered_efforts`, `DRAWS_ONLY`,
  `DRAWS_WITHOUT_TOOLS` e `BaseProvider.model_facts`, que por omissão não diz nada.
- `_anthropic.py`: `_SERVER_TOOLS` passou a tabela, mais `_profile_of` e `model_facts`.
- `_openai.py`: `_Model.thinking_mode()`, `_rules_of` e `model_facts`.
- `_responses.py`: `ResponsesProfile.facts()`.
- `_meta.py`: `_image_rules_of`, `_efforts_of` e `model_facts`.
- `_gemini.py`: `_Profile.facts()`, `_profile_of`, `_image_profile_of` e `model_facts`.
- `_xai.py`: `_profile_of` e `model_facts`.
- A C05 e a C06e mexem nos mesmos adaptadores. Quando a C05 mandar server tools do xAI, o
  `server_tools=frozenset()` do `XAIProvider.model_facts` tem de mudar no mesmo diff. Os testes
  de contrato falham se não mudar.

**Scripts**

- `scripts/probe_models.py`: `CATALOG_FACTS`, `catalog_fragment` e o terceiro ficheiro do run.
- `scripts/model_probe_notes.md`: entrada de 2026-10-08.

**Testes existentes**

- `tests/test_core_exports.py`: +9 testes.
- `tests/test_probe_models.py`: +2 testes.

**Docs**

- `docs/model-catalog.md`, a página nova.
- `mkdocs.yml`: a entrada no nav.
- `docs/index.md`: uma linha de feature e um link.
- `docs/api.md`: uma linha na tabela do core.
- `docs/llm.md`: um parágrafo na contagem de tokens e um "see also".
- `docs/pricing.md`: a diferença face ao catálogo.
- `docs/model-compatibility.md`: o ponteiro.
- `docs/framework-overview.md`: o módulo, a árvore e a feature.
- `AGENTS.md` e `README.md`: uma linha cada.

**Sem mudanças:** `pyproject.toml` (o wheel já leva o TOML; confirmado com `uv build` na
scratchpad) e `tests/quality_baseline.json` (nenhuma dívida nova nem paga).

### Prova: testes ligados à ficha

**C06a**

| Item da ficha | Teste |
|---|---|
| `claude-fable-5` registado ⇒ `get("claude-fable-5-1")` é `None` | `test_an_id_finds_its_entry_and_its_dated_snapshots_but_never_a_familys` |
| um alias dá a entrada canónica | `test_an_alias_gives_the_canonical_entry` |
| `ollama`/`gpt-4o` não muda `get("gpt-4o")` | `test_an_app_namespace_never_shadows_the_providers_own_entry` |
| sobrepor mantém a proveniência dos outros campos | `test_an_override_keeps_the_provenance_of_the_other_fields` |
| `load()` ganha e `reset()` repõe | `test_load_wins_over_the_seed_register_over_load_and_reset_restores` |
| TOML com `tool = true` ou sem `verified_at` ⇒ `ValueError` com entrada e chave | `test_the_loader_names_the_entry_and_the_key_it_refuses`, 14 casos: também tipos, vocabulário, fonte em falta, datetime, `sources` órfão, `aliases` |
| o filtro do `Auto` exclui `structured_output=None` | `test_the_auto_filter_excludes_an_unknown_fact` |

Também da C06a:

- `test_replace_drops_every_fact_below_it`;
- `test_unregister_removes_an_entry_until_reset`;
- `test_a_catalog_without_defaults_starts_empty`;
- `test_an_alias_that_names_another_entry_is_refused`;
- `test_the_loader_reads_every_fact_with_its_source`;
- `test_to_dict_is_plain_data_with_each_fields_source`;
- `test_the_model_catalog_is_exported` e `test_the_model_catalog_is_apart_from_pricing`.

**C06b** (`test_model_catalog_data.py`)

| Item da ficha | Teste |
|---|---|
| `_match_provider` concorda com cada `provider` em ids e aliases | `test_the_routing_agrees_with_every_id_and_alias` |
| `verified_at` ≤ hoje, e nenhum teste falha pela idade dos dados | `test_every_seed_fact_comes_from_its_providers_docs_on_a_past_day`: só o limite de cima, e cada ref é https no domínio de docs do fornecedor |

Também da C06b:

- `test_the_seed_is_the_current_line_and_the_image_models`: os 43 pares (provider, model), 30
  deles de chat;
- `test_the_seed_holds_only_published_limits_and_modalities`: C06.4 no TOML;
- `test_every_entry_says_what_it_outputs`: C06.7;
- `test_an_image_model_takes_no_tools`;
- `test_no_alias_moves_between_models`;
- `test_the_inventory_ids_find_their_entries`;
- `test_the_limits_fit_inside_each_other`;
- `test_a_gemini_image_models_output_limit_is_the_one_its_adapter_reserves`: TOML = D61.

**C06c**: os contratos correm sobre as 43 entradas e chamam o `prepare()` real de cada
adaptador.

| Item da ficha | Teste |
|---|---|
| `tools = false` ⇒ o adaptador levanta com tools | `test_tools_are_what_the_adapter_sends`, nos dois sentidos |
| esforços do Astra = `_openai._MODELS` | `test_astras_efforts_are_its_row_in_the_adapters_table` |
| Meta `tool_choice_modes == {"auto", "none"}` | `test_meta_takes_only_auto_and_none` |
| `server_tools` = tipos que chegam ao fio | `test_the_server_tools_are_the_types_that_reach_the_wire` |

Os contratos a mais:

- `test_each_tool_choice_the_catalog_lists_is_taken_and_no_other`;
- `test_each_effort_the_catalog_lists_is_taken_and_no_other`;
- `test_a_model_that_does_not_reason_refuses_thinking`;
- `test_a_thinking_budget_is_taken_where_the_catalog_says_so`;
- `test_structured_output_and_json_mode_are_what_the_adapter_sends`.

E os da D63:

- `test_the_adapters_facts_come_with_their_provenance`;
- `test_without_the_sdk_extra_the_adapters_facts_are_unknown`;
- `test_without_the_sdk_extra_the_published_modalities_stand`;
- `test_a_published_modality_the_adapter_does_not_send_is_left_out`;
- `test_the_adapter_narrows_what_a_page_publishes_not_what_a_run_saw`;
- `test_no_adapter_reads_the_catalog`, com o canário
  `test_the_catalog_reader_detector_sees_every_way_in`.

**C06d** (`tests/test_probe_models.py`)

| Item da ficha | Teste |
|---|---|
| `ok` → `true` com o run; `unsupported_capability` → `false`; `rate_limit` e `timeout` não afirmam nada | `test_the_catalog_fragment_states_only_what_a_run_proved`: o fragmento carrega num `ModelCatalog` como `kind="probe"` com a data do run |

Também `test_an_empty_run_gives_a_fragment_that_loads_empty`.

**Falha pela razão certa.** Antes do código, a colecção falhava com
`ImportError: cannot import name 'ModelCapabilities'`. Para provar os contratos, troquei dois
factos à mão:

- o `tool_choice_modes` da Meta para os quatro modos;
- os `server_tools` do xAI para `{"web_search"}`.

Os dois testes respectivos falharam.

### Ao vivo, pelo dono

Grátis, opcional. Confirma os limites publicados pelas APIs de metadados, que não custam tokens.
É o que a C06e automatizará.

```bash
set -a && source .env && set +a
curl -s https://api.anthropic.com/v1/models/claude-opus-5-5 -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"   # max_input_tokens, max_tokens
curl -s "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash?key=$GOOGLE_API_KEY"   # inputTokenLimit, outputTokenLimit
uv sync --extra dev --extra docs && uv run mkdocs build --strict   # links da página nova
```

Os probes pagos não são precisos para esta vaga. O fragmento do C06d só aparece no próximo run
do dono.

### Bloqueios e achados

Não houve bloqueios.

**Contradições e lacunas nas docs oficiais face às tabelas:**

1. **Anthropic, thinking sempre ligado.** As docs dizem "Adaptive (always on)" para a Opus 5.5 e
   a Fable 5.1, e "off until you set adaptive" para a Opus 4.8, 4.7 e 4.6 e a Sonnet 4.6. O
   `_Profile` não guarda isto, por isso o `thinking_mode` destes modelos fica `None`. Um campo
   no perfil, que não muda o fio, deixaria o catálogo dizê-lo. Fica para uma tarefa futura.
2. **Anthropic 4.5 com extended thinking.** As docs limitam o `tool_choice` a `auto`/`none`
   quando há thinking manual. O adaptador deixa forçar a tool com orçamento, e daí vem um 400 do
   fornecedor. É uma restrição condicional, fora do catálogo, e talvez seja um achado para o
   `FINDINGS.md`.
3. **xAI multi-agent.** A página do modelo diz "Function calling: Yes", e a página de
   capacidades diz que as client-side tools não são suportadas. O adaptador segue a segunda
   (`tools=False`).
4. **Meta Muse Image.** A página diz "an edit needs 1–10 items", e o adaptador tem
   `max_inputs=1`. Segundo a I03 ao vivo, o limite vem de o SDK nomear as imagens `image[]`. Vale
   rever.
5. **Meta, docs contraditórias.** As páginas de protocolo omitem o `max` dos esforços, e o
   modelo e o schema incluem-no. O adaptador aceita `max` na 1.3 standard.
6. **Gemini 2.5 Flash** não lista PDF nas entradas, ao contrário do 2.5 Pro e do Flash-Lite.
   Registei como publicado; parece omissão da página.
7. **OpenAI, endpoints dos modelos de imagem.** A página do gpt-image-1 marca a Responses como
   "Supported", as outras dizem "Not supported", e o guia manda usar os 2.5 na tool da Responses.
   Não afecta o catálogo.

**Achados fora do âmbito, para o coordenador:**

8. **Modelos novos sem preço nem inventário:** `claude-sonnet-5-5` e `claude-haiku-5-5` estão no
   overview da Anthropic (retiram-se não antes de 2027-09-28 e 2027-10-07), mas não no
   `_default_pricing.toml`, nem no inventário, nem na semente. Pela D16, falham sob um meter.
9. **Fins de vida próximos na semente:**
   - gpt-image-1 a 2026-10-23;
   - grok-imagine-image-quality a 2026-11-02;
   - claude-sonnet-4-5 a 2026-11-30;
   - gpt-image-1.5 e gpt-image-1-mini a 2026-12-01;
   - os snapshots do gpt-5, do gpt-5-mini, do gpt-5-nano e do o3 a 2026-12-11;
   - claude-haiku-4-5 "not sooner than" 2026-10-15, sem deprecação anunciada.

   A semente e o inventário precisam de poda quando estas datas passarem.
10. **Preços do Gemini:** o `_default_pricing.toml` usa `gemini-3.1-pro` e `gemini-3-flash` como
    chaves, com o preview como alias, mas as docs dizem que não há id estável. Não está errado
    (os aliases cobrem), mas a chave não existe na API.
11. **Preço do chatgpt-image-latest:** o `_default_pricing.toml` dá-lhe o preço do gpt-image-1.5.
    A página só diz "points to the Image snapshot previously used in ChatGPT", e está deprecado.
12. **README:** a matriz do `README.md` ainda dá o xAI sem imagens ("Multimodal (image) —"), mas o
    adaptador envia imagens desde a A13.

**Desvios à ficha, todos pela D63:**

- não há campo `source` ao nível da entrada em `ModelCapabilities`, porque as `sources` são por
  campo (C06.3); o TOML mantém `source` como a fonte por omissão da entrada;
- o `output_modalities` é novo (C06.7);
- o limite de saída dos modelos de imagem do Gemini está no TOML (publicado) e também no
  adaptador (D61, para o metering). Um teste mantém-nos iguais. Não é o TOML a repetir um facto
  do adaptador, é o adaptador a guardar um limite publicado para a reserva;
- os ids do xAI na semente são os do fornecedor (`-0309`), e não os do inventário, que ficaram
  como aliases.

### Correcções da revisão

Feitas no checkout principal, a 2026-10-08, sem commit. Para cada achado escrevi primeiro o
teste e vi-o falhar pela razão do achado; só depois mexi no código. O fio não mudou: o
`scratchpad/c06fix/wire_check.py` compara os adaptadores do HEAD com os da árvore através do
`tests/provider_calls.prepare`, em 672 pedidos (com uma mensagem com PDF e imagem, e com os
modelos de imagem), e dá 0 diferenças. O `review/wire_diff.py` do revisor também dá 0 em
2.325 pedidos.

**1. (Medium) `load`, `register` e `unregister` resolvem um alias.**

- **Mudança:** o `_claim` deu lugar ao `_canonical`. Põe cada entrada sob o id da entrada que o
  `get` encontraria para ela (o id, um alias ou um snapshot datado de um deles, pelo
  `_model_id.lookup`) e só depois junta. Um alias que já nomeia outro modelo continua a ser
  recusado, também pelo `lookup`. O `unregister` usa o mesmo `lookup`. Por coerência, um
  `replace=True` mantém os aliases de todas as camadas: são identidade, não factos.
- **Testes:**
  - `test_register_load_and_unregister_name_the_entry_get_finds`: no catálogo DEFAULT,
    `register` e `load` por `grok-4.20-reasoning`, e `register` e `unregister` por
    `claude-haiku-4-5-20251001`;
  - `test_replace_through_an_alias_keeps_the_ids_that_name_the_entry`;
  - `tests/test_probe_models.py::test_a_fragment_keyed_by_an_alias_loads_over_the_default_catalog`:
    um fragmento de probe com a chave `["xai"."grok-4.20-reasoning"]` carrega sobre o
    `ModelCatalog()`.

**2. (Medium) Threads.**

- **Mudança:** o catálogo passou a copy-on-write, uma geração por estado.
  - Um `_State` imutável guarda as camadas, os substituídos e o índice, construído pelo
    escritor. As entradas construídas guardam-se no próprio estado (`built`).
  - Cada mudança constrói o estado seguinte sob um `threading.Lock`, uma de cada vez.
  - Uma leitura toma o estado corrente e nunca espera. O que constrói fica no estado que leu,
    por isso nunca é visto depois de uma mudança.
  - Não há iteração sobre dicts mutáveis, nem lock tomado enquanto se importa um adaptador.
- **Testes:** dois, determinísticos, sem esperas por tempo:
  - `test_a_read_that_overlaps_a_register_never_keeps_the_entry_it_built`: o leitor fica
    parado no `model_facts` enquanto o escritor regista `context_window=7`. Antes da correcção
    dava 400000;
  - `test_a_read_that_overlaps_a_load_never_hides_the_model_it_added`: o leitor fica parado a
    percorrer os aliases enquanto o escritor carrega o `llama3`. Antes da correcção dava `None`.
  - Corri ambos 20 vezes seguidas, sempre verdes.

**3. (Low) O `register()` valida como o `load()`.**

- **Mudança:**
  - O `ModelCapabilities.__post_init__` corre os `_PARSERS` em cada facto, normaliza os sets
    para `frozenset` e os esforços para um tuplo ordenado, e verifica os aliases e as
    `sources`.
  - O `Provenance.__post_init__` verifica o `kind`, o `ref` e o `verified_at`, e recusa uma
    datetime. O `_provenance` do loader delega nele e junta o contexto.
  - O `_strings` aceita qualquer colecção que não seja uma string.
- **Testes:**
  - `test_register_refuses_what_load_refuses`, 11 casos;
  - `test_a_source_is_checked_as_the_loader_checks_it`;
  - `test_an_entry_keeps_its_sets_and_efforts_as_the_loader_does`, com o `json.dumps` do
    `to_dict()`.

**4. (Low) Lacunas do loader.**

- **Mudança:**
  - `SERVER_TOOL_TYPES` em `_base.py` (`web_search`, `code_execution`, `image_generation`) é o
    vocabulário dos `server_tools`;
  - o `ordered_efforts` deduplica;
  - o `catalog_version` exige `type(...) is int`;
  - um `TOMLDecodeError` passa a `ValueError` com o caminho do ficheiro.
- **Testes:**
  - um caso novo `server_tools = ["web_serch"]` no paramétrico do loader;
  - `test_the_loader_needs_its_version`, agora com 4 casos (`true` incluído);
  - `test_a_file_that_is_not_toml_is_refused_by_its_path`;
  - `test_the_loader_lists_each_effort_once`;
  - `test_the_server_tool_vocabulary_is_the_types_core_makes`: o vocabulário é igual aos
    `ServerTool(type=...)` do `core/_server_tools.py` (lidos por AST) e às chaves do contrato do
    fio. A C05 tem de acrescentar os seus tipos aos três.

**5. (Low) C06.1 "através do adaptador".**

- **Mudança:** só nos factos de `_base.py`. Nenhum classmethod `model_facts` mudou.
  - O `AdapterFacts` ganhou `output_modalities`, que estreita como as entradas.
  - O `DRAWS_WITHOUT_TOOLS` (imagem do Gemini, Muse Image) leva as `CONTENT_PARTS`, com o PDF,
    e devolve `{"text", "image"}`.
  - O `DRAWS_ONLY` (Images API da OpenAI, Grok Imagine) devolve só `{"image"}`.
  - O `OutputModality` mudou-se para `_base.py`.
- **Resultado:**
  - o `gemini-3.1-flash-image` dá `{"text", "image", "pdf"}` (o vídeo fica de fora);
  - o `gpt-image-1.5` dá `{"image"}` com `kind="adapter"`.
- **Testes:**
  - `test_a_pdf_reaches_the_wire_where_the_catalog_says_the_model_reads_one`, sobre as 43
    entradas: o `"pdf"` do adaptador é exactamente o que o `prepare()` põe no fio, e um PDF que
    o catálogo lista chega ao fio. Antes falhava nos 3 Gemini de imagem e na Muse Image;
  - `test_a_model_no_completion_reaches_writes_only_images`, sobre as 43;
  - `test_a_gemini_image_model_reads_the_pdfs_its_completions_carry`;
  - `test_the_images_api_narrows_a_published_text_output`.

**6. (Low) Estreitar também `kind="api"`.**

- **Mudança:** `_PUBLISHED = {"docs", "api"}`.
- **Teste:** `test_the_adapter_narrows_a_models_endpoint_list_too`.

**7. (Low) Um SDK partido.**

- **Mudança:** o `_adapter_of` apanha `Exception`, e os factos do adaptador ficam `None`.
- **Teste:** `test_a_broken_sdk_leaves_the_adapters_facts_unknown`: um módulo que levanta
  `TypeError` ao importar.

**8. (Low) O `false` do fragmento.**

- **Mudança:**
  - O `ProbeResult` ganhou `refused_by_adapter`, que o `run_probe` preenche com
    `_refused_by_adapter(exc)`: um `RequestError` exacto levantado dentro de um `prepare` ou
    `prepare_image` de um módulo de `core/_providers`. Isto exclui a validação do SDK (levantada
    ao enviar), o `UnpricedModelError` e os argumentos recusados pelo próprio `LLM`.
  - Só isso dá `false`. Um `unsupported_capability` heurístico não afirma nada: aparece num
    comentário `# [...] facto = false`, para rever.
  - Não usei "códigos documentados": não confirmei nenhum código de "unsupported" nas docs, e
    sem isso a heurística voltaria por outra porta.
- **Testes:**
  - `test_a_provider_error_that_reads_as_unsupported_is_left_for_review`;
  - `test_only_the_adapters_own_preparation_is_its_refusal`: o `prepare` real do
    `grok-4.20-multi-agent` com tools, o `refused_or_unread`, o `UnpricedModelError` e um
    `RequestError` solto;
  - o `test_the_catalog_fragment_states_only_what_a_run_proved` foi ajustado.
  - À parte, na scratchpad, confirmei pelo `LLM` real, com sockets bloqueados, que a recusa do
    adaptador é reconhecida e a dos argumentos não.

**Docs:**

- `docs/model-catalog.md`: modalidades de saída, imagem, `api`, SDK partido, ids nos três
  métodos, validação, threads, loader e fragmento;
- `scripts/model_probe_notes.md`: a regra do `false` e os aliases.

**Ficheiros:**

- `core/_model_catalog.py`;
- `core/_providers/_base.py`;
- `scripts/probe_models.py`, `scripts/model_probe_notes.md`;
- `tests/test_model_catalog.py`, `tests/test_probe_models.py`;
- `docs/model-catalog.md`.

**Gate:**

- O conjunto pedido (catálogo, dados, probes, exports, arquitectura, `test_*_provider.py`) e o
  `test_quality_budget.py`: 1411 passed.
- `pyright src`: 0 erros.
- `ruff check` e `ruff format --check` nos meus ficheiros: limpos.
- Suite completa: 8143 passed, 9 failed, todos em `tests/toolkit/test_eurostat.py`, de outro
  cartão e fora do meu âmbito.

**CHANGELOG, linhas propostas que substituem as anteriores:**

```
- `model_catalog`, `ModelCatalog`, `ModelCapabilities` and `Provenance` (`ai_arch_toolkit.core`): what each model takes through its adapter — context window, input and output limits as each provider publishes them, input and output modalities, tools and tool choices, structured output, JSON mode, streaming, reasoning mode and efforts, thinking budget, server tools — each fact with its source (`docs`, `probe`, `api`, `adapter`, `override`) and the day it was read. A fact is `None` when unknown, never "no". The shipped seed covers the current line (30 chat models still served) and the image models, with the providers' published limits and modalities read on 2026-10-08; the rest comes from each adapter's own rules, which also narrow a published modality list to what the adapter carries. Override with `load()` (a strict TOML loader) and `register()`, which checks its facts as `load()` does, field by field; both, and `unregister()`, name a model by its id, an alias or a dated snapshot, as `get()` does. `ModelCatalog(defaults=False)` starts empty. A catalog is safe to share between threads. Nothing in the toolkit reads the catalog: the adapters keep their rules (D63). See `docs/model-catalog.md`.
- `scripts/probe_models.py` writes `<run_id>.catalog.toml` next to each report: what the run proved, as `kind = "probe"` catalog facts, for review or `model_catalog.load()`. A fact is false only where the adapter refused the call; a provider error that reads as unsupported is left in a comment, for review.
```
