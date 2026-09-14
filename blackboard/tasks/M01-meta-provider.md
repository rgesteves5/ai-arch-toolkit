# M01 · Fornecedor Meta (Muse Spark)

- **Dono:** coordenador · **Estado:** done · **Depende de:** `main` @ `a0807cc`
- **Origem:** pedido do dono (2026-09-13): integrar a Meta Model API "da melhor forma e a correcta",
  com autorização para chamadas ao vivo com `MODEL_API_KEY`.
- **Decisões:** D11–D14 em `DECISIONS.md`.

## Problema

A Muse Spark é servida pela Meta Model API (`https://api.meta.ai/v1`, bearer `MODEL_API_KEY`). Não há
SDK da Meta: a documentação manda usar o SDK `openai` (Responses, Chat Completions) ou o `anthropic`
(Messages). O toolkit não tem adaptador para ela; `LLM("muse-spark-1.3")` não é reconhecido e o caminho
`provider="openai"` + `base_url` usa Chat Completions, que não mantém o raciocínio entre voltas.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/_meta.py` (novo), `_providers/__init__.py`
- `src/ai_arch_toolkit/core/_default_pricing.toml`, `src/ai_arch_toolkit/core/_tokens.py`
- `pyproject.toml`, `uv.lock`
- `tests/test_meta_provider.py` (novo), `tests/test_provider_registry.py`, `tests/test_pricing.py`,
  `tests/test_tokens.py`, `tests/test_probe_models.py`, `tests/integration/conftest.py`,
  `tests/integration/test_meta_live.py` (novo)
- `scripts/probe_models.py`, `scripts/model_probe_models.toml`, `scripts/model_probe_notes.md`
- Docs: `README.md`, `.env.example`, `CONTRIBUTING.md`, `AGENTS.md`, `docs/index.md`,
  `docs/getting-started.md`, `docs/llm.md`, `docs/model-compatibility.md`,
  `docs/framework-overview.md`; `CHANGELOG.md`;
  `.github/workflows/integration.yml`

## Registo do dono

- Probes ao vivo à Responses API (SDK `openai` 2.45.0) antes de escrever o adaptador: formato dos
  itens de saída, reenvio com e sem ids, `phase`, eventos de stream (incluindo tool calls paralelas
  intercaladas e `response.incomplete`), `tool_choice`, esforços, `stop`, `strict`, `json_object`,
  PDF, imagem, `web_search` com `url_citation`, `input_tokens.count`, erros 404/503.
- Implementado `MetaProvider` (`_meta.py`), routing `muse-spark-`, preços, extra `meta`,
  `count_tokens_local` com `o200k_base` (medido: contagem da Meta ≈ 0,98 × o200k).
- `scripts/probe_models.py`: `tool_choice` por modelo e redacção do formato de chave da Meta.
- Revisão adversarial (1 revisor, só leitura) encontrou 3 bugs confirmados, todos corrigidos com
  testes que falham antes:
  - tool calls paralelas em stream ficavam pela ordem de chegada e o reenvio perdia o raciocínio;
  - recusa em stream perdia o texto e o turno desaparecia do histórico;
  - códigos de falha causados pelo pedido viravam 500 e eram repetidos (agora 400).
  Também corrigido: item de raciocínio sem `encrypted_content` não é reenviado (400 confirmado ao
  vivo); `complete()` levanta perante `status: failed`; `openai.APIError` com payload de erro a meio
  do stream é mapeado. README ganhou a coluna Meta; job de integração com 30 minutos.
- Verificado ao vivo e descartado: reenvio de raciocínio da 1.3 para a 1.2 e a 1.1 é aceite;
  histórico com tool calls sem `tools` no pedido é aceite (`tool_choice="none"`, ReAct sem tools
  no último turno).

## Provas

- Unitários: `tests/test_meta_provider.py` (54), `tests/test_provider_registry.py` (+6),
  `tests/test_pricing.py` (+1), `tests/test_tokens.py` (+2), `tests/test_probe_models.py` (+1).
- Suite: 3020 passed, 20 skipped; `ruff check src tests examples` e format limpos; pyright 0 erros;
  `uv lock --check` OK.
- Ao vivo: `tests/integration/test_meta_live.py` 6 passed (e de novo `tool_loop` e `streamed` depois
  das correcções, com asserção de reenvio após stream); matriz `scripts/probe_models.py` 6/6 (stream
  à segunda, por 503); SDK mínimo `openai==2.6.0`: unitários 54 passed, `tool_loop`, `count_tokens` e
  `streamed` passed ao vivo.
- A Meta respondeu 503 `service_overloaded` a uma parte grande dos pedidos durante o dia.

## Linhas do CHANGELOG

Entrada "Meta provider (Muse Spark)" e a do `tool_choice` do inventário, em `[Unreleased]` → Added.

## Seguimento

- O dono não quer testes reais no CI do GitHub: `.github/workflows/integration.yml` removido e o job
  de testes do `ci.yml` com `-m "not live_api"`. Os testes ao vivo da Meta correm só localmente
  (`uv run pytest -m live_api -k meta`, com a `.env` carregada).
