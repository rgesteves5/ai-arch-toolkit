# O04 · Documentação, quebras visíveis e verificação ao vivo final

- **Dono:** Claude (agente, os docs; o coordenador, a verificação ao vivo, 2026-10-02) ·
  **Estado:** done · **Depende de:** O03
- **Origem:** D43 · **Decisões:** D43 · **Regras:** `R00-rules.md`

## Problema

A O03 muda o que o OpenAI aceita e devolve. Os docs, o `AGENTS.md` e o inventário de probes dizem
hoje "Chat Completions only" e "Astra calls no tools here".

## Objectivo

- Docs e `AGENTS.md` a dizer o que o código faz depois da O03.
- Linhas propostas para o `CHANGELOG`: a quebra, com "**Breaking:**", e o que passa a funcionar.
- Inventário de probes com `tools_loop` no Astra e no 6.1 Sol.
- Matriz ao vivo dos modelos OpenAI, corrida pelo dono.

## Ficheiros

- `AGENTS.md` (as linhas do OpenAI e da Meta), `docs/llm.md`, `docs/model-compatibility.md`,
  `docs/framework-overview.md`, `README.md` (se tiver a tabela de fornecedores)
- `scripts/model_probe_models.toml`
- `tests/integration/test_provider_hardening_live.py` (se as regras verificadas ao vivo mudarem)

## Verificação ao vivo (dono)

```bash
set -a && source .env && set +a
uv run python scripts/probe_models.py --suite full --timeout-seconds 120 \
  --model gpt-6-astra --model gpt-6.1-sol --model gpt-6-sol --model gpt-6-luna --model gpt-5.5
uv run pytest -m live_api -k openai -q
```

O `probe_models.py` filtra por modelo (`--model`, repetível), não por fornecedor: junta os outros
ids OpenAI do inventário que quiseres cobrir.

## Registo do dono

- Estado: done (2026-10-02; os docs pelo agente, revistos pelo coordenador). A verificação ao vivo correu-a o Claude
  (coordenador) a 2026-10-02, com autorização do dono para as chamadas pagas. O inventário de probes, as linhas
  do `CHANGELOG`, a tabela medida do `_openai.py` e os testes dela são do coordenador e não foram
  tocados nesta parte.

### Verificação ao vivo (Claude, 2026-10-02)

- `scripts/probe_models.py --suite full`, relatório `20261002T041947Z` (em
  `scripts/output/model-probes/`, ignorado pelo git): 77 de 77. Os 13 modelos OpenAI do
  inventário (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.5`, `gpt-5.4`,
  `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-4.1`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5`, `o3`), todos os
  cenários (`plain`, `tools_loop`, `structured`, `json_mode`, `stream`, `thinking`; o `gpt-4.1`
  sem `thinking`). O Astra e o 6.1 Sol passaram `tools_loop` pela primeira vez.
- `pytest -m live_api tests/integration -k openai`: 6 de 6, em três corridas (junção dos prompts
  de sistema, `output_schema` com um Pydantic simples, reenvio de calls paralelas completo e em
  stream, `thinking` ligado e desligado, um 4xx sob tecto de custo).
- Esforços medidos na Responses (esforços; esforço sem esforço enviado): GPT-5, 5 mini e 5 nano
  `minimal` a `high`, `medium`; GPT-5.1 `none` a `high` (sem `minimal`), `none`; GPT-5.2 e
  GPT-5.4 (mini, nano) `none` a `xhigh` (sem `minimal`), `none`; GPT-5.5 idem, `medium`;
  GPT-5.6 (Sol, Terra, Luna) `none` a `max` (sem `minimal`), `medium`; o3 `low` a `high`,
  `medium`; Astra e 6.1 Sol `low` a `max`, raciocinam sempre, sem sampling; Sol e Luna `none` a
  `max`, `medium`, sampling só a `none`; GPT-4o e 4.1 não raciocinam.

### Ficheiros tocados e o que dizem agora

- `AGENTS.md`: a linha do OpenAI reescrita. O host escolhe a API (D43): no host oficial,
  Responses sobre o núcleo partilhado, sem estado, reenvio do `_raw` só na mesma família,
  `max_output_tokens`, `reasoning.effort` (omissão `high`) com `summary: "auto"` só com
  `thinking=True`, tools a qualquer esforço, `strict: false`, `web_search()` alojada,
  `count_tokens` e batches em `/v1/responses`, as cinco kwargs que levantam, a tabela `_MODELS`
  medida e o que cai a raciocinar. Noutro host, Chat Completions sem regras de modelo. A linha da
  Meta diz que o núcleo serve os dois adaptadores.
- `docs/llm.md`: `system=` vai como `instructions` no OpenAI e na Meta, e como mensagem de sistema
  inicial nos servidores compatíveis; `stop_reason` `"completed"` no OpenAI e na Meta, `"stop"`
  nos compatíveis; o OpenAI conta tokens (o xAI e os compatíveis levantam
  `NotImplementedError`); a tabela de thinking ganha uma linha para o OpenAI (`api.openai.com`)
  e outra para os compatíveis; o parágrafo da Chat Completions deu lugar ao comportamento actual
  (tools a qualquer esforço, quem raciocina por omissão, `thinking=True, thinking_effort="none"`
  para não raciocinar, o que cai); o reenvio cifrado é do OpenAI e da Meta, na mesma família.
- `docs/model-compatibility.md`: secção OpenAI reescrita. O host decide; lista do que vale no
  host oficial (sem estado e famílias, limite e thinking, tools e `strict: false`, `web_search`,
  `text.format` e as kwargs recusadas com o que deram ao vivo, logprobs, `count_tokens`, batch,
  códigos de erro, modelos retirados); tabela nova "Efforts per model" com o medido; subsecções
  do Astra, do 6.1 Sol e do Sol/Luna sem a Chat Completions; "Recorded live baseline" com a
  corrida de 2026-10-02 e os 6 testes ao vivo, numa só tabela de 13 modelos (sai a tabela do
  Sol/Luna de 2026-09-25). Na introdução, um bloco com a corrida OpenAI (13 modelos, 77 de 77) e
  o Astra e o 6.1 Sol entre os que passaram; ficam sem resultado só o `grok-4.7` e o
  `claude-opus-5-5`. Em "Current Gaps": as verificações ao vivo das regras correram só no OpenAI,
  e no OpenAI faltam ao vivo o turno reconstruído, o batch em `/v1/responses` e os resumos numa
  organização por verificar.
- `docs/tools.md`: o OpenAI (host oficial, Responses) manda a `web_search()` como a Meta; o xAI e
  os servidores compatíveis recusam as server tools.
- `docs/framework-overview.md`: `OpenAICompatibleProvider` na lista, o OpenAI e a Meta como dois
  perfis sobre o núcleo da Responses, e o host a decidir a API no OpenAI.
- `docs/getting-started.md`: os servidores compatíveis recebem Chat Completions sem as regras do
  OpenAI; o host oficial vai pela Responses.
- `README.md`: na matriz, o OpenAI ganha "effort + summaries" no thinking e "web" nas server
  tools.
- `CONTRIBUTING.md`: a referência para um adaptador novo passa a `_anthropic.py` (contrato
  inteiro e tabela por modelo num só módulo); o `_openai.py` e o `_meta.py` ficam como exemplo de
  perfis sobre um núcleo partilhado.

### Verificações

- `uv run mkdocs build --strict`: não corrido, o extra `docs` não está instalado.
- Não há `tests/test_docs*.py`.
- `uv run pytest -q`: 6034 passed, 42 skipped (a linha de base).

### Incoerências encontradas (não corrigidas aqui)

1. **`CHANGELOG`, `[Unreleased]`.** Quatro entradas antigas contradizem a quebra da O03:
   - Added, GPT-6.1 Sol: "tool calls raise `RequestError`, since Chat Completions takes its
     requests only without tools";
   - Added, Opus 5.5/GPT-6 Sol e Luna/Grok 4.7: as tool calls do Sol e da Luna enviadas a
     `"none"` e as tools com `thinking=True` a levantar;
   - Changed: "OpenAI, GPT-5.4 and later: tools with `thinking=True` at an effort other than
     `"none"` raise `RequestError`";
   - Fixed: "`thinking_effort="max"` on a GPT-6 model raises `RequestError`" (agora os GPT-6
     aceitam `max`).
2. **`examples/25_server_tools.py`**, docstring: "the OpenAI (Chat Completions) and xAI adapters
   raise RequestError". É código, por isso não lhe toquei.
3. **`research/agent-strategies/02-react.md`** (linhas 13 e 144) e **`00-index.md`** (linha 65):
   "no reasoning replay on OpenAI (Chat Completions only)". A R00 proíbe tocar em `research/`.
4. **`_responses.py`**, comentário do desvio 1: diz que a forma reconstruída fica para a
   verificação ao vivo da O04. As probes e os testes ao vivo reenviam o turno inteiro pelo
   `to_message()`, por isso continua sem prova ao vivo (fica em "Current Gaps").
5. **Código, para o coordenador decidir:**
   - no host oficial, um `thinking_effort` sem `thinking=True` cai sem aviso. Com modelos que
     raciocinam por omissão (GPT-5, 5.5, 5.6, o3, GPT-6), quem passa `thinking_effort="none"`
     para desligar o raciocínio fica a `medium` sem saber;
   - variantes com preço e sem regra levam as da geração actual (todos os esforços, `medium`),
     sem medição: `gpt-5.3`, `gpt-5.3-codex`, `gpt-5.1-codex`, `gpt-5-pro`, `gpt-5.5-pro`,
     `gpt-5.6-cyber`, `o3-pro`.

### Divisão proposta em commits

- `fix(openai): take each model's efforts and default effort from live measurements (O04)`: o
  `_openai.py`, os testes, o inventário, o teste ao vivo e a entrada Fixed do `CHANGELOG` (do
  coordenador).
- `docs: describe OpenAI through the Responses API (O04)`: os oito ficheiros acima.
- `docs(blackboard): record O04`, com esta ficha.

### Fecho pelo coordenador (2026-10-02)

- A verificação ao vivo achou a tabela de regras do OpenAI errada para a maioria dos modelos (o
  gpt-5, os 5.5, os 5.6, o o3 e os GPT-6 raciocinam sem esforço enviado e recusavam a
  `temperature` do `LLM`). Os esforços e o esforço por omissão passaram a vir de medições ao vivo,
  com as variantes pro e codex. Commit `17e0938`, com o `thinking_effort` a aplicar-se sozinho
  (D45) e os resultados de batch à tarifa de batch (OpenAI, compatíveis e Anthropic).
- Provados ao vivo depois da matriz: um turno reconstruído (texto antes da call, sem `_raw`),
  aceite no `gpt-5-nano` e no `gpt-6-luna`; um batch de dois pedidos em `/v1/responses`
  (`batch_6abf34d53ddc8190a6949a56e500f0c9`), lido com texto, uso, resumo do raciocínio e custo à
  tarifa de batch (14 + 113 tokens do `gpt-5-nano`: $0.00002295).
- Das incoerências acima: o `CHANGELOG` foi consolidado; o exemplo 25 corrigido; o comentário do
  desvio 1 do `_responses.py` actualizado; o `research/` fica anotado em `FINDINGS.md`.
- Docs: commit `2e73faa`.
