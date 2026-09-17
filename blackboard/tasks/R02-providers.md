# R02 · Fase 2 — Fornecedores: ids e preços, contrato de três fases, um adaptador de cada vez

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** R01 · **Regras:** `R00-rules.md`
- **Plano:** `docs/internal/hardening-plan.md` — causas 2, 3 e 4, costuras B e D, secção 4
- **Achados (`FINDINGS.md`, 2026-09-15 e 2026-09-17):** todos os que falam de adaptadores, streams,
  preços e regras por modelo; em particular "Erros de transporte a meio de um stream…", "Validação
  local do SDK…", "`stream_events()` perde `parsed`…", "Gemini em stream reenvia só o último chunk",
  "Stream sem usage…", "Gemini nunca junta os resultados de tools…", "Anthropic: um `user` por
  `tool_result`", "`thinking=True` no Anthropic…", "`tool_choice="required"` recusado no Fable 5.1",
  "xAI: `tool_choice` com nome…", "Gemini 3: `thinking_effort`…", "O Gemini muda de pilha HTTP…",
  "Server tools no fio…", "O preço por prefixo…", "Regras por modelo: o ramo por omissão é o antigo"
- **Decisões em vigor:** D3, D11–D14, D15, D16, D18, D20
- **Ordem dos adaptadores (D18):** OpenAI → xAI → Gemini → Meta → Anthropic. Acaba um antes de
  começar o seguinte.

## Objectivo

Cada adaptador fica reduzido a quatro peças pequenas — construir o pedido (tipado pelo SDK), mapear
erros (um só sítio), descodificar eventos, montar a resposta (uma só vez) — sobre uma base que é dona
do algoritmo. Streaming e `complete()` dão a mesma `Response` por construção. As regras por modelo e
os preços resolvem ids por uma gramática única, e um modelo novo recebe por omissão o comportamento
da geração mais recente.

## Passos, por ordem

### 1. Costura D — ids de modelo e preços (`core/_model_id.py`, novo)

- Uma função resolve um id em: exacto; id com sufixo de snapshot (gramática de datas e carimbos por
  fornecedor: `-20250929`, `-2025-04-14`, `@20251101`, `-latest`, `-001`, `-preview-05-20`, `-0709`);
  família; desconhecido. Casos que têm de dar "desconhecido": `o3-pro`, `o3-deep-research`,
  `gpt-4o-audio-preview`, `gpt-5-codex`, `gemini-2.5-flash-image`, `grok-4.6-mini`, `claude-fable-5-1`
  sem entrada própria.
- Usam-na o `PricingRegistry` (deixa de casar por prefixo; `register(..., match="prefix")` fica como
  opção explícita para famílias locais), o encaminhamento por fornecedor (`_providers/__init__.py`), a
  escolha do tokenizer (`core/_tokens.py`) e os perfis dos adaptadores. Regra de arquitectura:
  nenhuma verificação `startswith` sobre ids de modelo fora de `_model_id.py`.
- D16/D20: com um `MeterScope` ligado, um modelo sem preço falha antes de qualquer pedido, com um
  erro que diz como registar o preço; um modelo local regista zero de forma explícita; sem scope a
  chamada corre e `Response.cost` fica `None`.
- `scripts/audit_models.py`: compara as listas de modelos dos fornecedores (APIs de modelos, grátis)
  com a tabela de preços e os perfis. Só o dono o corre; não o corras.

### 2. Costura B — contrato do provider (`core/_providers/_base.py`)

- Três fases obrigatórias: `prepare()` síncrono e puro (levanta `RequestError`), `send()` e
  `open_stream()` com toda a E/S, e uma só montagem da `Response`. `complete`, `stream` e
  `stream_events` existem uma vez, na base. A pipeline da R01 passa a chamar `prepare()` antes de
  admitir a operação e `send()` depois: um erro de construção nunca abre operação nem conta como
  tentativa.
- Um duplo de teste partilhado (`tests/` — um `FakeProvider` que cumpre o contrato) substitui os
  providers duck-typed que só têm `complete`/`stream` (27 sítios em 21 ficheiros). Não fica nenhum
  caminho alternativo em `LLM` para providers sem contrato.
- Confirma SDK a SDK, nas versões instaladas, se o próprio SDK acumula o objecto final do stream; onde
  acumula, usa-o; só escreve junção à mão onde não há (Gemini).

### 3. Por adaptador, pela ordem de D18

Para cada um, tudo isto antes de passar ao seguinte:

- **Um mapeador de erros** à volta de cada `await` do SDK e de cada passo da iteração do stream.
  Nenhuma excepção de SDK, `httpx2`, `aiohttp` ou gRPC sai crua. Cada erro sai tipado (costura A) com
  `delivery` segundo a política de facturação documentada do fornecedor, citada em comentário com URL:
  Anthropic não cobra respostas de erro e cobra timeouts do cliente; Gemini não cobra 400 nem 500;
  429 é `unbilled` em todos; o que não está documentado fica `indeterminate`. Falhas da fase de ligação
  (`ConnectError`, `ConnectTimeout`, `PoolTimeout` na causa) são `not_sent`. Erros dentro do stream
  mapeiam para o estado documentado equivalente, nunca 200 nem 5xx inventado sem base.
- **Pedido tipado:** o construtor devolve o `TypedDict` do SDK (modelos pydantic no Gemini, protobuf
  no xAI), para o pyright verificar o fio. Regra de arquitectura: excepções de SDK só no mapeador.
- **Perfil por modelo:** uma tabela declarativa por adaptador (família → forma do pedido), resolvida
  pela costura D; o perfil por omissão é o da geração mais recente e as famílias antigas são lista
  fechada. O que o modelo não suporta levanta `RequestError` em `prepare()`, como a Meta (D13).
- **Paridade:** a mesma resposta lógica por `complete()` e por stream em chunks dá `Response` igual
  (texto, tool calls, thinking, `parsed`, `citations`, `response_id`, usage, e o que o histórico
  reenvia). Stream sem usage dá custo desconhecido.
- **Contrato de conversa:** históricos neutros canónicos (uma chamada; duas chamadas paralelas e dois
  resultados; texto com chamadas; thinking com chamadas; histórico depois de stream) convertidos
  segundo as regras documentadas do fornecedor: resultados de um turno juntos e pela ordem das
  chamadas, com `id` onde existe, assinaturas preservadas.
- **Testes de transporte** com `prototypes/…/adapter-phases/fakeserver.py` como ponto de partida:
  ligação recusada, servidor mudo, fecho sem resposta, RST, 429, 500, 529, erro dentro do stream,
  corte a meio, e o caminho de sucesso. Cada caso dá o tipo de erro e o `delivery` certos e bate com a
  matriz da R01.

Específico de cada um (além do que os achados listam):

- **OpenAI:** o erro dentro do stream deixa de sair como `openai.APIError` cru; o limite de tokens
  vai em `max_completion_tokens` no host oficial e em `max_tokens` só com `base_url` próprio; server
  tools no Chat Completions e `ServerTool.config` não vazio levantam `RequestError` (a funcionalidade
  é do `C05`); o corpo JSONL do batch passa pelo mesmo construtor tipado.
- **xAI:** `tool_choice` com nome de tool usa a forma do SDK; os códigos gRPC deixam de virar 5xx
  inventados (`UNAVAILABLE` e `DEADLINE_EXCEEDED` são falhas de transporte); o comentário sobre server
  tools fica certo.
- **Gemini:** resultados de tools juntos num só `Content`, com o `id` da chamada; o `raw` de um stream
  guarda todas as partes e assinaturas; pilha fixa em `httpx` por `HttpOptions.httpx_async_client`
  (acaba o reenvio escondido); `thinking_effort` inválido levanta `RequestError`. A nota "Known issue"
  em `docs/model-compatibility.md` e no `AGENTS.md` só sai depois de o dono confirmar ao vivo.
- **Meta:** os desvios aos tipos do SDK `openai` que o servidor aceita ficam numa lista explícita,
  cada um com a nota de verificação ao vivo do M01; o usage de um `response.failed` deixa de se perder.
- **Anthropic:** thinking segundo a documentação actual — 4.6 e seguintes com `{"type": "adaptive"}`
  e `output_config.effort` (fundido com `output_config.format`), `display: "summarized"` quando se pede
  thinking, `thinking=False` omite o parâmetro onde o thinking vem ligado, só as famílias antigas levam
  `budget_tokens` com `max_tokens = budget + max_tokens` (D20); `tool_choice` forçado em Fable 5.1 e
  Mythos 5.1 levanta `RequestError`; todos os `tool_result` de um turno num só `user`; server tool com
  `name`; ramo 429 em `count_tokens` e batch. Confirma cada regra na documentação oficial antes de a
  codificar.

### 4. Rede de testes do fio

A fixture que valida cada pedido construído contra os tipos do SDK (protótipos em
`prototypes/…/wire-contract/`) fica confinada a `tests/`, com os canários: os pedidos maus conhecidos
têm de falhar e os bons de passar. Serve para o que a tipagem estática não vê.

### 5. Verificação ao vivo (só preparar)

Não há créditos Anthropic nem xAI e o agente nunca chama fornecedores. Deixa testes `live_api`
pequenos e baratos por adaptador (modelo mais barato, `max_tokens` baixo): chamadas paralelas com
reenvio em `complete` e em stream, thinking ligado e desligado, um erro 4xx sob `max_cost` seguido de
sucesso. Escreve na ficha o comando exacto por fornecedor para o dono correr.

### 6. Documentação e registo

`docs/llm.md`, `docs/pricing.md`, `docs/model-compatibility.md`, `docs/safety.md`, `AGENTS.md`
(gotchas e a deriva sobre o structured output, se a F25 não a tiver fechado), `CHANGELOG.md`, blackboard.

## Aceitação

- Testes de paridade, de contrato de conversa e de transporte verdes nos cinco adaptadores.
- Regras de arquitectura verdes: excepções de SDK só nos mapeadores; ids de modelo só em
  `_model_id.py`; nenhum provider sem contrato.
- Saldo de linhas negativo em `core/_providers/` (hoje 4294 linhas nos cinco adaptadores; 44 blocos
  `except` de SDK); nenhuma função nova acima de complexidade 10; pyright limpo com os pedidos tipados.
- A matriz da R01 continua verde, agora com erros de construção a não abrirem operação.

## Fora do âmbito

Funcionalidades novas de server tools, `server_tools=` nas estratégias e custo por uso (`C05`);
`Agent.stream()` (`C01`); catálogo de modelos (`C06`); porta para a Responses API do OpenAI; tools e
motor (R03).

## Registo do dono

- Estado: todo
- Notas de desenho:
- Por adaptador (feito, testes, linhas antes e depois):
- Ficheiros tocados:
- Testes novos e corrigidos:
- Verificações:
- CHANGELOG:
- Comandos de verificação ao vivo para o dono:
- Bloqueios:
- Desvios ao plano:
- Commits propostos:
