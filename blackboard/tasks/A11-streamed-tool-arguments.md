# A11 · Os argumentos das ferramentas a chegar (G-37)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 12 (G-37) · **Decisões:** D60 ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `3273a9a`)

- Um `StreamEvent` era `text`, `thinking`, `tool_call` ou `image`, e só o `thinking` e o `image`
  tinham `partial=True`.
- Os adaptadores juntavam os pedaços de uma chamada e só a entregavam completa, depois do fim do
  stream. Na app, um artefacto só aparece quando está guardado.

## Registo do dono

- Estado: done (Claude, 2026-10-05), pela D60. O dono tinha pedido para parar antes deste grupo,
  e depois pediu para terminar a frente A toda.
- **O evento:** `StreamEvent(kind="tool_call_delta", partial=True)` com um `ToolCallDelta`. Os
  campos são o `index` (o lugar na resposta), o `id`, o `name` e o `input_json` (o pedaço
  seguinte do input). É público em `ai_arch_toolkit` e em `ai_arch_toolkit.core`.
- **Os adaptadores**, todos pelo `CallPieces` de `_base.py`:
  - **Anthropic:** o `content_block_start` de um `tool_use` e os `input_json_delta`. Os blocos
    `server_tool_use` não entram;
  - **Responses API (OpenAI e Meta):** o `output_item.added` de um `function_call` e os
    `function_call_arguments.delta`, pelo `output_index`. O `done` dá o que os deltas não
    escreveram;
  - **Chat Completions:** os deltas por `index`;
  - **Gemini e xAI:** cada chamada inteira, num pedaço, quando chega. O `stream_function_call_arguments`
    é só da Vertex AI, e o xAI manda a chamada "in whole in a single chunk".
- **De passagem:** o `parse_tool_args` lê um texto vazio como uma chamada sem argumentos (`{}`).
  Antes dava `{"_raw": ""}`, que um servidor compatível manda para uma tool sem parâmetros.
- **A revisão independente** não achou erros no mapeamento dos lugares. Apontou:
  - **média:** os pedaços contam como entregues (D54). Uma resposta só com chamadas que falhe
    depois do primeiro pedaço já não se repete no `stream_events()` nem num flow iterado. Ficou
    decidido assim e escrito na D60, no CHANGELOG e num teste (`TestRetries`). A alternativa
    misturava, à vista do consumidor, os pedaços da tentativa falhada com os da nova;
  - **baixas, todas tratadas:**
    - o texto vazio de uma chamada sem input;
    - o `done` da Responses API;
    - a exportação de topo;
    - a documentação (`docs/llm.md`, `docs/examples.md`, `docs/framework-overview.md`) e o
      exemplo 20;
    - seis testes que faltavam: um server tool da Anthropic, itens entre as chamadas da
      Responses, a Meta sem deltas, o `stream_events_sync`, um flow iterado e a retentativa.
  - **Recusado:** um nome que chegue depois do id na Chat Completions. O acumulador do próprio
    SDK já falha nesse caso, antes do adaptador, por isso tirei o ramo que o tratava.
- **Testes:** `tests/test_stream_tool_call_pieces.py` (22). Quatro testes antigos passaram a
  contar os pedaços.
- **Docs:** `docs/getting-started.md`, `docs/llm.md`, `docs/api.md`, `docs/flow-architecture.md`,
  `docs/framework-overview.md`, `docs/examples.md`, `examples/20_rich_streaming_events.py`,
  `AGENTS.md`, `CHANGELOG.md`.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 12 fechou. O
  `stream_events` dá eventos `tool_call_delta`, cada um com um `ToolCallDelta`: o `index` da
  chamada na resposta, o `id`, o `name` (desde o primeiro pedaço, logo que o nome se sabe) e o
  `input_json`, o pedaço seguinte do input. Os pedaços, juntos, são o input.

  A Anthropic, a OpenAI, a Meta e os servidores compatíveis mandam-nos enquanto o modelo
  escreve. O Gemini e o xAI mandam a chamada inteira, num pedaço só, quando chega. A chamada
  completa continua a vir no fim, como `tool_call`.

  Num flow iterado chegam como `llm_event`. Um pedaço conta como entregue, como o texto: depois
  dele, a chamada não se repete. Na app, um artefacto pode aparecer enquanto o modelo o escreve."
