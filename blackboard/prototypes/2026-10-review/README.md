# Protótipos · revisão completa do código (2026-10-11)

As sondas que provam os achados `RC01`–`RC74` de `blackboard/FINDINGS.md` (secção de 2026-10-11).
São evidência e ponto de partida, não código a importar: quem corrigir um achado transforma a
sonda num teste do repositório e apaga-a.

Correr a partir da raiz do repositório:

```bash
uv run python blackboard/prototypes/2026-10-review/toolkit/p6_as_step_pipeline.py
```

Nenhuma faz chamadas a fornecedores nem à rede: usam o `FakeProvider` (`tests/fake_provider.py`,
que as sondas encontram pela raiz do repositório), o `fakeserver` de loopback, o `prepare()` dos
adaptadores ou o `_http._open` simulado. `bound.py` e `p9_manifest.py` criam os seus ficheiros numa
pasta temporária (ou na que se lhes passar como argumento). A saída de cada uma mostra o defeito
enquanto ele existir; depois da correcção, deve mostrar o comportamento certo.

## `core/` — núcleo, metering e fornecedores

| Sonda | Achados |
|---|---|
| `probe_worst_case.py` | RC10: o pior caso reserva 4096 tokens de saída quando o adaptador manda 100000 (ou 14096). |
| `probe_no_usage.py` | RC11: um sucesso sem usage fica com custo ilimitado e bloqueia a chamada seguinte sob `max_cost`. |
| `probe_cross_loop.py` | RC12: `complete_sync()` dentro de um loop a correr usa o client do primeiro loop (servidor de loopback). |
| `probe_wire.py` | RC13 (A, B, C, E: documentos por URL ou `data:` URL), RC19 (D: `max_output_tokens` do Gemini), RC10 (F). |
| `probe_fallback_refusal.py` | RC14: um fallback que recusa no `prepare` acaba a cadeia. |
| `probe_tracing_image.py` | RC15: o `TracingMiddleware` perde o `image` do pedido. |
| `probe_step_timeout.py` | RC17: um `ProviderTimeout` passa por timeout do step (sem retry). |
| `probe_dropped_stream.py` | RC22: um stream largado guarda a reserva. |
| `probe_request_size.py` | Achado de 2026-09-30 ("Estimativa do `reserve="strict"`"): a imagem desenhada num turno do assistente conta como 0 partes de média. |

A RC16 (`StreamResponse` em O(n²)) foi medida sem sonda guardada: 40 000 chunks de JSON → 33,6 s.

## `toolkit/` — executor de tools, agentes, flow, budget, memória, recursos

| Sonda | Achados |
|---|---|
| `p6_as_step_pipeline.py` | RC01: num pipeline de dois agentes, o segundo faz 0 chamadas. |
| `p7_compiler_orphan.py` | RC02: o LLM Compiler devolve com uma chamada do executor ainda em curso. |
| `p12_nested_dag.py` | RC03: o resultado de um DAG aninhado depende da ordem de conclusão. |
| `p8_misc.py` | RC04 (ReAct corre tools no último turno), RC05 (`parse_score`), RC41 (selector Markdown). |
| `p9_manifest.py` | RC08 (`null` nos manifestos), RC09 (`system_file` com `://`; cria uma pasta `x:`, que o Windows não aceita). |
| `p14_provider_timeout.py` | RC17, pelo `plan_execute_flow`. |
| `p3_strict_pricer.py` | RC18: budget strict ignora o `RunConfig.pricer`. |
| `p1_json.py`, `p10_run_tools.py` | RC32: uma tool que devolve `datetime`/`set`/`date` rebenta o executor depois do efeito. |
| `p13_late_bill.py` | RC33: a conta tardia de uma tool paga não é cobrada. |
| `p11_group_in_list.py` | RC34: `run_tools`/`execute_tool` não vêem dentro de um grupo numa lista. |
| `p4_memory_image.py` | RC15: o `MemoryMiddleware` perde o `image` e põe memórias no sistema. |
| `p5_memory_search.py` | RC38: a pesquisa vectorial filtrada por tipo devolve vazio. |
| `p2_pagerank.py` | RC40: o `pagerank` soma 0,508 com arestas paralelas (a comparação com o do networkx precisa de numpy). |

## `tools/` — tools do toolkit

| Sonda | Achados |
|---|---|
| `table_bomb.py` | RC46: tabelas aninhadas multiplicam o texto do `wiki_text` (×8 199 medido). |
| `retry_after.py` | RC47: `Retry-After: inf` faz o host descansar para sempre e dá `OverflowError`. |
| `nonascii.py` | RC48: um URL com caracteres não ASCII dá `UnicodeEncodeError`. |
| `findwin.py` | RC51: o custo do `find_window` cresce com o texto vezes a janela. |
| `misc1.py` | RC49 (antípodas), RC52 (`"nm"`), RC53 (`python_repl`), RC54 (`math_eval`), RC55 (`_human_size`), RC56 (`json_extract`). |
| `tilde.py` | `~utilizador` cru (achado de 2026-10-04) e o `csv_read` da RC61. |
| `bound.py` | Prova negativa: o `FilesystemPolicy` recusa symlinks, `..`, caminhos absolutos e moves para fora da raiz. |
| `xmlbomb.py` | Prova negativa: o expat 2.7.4 recusa a bomba de entidades XML. |
