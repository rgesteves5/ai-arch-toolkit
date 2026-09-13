# F22 · Revisão adversarial depois da implementação

- **Dono:** coordenador · **Estado:** done · **Depende de:** F01–F21 commitados (`5e29759`)
- **Origem:** pedido do dono ("revê o que fizeste"): três revisores independentes, só leitura, sobre o
  diff de `48a43ac` — providers/LLM/sync, tools/nanope, flow/trace/agents. Só achados confirmados
  com reprodução.

## Corrigido (commits de seguimento)

- `2411831` **tools**
  - uma string de inteiro com mais de 4300 dígitos fazia `ValueError` escapar de `execute()`;
  - `{"type": "null"}` aceitava qualquer valor;
  - `GateModify` com argumentos que não são mapping dava `AttributeError`;
  - `Literal[True]`/enums booleanos descritos como inteiros e recusados pelo validador
    (regressão do F13);
  - frase do `CHANGELOG` com um exemplo que agora é coagido.
- `7559d5e` **llm**
  - a reserva do stream usava o pedido de antes do `abefore` (admissão e preço errados quando o
    middleware muda tools, `max_tokens` ou conteúdo);
  - os fallbacks recebiam tools e kwargs de antes do middleware;
  - rejeição pelo middleware registava uma tentativa `StreamAbandoned`.
- `cc83cb9` **flow**
  - timeout numa onda paralela deitava fora os irmãos já terminados;
  - eventos contínuos adiavam o prazo;
  - `max_iterations=0` fazia uma passagem;
  - segundo cancelamento saltava `scope.close()`;
  - o evento `timeout` chegava antes de cancelar os steps;
  - `ReasoningSpec` não validava `trace_capture`;
  - overflow do backoff depois de ~1000 tentativas (anterior ao plano, em `_retry.py` e
    `_step_engine.py`);
  - `SyncFlowExecution` sem export nem `with`;
  - docs prometiam que `break` parava a execução.

## Provas

- Cada correcção tem teste que falha antes: `tests/test_tools_validation.py` (+11),
  `tests/test_tools_schema.py` (+1), `tests/test_stream_middleware.py` (+5),
  `tests/flow/test_engine.py` (+8), `tests/agents/test_agent_trace_capture.py` (+1),
  `tests/test_retry.py` (+1), `tests/test_step_engine.py` (+1).
- O teste do prazo passou 20/20 com a correcção e 3/20 sem ela.
- Suite: 2956 passed, 14 skipped; ruff e format limpos; pyright 0 erros; `live_api` sem xAI nem
  Gemini: 11 passed.

## Não corrigido (ver `FINDINGS.md`)

- `def f(q: int | None)` sem default: o schema diz opcional, a chamada sem `q` falha no binding.
- `ToolGroup(functools.partial(...))` falha com `AttributeError` em `infer_schema`.
- `ApprovalDecision.approve(modified_args={})` é ignorado (lista vazia é falsa).
- Coerção aceita `"1e3"`, `"1_000"`, `"5."` como inteiros (regras do `float()`); inofensivo.
- Não verificável offline: o Gemini aceitar, na mesma `Tool`, declarações com `parameters` e com
  `parameters_json_schema`.
