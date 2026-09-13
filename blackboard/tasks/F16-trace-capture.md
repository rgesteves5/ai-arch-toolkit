# F16 · Política de captura do trace

- **Dono:** coordenador · **Estado:** done · **Depende de:** F08
- **Plano:** I, N4, N7, F16 · **Decisões:** D6

## Problema

Cada `StepTrace` guarda `input_state` (cópia rasa do snapshot) e `output_result` com os artefactos. O
JSON do trace cresce ao quadrado com os turnos (20 turnos com `raw` de 20 KB → 16 MB em 1.4 s), e as
mutações in-place dos steps (`tot`, `lats`) reescrevem traces anteriores.

## Mudança

`Flow(trace_capture="keys" | "full" | "none")`, por omissão `"keys"`: `input_state` vazio mais
`input_keys` por camada; `output_result` sem artefactos mais `output_keys`. `"full"` faz cópias
profundas (cópia rasa para valores que não se deixem copiar). `"none"` só metadados. `initial_state`
passa a cópia profunda (best effort) em `"keys"` e `"full"`.

## Prova

- Por omissão: `input_state == {}` e `input_keys["operational"]` lista as chaves.
- `"full"` com a `deque` de N4 → `[1, 2, 3]`, `[2, 3]`, `[3]`.
- React com 10 e 20 turnos: o JSON cresce de forma linear (razão < 2.5).
- `Trace.from_dict` aceita traces antigos sem os campos novos.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `core/_trace.py` (`TraceCapture`, `TRACE_CAPTURE_MODES`, `StepTrace.input_keys`
  e `output_keys` com `to_dict`/`from_dict`, `capture_state`, `capture_result`, `copy_state`),
  `core/_step_engine.py` (`execute_step(..., capture=)`), `toolkit/flow/_flow.py`
  (`Flow(trace_capture=)`, validação, propriedade), `toolkit/flow/_executor.py` (`initial_state`,
  `execute_step`, traces sintéticos de erro de orquestração, `flow_timeout` e `budget_exceeded`),
  as nove fábricas em `toolkit/agents/flows/` e os `react_flow` internos, `toolkit/agents/_builders.py`,
  `toolkit/agents/_spec.py`, `toolkit/agents/_manifest.py`, exports (`core/__init__.py`,
  `ai_arch_toolkit/__init__.py`), `docs/flow-architecture.md` ("What a trace captures"),
  `docs/safety.md`, `docs/agents.md`.
- Desvios ao plano:
  - `trace_capture` também em `ReasoningSpec`, nas fábricas, nos loops ReAct internos e no manifesto
    (`strategy.trace_capture`). Com o default `"keys"`, quem usa `Agent` ficaria sem forma de pedir
    `"full"`.
  - Em `"full"`, `world` fica por referência, como em `State.fork()`; valores que o `deepcopy` recusa
    também. O plano dizia cópia profunda do snapshot inteiro.
  - O input é capturado antes de o step correr. O motor serializava o snapshot depois, e em `"full"`
    isso gravaria o estado já mutado.
  - `"keys"` mantém `value` no `output_result`; só os artefactos passam a nomes.
  - Os traces sintéticos (erro de orquestração, `flow_timeout`, `budget_exceeded`) seguem a mesma
    captura. `FlowResult.results` continua com os artefactos completos.
- Testes novos: `tests/flow/test_trace_capture.py` (12; antes: 12 falhavam) e
  `tests/agents/test_agent_trace_capture.py` (14; sem a mudança em `_builders.py`: 11 falhavam).
- Tamanho do JSON do trace no react, 10 → 20 turnos (tool de 2 KB, `raw` de 2 KB):

  | modo | 10 turnos | 20 turnos | razão |
  |---|---|---|---|
  | antes | 849 800 | 2 961 714 | 3.49 |
  | `keys` | 62 636 | 123 261 | 1.97 |
  | `full` | 853 840 | 2 969 638 | 3.48 |
  | `none` | 9 413 | 17 614 | 1.87 |

- Verificações: 2901 passed, 7 skipped (os 2 avisos de `TestAstra` já existem no `HEAD`); ruff limpo;
  pyright 0 erros.
- CHANGELOG (Changed, quebra): por omissão o `StepTrace` guarda os nomes das chaves
  (`input_keys`, `output_keys`) e não os valores; `input_state` fica vazio e `output_result` sem
  artefactos. `Flow(trace_capture="full")` repõe os valores.
- CHANGELOG (Added): `Flow(trace_capture=)`, `TraceCapture`, `StepTrace.input_keys`/`output_keys`,
  `execute_step(capture=)`, `ReasoningSpec.trace_capture` e `strategy.trace_capture` nos manifestos.
- CHANGELOG (Fixed): o trace de um loop longo deixa de crescer com o quadrado dos passos; em
  `"full"` e no `initial_state`, mutações in-place de steps posteriores deixam de reescrever registos
  anteriores.
