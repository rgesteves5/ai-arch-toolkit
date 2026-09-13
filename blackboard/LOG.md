# Diário

## 2026-09-13

- Revisão `docs/internal/agentes-app-toolkit-review.md` verificada por execução; plano escrito em
  `docs/internal/toolkit-fix-plan.md`.
- Revisão cruzada de outro modelo integrada no plano (secção 0): D1a, D3 e D7 revistos; D8 e D9 novos.
- Confirmados por execução N8, N9 e N10 (ver `FINDINGS.md`).
- Baseline em `main` @ `48a43ac`: 2608 passed, 7 skipped (11 s); pyright 0 erros; ruff limpo.
- Blackboard criado. Próximo: workers A–E nas worktrees (F01, F04/F06/F07/F17, F11/F12, F15,
  F05/F10); coordenador começa F02, depois F09 e F14.
- F02, F09, F14 feitos pelo coordenador (2650 passed). F08 implementado no checkout principal: motor
  único (gerador + task por step), `FlowExecution`/`AgentExecution`, eventos de policy, children,
  irmãos preservados, span em flows aninhados (2691 passed, testes do motor estáveis em 3 repetições).
- Workers A, B, C e E terminaram. Diffs revistos e aplicados sem conflitos: F01, F04, F05, F06, F07,
  F10, F11, F12, F17. Suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos. Um teste do F12
  dependia da ordem do cache do `typing` em 3.13 e foi corrigido.
- Workers E e C não conseguiram escrever no checkout principal (a harness recusou); registos
  transcritos pelo coordenador. Achados novos: gates não encadeiam `GateModify` (worker-B) e C2/C3
  entram no F13; C1 vira F19; `run_tools` com `KeyError` depois de efeitos vira F20.
- Falta: F03, F13, F15 (worker-D em curso), F16, F19, F20, F18, e a documentação do F08.
- F15 (worker-D) aplicado: wrappers síncronos cancelam no timeout e têm backpressure (2867 passed).
- F03 e F13 feitos pelo coordenador; ver os registos nas fichas (D8 mudou quatro testes de metering).
- F20 feito: `run_tools` verifica os nomes antes de executar. F19 feito: `_tool_to_sdk` recorre a
  `parameters_json_schema` quando o SDK recusa o schema.
- F16 feito: `Flow(trace_capture=)`, default `"keys"`; ligado a `ReasoningSpec` e manifesto
  (2901 passed, 7 skipped).
- F18 feito: docs (`tools.md`, `api.md`, `memory.md`, docstring do `RateLimitMiddleware`) e
  `CHANGELOG` `[Unreleased]` com F01–F20.
- Plano completo. Verificação final: 2901 passed, 7 skipped; ruff e format limpos; pyright 0 erros.
  Nada commitado. Worktrees dos workers A–E ficam em `.claude/worktrees/agent-*` para o dono remover.
- F21 feita a pedido do dono: aprovações no nanope (agente configurável, solvers BBEH e o gestor do
  `research_center`, regressão do F11 que nenhum teste apanhava), conteúdo de sistema com partes de
  texto nos quatro adaptadores e `cache()` como texto fora do Anthropic, envelope de erros no
  `_stream_sync`, `Any`/`object` com schema vazio (verificado live em Anthropic, OpenAI e Gemini; xAI
  sem créditos), helpers de middleware removidos. 2927 passed, 7 skipped; ruff e pyright limpos.
- Revisão final a pedido do dono: três revisores independentes sobre o diff; testes ao vivo em
  Anthropic, OpenAI e Gemini (ver F21); dois bugs encontrados ao vivo e corrigidos (`$defs` de
  pydantic aninhados; Gemini sem `parts`); teste `live_api` antigo actualizado; 7 testes `live_api`
  novos. xAI fica pendente (créditos).

