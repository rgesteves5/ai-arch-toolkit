# F02 · `Flow.policy` por step e `Flow(timeout=)`

- **Dono:** coordenador · **Estado:** done · **Depende de:** —
- **Plano:** G, G-bis, F2 · **Decisões:** D1, D1a, D2

## Problema

`Flow(policy=)` só chega ao step engine quando o flow corre aninhado (`as_step()`). Corrido
directamente, é ignorado: `ReasoningSpec.timeout`, `ReasoningSpec.policy` e o `limits.timeout_seconds`
dos manifestos não fazem nada. As factories descartam `timeout` quando recebem `policy`.

## Mudanças

- `execute_step(step, snapshot, *, policy=None)`: policy efectiva `step.policy or policy or Policy()`.
- Executor: passa `flow.policy`; `_should_halt` usa `fs.step.policy or flow.policy`.
- `Flow(timeout=)`: positivo e finito; limita o run inteiro; ao expirar, pseudo-step `flow_timeout` com
  erro e decisão `timeout`; `iter()` emite `timeout`.
- `as_step()`: wrapper sem policy (D2).
- Nove factories e `_build_completion`: `Flow(policy=policy, timeout=timeout)`, sem a condição que
  engolia o timeout.
- Docs: `docs/agents.md` (policy por step, timeout do run, D1a), `docs/flow-architecture.md` (policy
  num `Flow`, frase sobre timeout nunca retentado).

## Prova

- Flow com `Policy(timeout=0.001)` corrido directo → `'Step timed out'`.
- `Step.policy` prevalece sobre `Flow.policy`; `on_exhausted="continue"` herdado não pára o flow.
- `Flow(timeout=0.05)` com dois steps de 100 ms → erro de timeout, o 2.º step não correu, sem ops
  vivas; `iter()` emite `timeout`.
- Flow aninhado com `retry=1` num step contador → 2 tentativas, não 4.
- `react_flow(policy=P, timeout=30)` → `flow.policy is P` e `flow.timeout == 30`, e o mesmo nas outras
  factories.
- `Agent(ReasoningSpec(timeout=0.001))` com provider de 50 ms → erro de timeout e `text == ""`; idem
  via manifesto com `limits.timeout_seconds`.
- D1a: reflexion com `policy=Policy(timeout=…)` e LLM lento → `attempt` expira; plan_execute com
  `timeout=…` → o envelope corta o flow interno em curso.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `core/_step_engine.py` (kwarg `policy`), `toolkit/flow/_flow.py` (`timeout`,
  `as_step` sem policy), `toolkit/flow/_executor.py` (policy do flow, prazo do run nos quatro
  caminhos, `flow_timeout`), as nove factories em `toolkit/agents/flows/`, `toolkit/agents/_builders.py`,
  `docs/agents.md`, `docs/flow-architecture.md`.
- Implementação: o prazo é um instante do loop; `_within_deadline` cria o awaitable só depois de o
  verificar e usa `asyncio.timeout_at` à volta de cada step (ou da vaga paralela inteira), nunca
  através de um `yield`. `_FlowTimeout` é interno e tratado só pelo flow dono do prazo. F08 substitui
  este mecanismo por esperas limitadas (D9).
- Testes novos: `tests/flow/test_flow_policy.py` (16), `tests/agents/test_agent_timeout.py` (18
  com a parametrização por estratégia). Antes da correcção: 20 falhavam pela razão certa.
- Verificações: 2642 passed, 7 skipped; ruff limpo; pyright 0 erros.
- CHANGELOG (Fixed): `Flow(policy=)` passa a ser a policy por omissão de cada step também quando o
  flow corre directamente; `ReasoningSpec.policy`, `ReasoningSpec.timeout` e `limits.timeout_seconds`
  deixam de ser inertes; as factories deixam de perder `timeout` quando recebem `policy`.
- CHANGELOG (Added): `Flow(timeout=)` limita o run inteiro e termina o trace com `flow_timeout`.
- CHANGELOG (Changed): o step que envolve um flow aninhado (`as_step()`) deixa de levar a policy do flow.
