# A09 · Os workflows: o gasto de cada passo e as dependências fracas (G-32, G-33)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 9 (G-32, G-33) · **Decisões:** D58 ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `cf8df2f`)

- **G-32:**
  - o `execute_step` só abre um span do meter com `Policy.max_cost`, e o `StepTrace` não diz o
    que o meter mediu no passo;
  - `open_span`, `current_meter`, `current_span_id` e `bind_meter` só existem em
    `core/_metering/_scope.py`;
  - a app importa-os de lá (`structures/runs.py`, `agents/delegation.py`) para medir cada passo,
    cada corrida aninhada e cada delegação.
- **G-33:**
  - o `_check_skip_propagation` salta um passo quando qualquer dependência falhou ou foi saltada;
  - não há forma de juntar os caminhos depois de um `when`, nem de dizer que uma dependência é
    opcional;
  - a razão do salto já chega no `step_skipped` (a A05 deu-lhe o `step_trace`, com o
    `skip_reason`), mas só em texto. Para dizer que dependência saltou o passo, a app refaz a
    regra.

## Nota de desenho (antes do código)

- **G-32, no core:**
  - `open_span`, `current_meter`, `current_span_id` e `bind_meter` passam a sair de
    `ai_arch_toolkit.core`;
  - num flow medido, o `execute_step` abre sempre o span do passo. O `StepTrace.metered` (um
    `MeterSnapshot`) guarda o que o meter lá mediu: as tentativas, o fallback e os flows
    aninhados. Lê-se antes de o span fechar, e é `None` sem meter;
  - um passo que a corrida corta (o timeout, o budget, uma excepção) leva o que se mediu até ao
    corte. O `execute_step` dá a leitura a um `on_metered=` também quando é cancelado ou
    levanta;
  - o `MeterSnapshot` ganha `to_dict()`/`from_dict()`, para o trace serializado. O dinheiro vai
    em USD exactos, em texto.
- **G-33, no toolkit:**
  - o `FlowStep` ganha `after_any=` e `after_optional=`, no fim dos campos:
    - `after`: cada uma tem de ter corrido bem, como hoje;
    - `after_any`: o passo espera por todas, e corre se pelo menos uma correu bem;
    - `after_optional`: o passo espera por ela, e corre acabe ela como acabar;
  - cada nome só num dos três. Validam-se como o `after` (nomes e ciclos), e qualquer deles faz
    do flow um DAG;
  - o `StepTrace.blocked_by` diz que dependências saltaram o passo e como acabaram (`"failed"`
    ou `"skipped"`). Fica vazio quando foi a condição do próprio passo. O texto continua no
    `skip_reason`.
- **Provas:**
  - G-32:
    - cada passo de um flow medido traz no `metered` o seu custo e os seus tokens;
    - passos em paralelo não se misturam, e um flow aninhado conta no passo que o corre;
    - um passo cortado pelo timeout ou pelo budget traz o que gastou antes do corte;
    - sem meter, `metered` é `None`;
    - o trace serializado guarda o `metered` e relê-o;
    - `open_span` mede um bloco fora de um flow, como a delegação da app;
  - G-33:
    - depois de um `when`, o passo de junção corre com um ramo saltado e o outro bem;
    - sem nenhum ramo bem, é saltado, com o `blocked_by` e o `skip_reason`;
    - uma dependência opcional que falha não salta o passo;
    - a validação recusa um nome desconhecido, um nome em dois campos e um ciclo pelo
      `after_any`;
    - um flow só com `after_any` é um DAG.

## Ficheiros

- `src/ai_arch_toolkit/core/__init__.py`, `core/_step_engine.py`, `core/_trace.py`,
  `core/_metering/_admission.py`
- `src/ai_arch_toolkit/toolkit/flow/_flow.py`, `toolkit/flow/_executor.py`
- os testes novos em `tests/flow/` e `tests/metering/`
- `docs/flow-architecture.md`, `docs/api.md`, `AGENTS.md`, `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-05), pela nota de desenho (D58).
- **G-32:**
  - o `execute_step` abre o span do passo sempre que há meter, e lê o `metered` num `finally`
    dentro dele. O `on_metered` dá a leitura ao executor, que a põe no trace de um passo cortado
    (`_FlowRun._metered`);
  - ao abrir um span por passo, apareceu uma falha que já existia com o `max_cost`: uma chamada
    que o passo deixou a correr (`test_a_call_left_running_finishes_after_a_sync_iteration_has_closed_its_loop`)
    abria debaixo de um span já apagado e levantava `ValueError`. A correcção é de raiz: o id de
    um span é o caminho desde a raiz (`run/3/7`), e o `MeterStore._open_span_of` sobe até ao
    antepassado aberto mais próximo, sem guardar nada dos spans fechados. Um id que o store
    nunca deu continua recusado;
  - o `MeterSnapshot.to_dict()`/`from_dict()`, com o dinheiro em USD exactos (`Money.to_usd()`);
  - cada corrida aninhada abre também um span seu (`_FlowRun._close_meter`), e a sua entrada nos
    `children` do passo traz o gasto dela.
- **G-33:**
  - `FlowStep.after_any`, `after_optional` e `dependencies`. O `Flow.__init__` passou a validação
    dos nomes para `_validate_names` e a normalização para `_as_flow_step`; a dívida de
    complexidade dele saiu do `tests/quality_baseline.json`;
  - no executor, o `_blocked_by` e o `_skip_reason` substituem o `_check_skip_propagation`;
  - `StepTrace.blocked_by` e `DependencyOutcome`, exportado do core.
- **Revisão independente** (um agente sobre o diff): nenhum erro de correcção no executor, no
  motor ou na árvore de spans. Corrigido o que apontou:
  - o `_open_span_of` aceitava ids mal escritos (`run/01`, um dígito árabe-índico) e falhava com
    o `²`. Agora só aceita números como o store os escreve. A docstring diz que um id bem formado
    de outro store não se distingue;
  - o `Money.to_usd()` e o `from_usd()` arredondavam se a app baixasse a precisão do contexto
    decimal. Agora constroem e escalam sem arredondar;
  - as entradas das corridas que um passo corre ele próprio tinham `metered=None`. Agora têm o
    span delas;
  - um exemplo errado e o "sem `after`" na documentação e no `ValueError`; as quebras nas notas
    de actualização do CHANGELOG;
  - na D58, o custo real de um span por passo, medido, e os spans que ficam com uma operação
    viva.
- **Testes:**
  - `tests/flow/test_step_spend.py` (12) e `tests/flow/test_flow_dependencies.py` (14);
  - três em `tests/metering/test_spans.py`, e um em `tests/metering/test_money.py` (cinco
    casos).
- Gate: 6441 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 9 fechou.

  G-32: cada passo de um flow medido traz no `StepTrace.metered` (um `MeterSnapshot`) o que o
  meter mediu nele: as chamadas, as tentativas, o fallback e os flows que correu, também até ao
  corte de um passo cortado. Chega no `step_trace` do `step_end`. A entrada de uma corrida que um
  passo corre ele próprio, nos `children`, traz o gasto dela. `open_span`, `current_meter`,
  `current_span_id` e `bind_meter` saem de `ai_arch_toolkit.core`.

  G-33: `FlowStep(after_any=...)` corre se pelo menos uma das dependências correu bem (junta os
  caminhos depois de um *se*), e `FlowStep(after_optional=...)` espera por um passo sem o exigir.
  O `StepTrace.blocked_by` diz que dependências saltaram o passo e como acabaram (`"failed"` ou
  `"skipped"`), e o `skip_reason` di-lo em texto.

  Na app: os `_metering._scope` saem dos imports; o `measured` dos workflows pode ler o
  `metered` do `step_end`, em vez de abrir um span; o `why` lê o `blocked_by`; e os caminhos podem
  juntar-se depois de um *se* (a D-61 deixa de os separar)."
