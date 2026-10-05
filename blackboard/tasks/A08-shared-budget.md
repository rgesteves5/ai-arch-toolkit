# A08 · O tecto partilhado: um orçamento que várias execuções gastam ao mesmo tempo (G-28)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 8 (G-28, a D-55 dele) · **Decisões:** D57 ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `86f2033`)

- Cada `MeterScope` tem o seu `MeterStore`, e o `BudgetController` admite sobre o snapshot da
  execução. Só os agentes aninhados partilham o scope.
- Duas execuções em paralelo não gastam do mesmo tecto. O ai-network dá a cada uma o que
  restava, e juntas passam dele.

## Nota de desenho (antes do código)

- **Core:**
  - `core/_metering/_shared.py` (novo): o `SharedMeter(limits, spent=Money, fail_on_unknown=True)`,
    com um `threading.Lock`, um `_Counters` semeado com `spent`, e `snapshot()`;
  - o `MeterStore` recebe `shared` e uma função de pior caso. No `open`, sem lock, a reserva
    passa a ser o maior entre a do controller e o pior caso, que se não houver nega. Já sob o lock
    da execução, a operação é verificada e reservada no partilhado, sob o lock dele;
  - cada transição (`mark_started`, `settle`, `fail`, `abort`) aplica-se também a ele;
  - o `RunConfig` ganha `shared`, e o `MeterScope` passa-o ao store;
  - o `request_facts` mede o pedido quando há tecto partilhado (o pior caso precisa do tamanho).
- **Toolkit:** o `SharedBudget(policy, spent=0.0)` é um `SharedMeter` feito de um `BudgetPolicy`,
  que recusa caps de tokens e de tempo. Tem um `report()` com o `BudgetReport` do gasto
  partilhado.
- **Provas:**
  - duas execuções com o mesmo tecto: com uma chamada de uma em curso, a outra é recusada se os
    dois piores casos passarem do tecto, e é admitida quando a primeira acerta;
  - a semente conta: com quase tudo gasto, a primeira chamada é recusada;
  - uma execução só a medir, ligada ao tecto, é limitada por ele;
  - muitas chamadas em paralelo, em duas execuções, nunca passam juntas do tecto;
  - o snapshot partilhado soma a semente e o que cada execução gastou;
  - uma chamada que falha deixa no partilhado o seu tecto (D49);
  - o `SharedBudget` recusa caps de tokens e de tempo.

## Ficheiros

- `src/ai_arch_toolkit/core/_metering/_shared.py` (novo), `_store.py`, `_scope.py`,
  `core/_metering/__init__.py`, `core/__init__.py`, `core/_attempts.py`
- `src/ai_arch_toolkit/toolkit/budget/_shared.py` (novo), `toolkit/budget/__init__.py`
- os testes novos em `tests/metering/` e `tests/budget/`
- `docs/safety.md` (os budgets), `docs/pricing.md` (se falar de budgets), `AGENTS.md`,
  `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-05), pela nota de desenho (D57; o resultado vem da D-55 do
  ai-network).
- **Core:**
  - o `SharedMeter` ficou no `_store.py`, e não num `_shared.py`, porque partilha os
    `_Counters` e as transições do store, que são privados;
  - o snapshot de um `_Counters` passou a função do módulo (`_snapshot`), usada pelos dois;
  - o `MeterStore` recebe `shared`. No `open`, o `_shared_hold` põe na reserva o pior caso (a
    função de tecto da D49, que o scope já dava), e sem preço recusa. Sob o lock da execução, o
    `SharedMeter._reserve` verifica e reserva sob o lock dele. O `_apply` aplica cada transição
    também ao partilhado;
  - `RunConfig(shared=...)`, `MeterScope.shared`. O `request_facts` mede o pedido sob um tecto
    de custo partilhado.
- **Toolkit:** `SharedBudget(policy, spent=0.0)` (recusa caps de tokens e de tempo, e uma
  semente negativa), com `report()`.
- **Testes novos (11)**, em `tests/budget/test_shared_budget.py`:
  - uma chamada em curso numa execução faz recusar a da outra, que é admitida depois do acerto;
  - a semente conta;
  - uma execução com budget próprio gasta dos dois;
  - duas execuções em tasks, e quatro em threads, nunca passam juntas do tecto, e no fim não fica
    reserva nenhuma;
  - uma falha deixa o seu tecto no partilhado;
  - o relatório;
  - três casos de tokens e de tempo, e a semente negativa.
- Gate: 6407 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 8 fechou: um
  `SharedBudget(BudgetPolicy(max_cost=...), spent=<o que o ledger já gastou>)`, ligado a cada
  execução com `config=RunConfig(controller=..., shared=budget)`. As execuções em paralelo são
  admitidas e acertadas contra ele sob um lock, e juntas não passam do tecto: cada operação
  reserva lá o seu pior caso.

  `budget.snapshot()` e `budget.report()` dão o gasto partilhado. Partilha o custo e o número de
  chamadas; o tempo e os tokens ficam por execução.

  Na app, sai o `budgets.policy_for` a dar a cada execução o que restava."
