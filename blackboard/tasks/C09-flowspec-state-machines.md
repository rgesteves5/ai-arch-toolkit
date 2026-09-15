# C09 · FlowSpec e máquinas de estados declarativas

- **Dono:** por atribuir · **Estado:** blocked · **Depende de:** C04a, C04b; formas validadas na app
- **Origem:** `agentes-app-toolkit-review.md` L3 (`:246-266`), L4 (`:268-281`), prioridade 5 (`:515`)
- **Decisões:** por fixar (ver abaixo)
- **Desbloqueio:** o dono traz da app, já em uso, três workflows e duas máquinas reais como dados (refs,
  parâmetros, policy por nó); as falhas vistas (ref desconhecida, ciclo, condição que levanta, estado
  sem saída, limite, ramos paralelos); como muda um spec com runs parados; o que a UI mostra do caminho.

## Problema

- `Flow` só guarda callables (`FlowStep.step`/`when`/`scope`, `toolkit/flow/_flow.py:35-42`), sem
  `to_dict`; não há transição, prioridade, estado terminal nem caminho. `max_iterations` conta
  passagens e esgotá-lo acaba o run em silêncio (`toolkit/flow/_executor.py:331-332`).
- Repro: `start → review → done`, um step por estado e `max_iterations=1` → 3 transições numa
  passagem; com a aresta `review → start` pára em `start`, não terminal, sem erro. Já resolvido: um
  `when` que levanta no modo cíclico pára com erro (F08 8a, `_executor.py:339-347`).

## Objectivo

Workflows e máquinas de estados como dados versionados, compilados para `Flow` pelo registo da app.

## Pré-requisitos no toolkit

- C04a/C04b: `json_fingerprint()` e `Flow(version=)`. O fingerprint do C04 fica sobre a estrutura
  declarada (nomes, `after`, condicional, versão), nunca sobre callables, para o mesmo spec compilado
  dar o mesmo fingerprint em qualquer processo (`version=f"{spec.id}@{spec.version}"`); o cursor fica
  por passagem e o `CheckpointMismatchError` traz as diferenças, para ligar versões a migrações.

## API proposta

```python
NodeSpec(id, ref, after=(), condition_ref=None, parameters={}, policy=None)  # step|agent|workflow
FlowSpec(id, version, nodes, max_iterations=None)                            # to_dict/from_dict
compile_flow(spec, registry) -> Flow     # ref desconhecida, ciclo, id repetido → FlowSpecError
TransitionSpec(target, condition_ref=None, priority=0)                       # maior primeiro
StateMachineSpec(id, version, initial, states, terminal_states, max_transitions)
compile_state_machine(spec, actions, conditions) -> Flow
```

## Decisões a fixar antes de codificar

1. **Onde vive.** (a) app; (b) `toolkit.flow`; (c) estratégia. Recomendo (a), e (b) no que se repetir.
2. **Compilação.** (a) um step por estado com `when`; (b) um step de despacho que corre a acção do
   estado actual, escolhe a primeira transição válida e grava `machine_state`/`machine_path`, com erro
   sem transição fora de terminal ou no limite. Recomendo (b): (a) faz várias transições numa passagem.
3. **Caminho e refs.** Caminho no estado ou em `FlowResult`: recomendo o estado, que entra no
   checkpoint sem API nova. Refs (strings ou tipos) e trace ao esgotar o limite: decidir com casos reais.

## Sub-tarefas, por ordem

- Primeiro corte: **C09a** `FlowSpec`/`compile_flow`; **C09b** `compile_state_machine`; **C09c** retoma.

## Ficheiros

- `toolkit/flow/_spec.py`, `_machine.py` (novos), `__init__.py` (partilhado com C04);
  `tests/flow/test_spec.py`, `test_machine.py`; `docs/flow-architecture.md` (partilhado com C01, C04).

## Prova

- `max_transitions=1` faz uma transição e dá erro fora de terminal; `from_dict(to_dict())` igual; ref
  desconhecida → `FlowSpecError` antes de correr; mesmo spec em dois processos → mesmo fingerprint.

## Fora do âmbito

- Na app: DSL, editor, biblioteca, conteúdo do registo, versões e migrações, permissões, persistência,
  agendador, UI e resolver `agent_ref` (sem `agent_as_tool`). Nunca serializar callables.

## Riscos

- Fixar formas antes dos casos reais cria API que a app contorna: por isso a ficha fica `blocked`.

## Registo do dono

- Estado: blocked
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
