# R03 · Fase 3 — Tools, motor de flows e dívida de manutenção

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** R01 e R02 · **Regras:** `R00-rules.md`
- **Plano:** `docs/internal/hardening-plan.md` — causas 5 e 6, secções 9 e 10
- **Achados (`FINDINGS.md`):** "Tools: nenhum limite central", "`mediawiki_*`…", "`ip_lookup`…",
  "`math_eval("9**9**9")` não termina", "Leituras sem limite em linhas longas", "`list_directory`
  levanta…", "Imutabilidade só à superfície", "`step_end` em falta…", e a dívida confirmada das
  revisões externas (2026-09-17)
- **Decisões em vigor:** D4, D7, D17, D19, D20

## Objectivo

As tools passam a ser seguras por construção e não por convenção: um só ponto de saída HTTP, limites
de saída e de tempo impostos pelo executor a qualquer tool, capacidades declaradas e verificadas, e
invariantes executáveis que apanham a próxima tool mal feita. O motor de flows perde os caminhos
duplicados. E sai a dívida de manutenção que duas revisões externas confirmaram.

## Passos, por ordem

### 1. Tools — infra-estrutura comum

- **`toolkit/tools/_http.py`, o único sítio que toca em `urllib`, `http.client` ou `socket`:** só
  `https` (excepções explícitas e justificadas), host numa lista dada pelo módulo chamador, sem
  redirects para outro host, leitura com `max_bytes`, prazo total, segmentos de caminho com
  `quote(safe="")`, erros de rede, de charset e de forma do JSON devolvidos como string. Migra os 47
  sítios (35 helpers privados e 11 `urlopen` inline) nesta fase; não fica lista de legado.
  `prototypes/…/tool-invariants/hosts_by_tool.json` é a semente da lista de hosts.
- **Limites no executor governado (D17), para qualquer tool — do toolkit, de quem usa a framework,
  dinâmicas ou de MCP:** `ToolRuntimePolicy.max_output_chars` (por omissão 200 000) e `timeout_s` (por
  omissão 120; `None` desliga), com override em `@tool(...)` e no `ToolGroup`. O corte fica marcado no
  texto e em `ToolResult.metadata`; o timeout dá `ToolResult.failure("timeout")`. Uma tool síncrona
  numa thread não se mata: o executor deixa de esperar, e as tools de cálculo ganham guardas à entrada
  (`math_eval`: tamanho de expoentes e de resultados; `regex_search`: tamanho do padrão e do texto).
  Grampos nos parâmetros de tamanho (`max_chars=-1` e afins).
- **Metadados verdadeiros:** toda a tool declara `capability` (`network`, `filesystem`, `compute`, …).
  Um teste por AST confirma que o declarado bate com o que o código alcança
  (`prototypes/…/tool-invariants/ast_survey.py`). Tools com efeitos ou acesso local ficam em
  `dangerous` com os metadados de D4. `youtube_*` ficam como excepção declarada (extra `youtube`) mas
  cumprem os mesmos invariantes (nunca levantam, saída limitada).
- **Invariantes executáveis** em `tests/toolkit/test_tool_invariants.py`, com descoberta automática
  por `pkgutil` e `__tool_definition__`: partição de exports; capacidades; host fixo (argumentos
  hostis não mudam esquema, host nem caminho); nunca levanta, com argumentos hostis e com corpos de
  resposta hostis (sockets bloqueados, `time.sleep` substituído, `timeout(10)`); saída dentro do
  limite. `prototypes/…/tool-invariants/harness*.py` é o ponto de partida. No fim da fase não há
  `xfail`: as 12 tools que levantam com argumentos hostis e as 90 que levantam com corpos inesperados
  ficam corrigidas pela migração para `_http.py`.

### 2. Motor de flows

- `_run_dag` (`toolkit/flow/_executor.py`, a função mais complexa do motor): uma vaga de um step é
  uma vaga; fica um só caminho, com os checks de budget, prazo e erro de orquestração num só sítio e um
  único ponto de emissão de `step_end`.
- `_run_attempts` (`core/_step_engine.py`): o bloco de fallback escrito várias vezes passa a um.
- Verificador de gramática de eventos, reutilizado pelos testes do motor: cada `step_start` fecha com
  exactamente um evento terminal e o stream concorda com o trace, em todos os modos e causas de paragem.

### 3. Contrato de imutabilidade (D20)

`State.snapshot()` documenta-se como vista só de leitura; o mesmo step não pode comportar-se de forma
diferente em vaga paralela e em modo sequencial sem que isso esteja escrito e testado.
`ReasoningSpec.knobs` e `llm_kwargs` congelam à entrada (`MappingProxyType`). Sem cópias profundas por
step: trariam de volta o custo quadrático medido em N7.

### 4. Dívida de manutenção (secção 10 do plano), por esta ordem

1. `toolkit/memory/graph/_store.py` deixa de reimplementar a fachada de `core/graph/_store.py` (20
   métodos com o mesmo nome, 54 janelas duplicadas): compõe ou herda, e fica só com o que é da
   memória (embeddings, índice, ciclo de vida). Decide com uma nota de desenho; a API pública de
   `GraphStore` não muda.
2. Flow factories e builders: cada knob comum declara-se uma vez (hoje repete-se nas nove factories,
   dez vezes em `_builders.py`, no schema do manifesto e nas docs). `budget_policy` nas factories:
   decide se fica como caminho documentado ou sai, e regista.
3. Chaves de estado: as strings que as estratégias e o runner partilham (`"task"`, `"response"`,
   `"answer"`, `"last_answer"`, …) passam a constantes tipadas num só módulo, e `extract_text` deixa
   de adivinhar por quedas sucessivas.
4. Validadores de manifesto: os dois escritos à mão (agentes e prompts) e o JSON Schema à parte
   passam a ter uma só fonte de verdade por manifesto.
5. `_eval_expr` (`toolkit/tools/_python.py`): tabela de despacho por tipo de nó, com o conjunto
   permitido enumerável e um teste por nó permitido e por nó recusado.
6. O padrão "criar um ReAct interno, correr, extrair a resposta, inspeccionar o trace" repetido em
   seis estratégias passa a uma primitiva.
7. `__all__` de topo com 198 nomes: só um levantamento para o dono decidir antes da 1.0; não mudes
   exports.

### 5. Fecho da frente

Deriva de documentação (`AGENTS.md`, `CONTRIBUTING.md`, `docs/`), secção "Upgrade notes" no
`CHANGELOG.md` com todas as quebras das três fases, linha de base de complexidade actualizada, e o
`BOARD.md` a devolver a vez à frente C.

## Aceitação

- Regras de arquitectura verdes: `urllib`, `http.client` e `socket` só em `toolkit/tools/_http.py`;
  capacidades declaradas iguais às alcançadas; nenhuma tool com acesso local fora de `dangerous`.
- `tests/toolkit/test_tool_invariants.py` verde sem `xfail`; um resultado de 5 MB sai cortado ao
  limite; `math_eval("9**9**9")` e o regex com backtracking devolvem erro dentro do prazo.
- `_run_dag` e `_run_attempts` dentro do orçamento de complexidade; gramática de eventos verde.
- Duplicação entre os dois stores de grafo eliminada; `_eval_expr` dentro do orçamento.
- Saldo de linhas negativo em `toolkit/tools/` e nos módulos da dívida.

## Fora do âmbito

Tools novas (escrita tipada `C07`, pesquisa web `C08`), MCP (`C03`), tools dinâmicas (`C02`),
checkpoint (`C04`), `FlowSpec` (`C09`); mudar a API pública de `Flow`, `Agent` ou `GraphStore`.

## Registo do dono

- Estado: todo
- Notas de desenho:
- Ficheiros tocados:
- Testes novos e corrigidos:
- Verificações:
- Saldo de linhas e complexidade:
- CHANGELOG:
- Bloqueios:
- Desvios ao plano:
- Commits propostos:
