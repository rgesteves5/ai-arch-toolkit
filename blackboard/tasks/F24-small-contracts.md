# F24 · Quatro contratos pequenos

- **Dono:** coordenador · **Estado:** done (2026-09-17) · **Depende de:** nada
- **Origem:** pedido do dono ("faz fix disto já"); achados de 2026-09-15 e 2026-09-17 em `FINDINGS.md`
- **Decisões:** D19

## O que estava mal e o que ficou

1. **Grupo de tools vazio tratado como ausente.** `tools or ToolGroup()` no `Agent` e `x_tools or tools`
   em cinco estratégias: um grupo vazio era trocado por outro, e uma fase declarada sem tools herdava
   as tools principais. Agora `is None` nos seis sítios.
2. **`LLM(fallback=outro)` alterava `outro`.** `_normalize_fallbacks` esvaziava a cadeia e a posse do
   `LLM` recebido. Agora achata sem tocar no que recebe, não repete um modelo alcançável duas vezes, e
   cada `LLM` só fecha os fallbacks que criou a partir de strings. Para o modelo não ser tentado duas
   vezes, o pai chama `fb._complete(..., follow_fallbacks=False)`; `complete()` público delega em
   `_complete(follow_fallbacks=True)`.
3. **`null` aceite em qualquer parâmetro.** A validação passa a ler da assinatura se o parâmetro
   admite `None` (default `None`, anotação com `None`, ou sem anotação utilizável); caso contrário dá
   `validation_error` antes dos gates. Reutiliza `_hint_to_json_schema` (uma só casa para "é opcional").
4. **`ReasoningSpec.from_mapping` descartava em silêncio.** Chave desconhecida, `policy` que não é
   `Policy` nem mapping, e `output_schema` malformado levantam `ValueError`; `policy` em mapping passa
   a construir um `Policy` (`retry` em mapping → `RetryConfig`; `fallback` não tem forma em mapping).

## Registo do dono

- Ficheiros tocados: `toolkit/agents/_agent.py`, `flows/_plan_execute.py`, `_reflexion.py`,
  `_llm_compiler.py`, `_lats.py`, `_self_discovery.py`, `toolkit/agents/_spec.py`, `core/_llm.py`,
  `core/_tools/_validation.py`; docs `agents.md`, `safety.md`; `CHANGELOG.md`.
- Testes novos (falhavam antes): `tests/agents/test_empty_tool_groups.py` (7, com um guarda por AST
  contra `x_tools or …` no pacote de agentes), `tests/test_llm_fallback.py` (+4),
  `tests/test_tools_validation.py` (+7), `tests/agents/test_configurable.py` (+8). Alterados:
  `test_nested_fallbacks_flattened` (afirmava a mutação) e o duplo `_FakeFallback` de
  `tests/test_admission_terminality.py` (passa a expor `_complete`).
- Verificações: 3070 passed, 22 deselected; ruff e formatação limpos; pyright 0.
- Fora do âmbito, registado em `FINDINGS.md`: as tentativas falhadas de fallbacks intermédios não
  ficam em `Response.attempts` no caminho `complete`.
