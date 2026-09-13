# F21 · Achados restantes do fecho

- **Dono:** coordenador · **Estado:** done · **Depende de:** F01–F20 aplicados
- **Origem:** `FINDINGS.md` (C4 nanope; worker-E conteúdo de sistema; worker-D `_stream_sync`;
  fecho: `Any` como texto e helpers de middleware mortos). Pedido do dono do repositório.

## Problemas e mudanças

1. **nanope sem handler de aprovação.** Com o F11, as sete tools perigosas exigem aprovação.
   - Agente configurável: `--allow-dangerous-tools` passava o `DangerousToolGate` mas o
     `ApprovalGate` recusava. Agora o grupo recebe `_approve_allowed_dangerous_tools`, que aprova só
     `DANGEROUS_TOOLS`; outras tools com aprovação continuam recusadas. Texto de ajuda da flag e
     README actualizados (a flag cobre as sete, não duas).
   - Solvers BBEH: `_solver_tools(*fns)` cria os grupos com `_approve_python_repl` (só `python_repl`,
     um avaliador AST restrito).
   - Encontrado agora: o `manager_agent` do `research_center` dá `http_get`/`scrape_text` ao modelo
     e o prompt manda usá-los; ficaram sempre recusados desde o F11. Agora aprova só essas duas.
2. **Conteúdo de sistema não-texto.** Helper `system_content_text` em `_providers/_base.py`: string,
   ou lista de strings e partes `cache()` unidas por linha em branco; imagem/documento dá `TypeError`
   a apontar para uma mensagem de utilizador. OpenAI, Gemini e xAI enviam texto; o Anthropic guarda o
   marcador de cache e envia blocos de texto quando há `cache()` (`_system_blocks`, `_merge_system`,
   `_with_system_suffix` para `json_mode` e o modo prompt de structured output; batch incluído).
   - Encontrado agora, mesma família: em mensagens de utilizador, OpenAI, Gemini e xAI enviavam o
     `repr` de `CachePart` ao modelo; o xAI enviava o `repr` de `DocumentPart` (com os bytes). Agora
     enviam o texto; o xAI larga documentos com aviso, como já fazia com imagens.
   - Com blocos nativos em `system=`, o Anthropic deixava cair o texto das mensagens `system()`
     (registado no F05); agora junta-o a seguir aos blocos.
3. **`_stream_sync` levantava excepções produzidas como valores.** Os erros da fonte vão num envelope
   privado `_SourceError`; só esses são levantados.
4. **`Any`/`object` descritos como `string`.** Passam ao schema vazio (qualquer valor JSON).
   Parâmetros sem anotação e tipos desconhecidos continuam `string`, para não partir tools que
   esperam texto.
5. **`_run_before`/`_run_after` mortos.** Removidos; os testes de ordem passam a usar os runners
   async, que são os que o `LLM` chama.

## Prova

- `tests/test_provider_system_content.py` (15; antes: 15 falhavam) e
  `tests/test_batch.py::TestAnthropicBatchSubmit::test_cached_message_system_keeps_its_cache_marker`.
- `tests/test_sync.py::TestStreamSync::test_an_exception_yielded_as_a_value_is_delivered_not_raised`
  (antes: `ValueError` levantado).
- `tests/test_tools_schema.py`: `Any`/`object`, `list[Any]`, `Any | None`, descrição preservada, os
  quatro `_tool_to_sdk` aceitam propriedades sem tipo (3 falhavam antes).
- `tests/test_middleware.py`: ordem dos `before` e ordem inversa dos `after` pelos runners async.
- `tests/nanope/test_advanced_configurable_agent.py` (2), `tests/nanope/test_bbeh_solvers.py` (2, com
  `inspect_ai` simulado), `tests/nanope/test_research_center_agents.py` (1). Antes: a execução com
  `allow_dangerous` e a busca do gestor falhavam; o helper dos solvers não existia.

## Registo

- Estado: done (2026-09-13)
- Verificação live (chaves do `.env`, uma chamada mínima por provider) de uma tool com propriedade
  sem tipo e itens sem tipo: `claude-haiku-4-5`, `gpt-4.1-nano` e `gemini-2.5-flash` (caminhos
  `parameters` e `parameters_json_schema`) aceitaram e o modelo enviou objectos e listas. O xAI não
  foi verificado: a conta respondeu 403 por falta de créditos.
- Teste alterado: `tests/test_anthropic_provider.py::…native_system_blocks…` fixava a perda do
  texto das mensagens `system()`; passa a exigir que chegue a seguir aos blocos.
- Ficheiros tocados: `core/_providers/_base.py`, `_openai.py`, `_gemini.py`, `_xai.py`,
  `_anthropic.py`; `core/_sync.py`; `core/_tools/_schema.py`, `_validation.py` (docstring);
  `core/_middleware.py`; `nanope/advanced_multi_purpose_configurable_agent/_tools.py`, `_chat.py`,
  `README.md`; `nanope/bbeh/_solvers.py`; `nanope/research_center/_agents.py`; `docs/llm.md`,
  `docs/content.md`, `docs/tools.md`; `CHANGELOG.md`.
- Verificações: 2927 passed, 7 skipped (os 2 avisos de `TestAstra` já existem no `HEAD`); ruff e
  format limpos (413 ficheiros); pyright 0 erros. Nenhuma tool do toolkit usa `Any`, por isso nenhum
  schema do catálogo muda.

## Revisão final e testes ao vivo (2026-09-13)

- Ao vivo, com as chaves do `.env` e modelos baratos (`claude-haiku-4-5`, `gpt-4.1-nano`,
  `gemini-2.5-flash`): usage igual em `complete`/`stream`/`stream_events` na Anthropic (F01);
  `system=` + `system()` fundidos nos três (F05); prompt de sistema com `cache()` escreve e lê a
  cache na Anthropic (14 595 tokens) e `count_tokens` aceita os blocos (F21); `json_mode` com blocos
  de cache aceite; `cache()` em conteúdo de utilizador chega como texto ao OpenAI; tools com `tuple`
  e `Any` chamadas e executadas pelo `run_tools` nos três; `prefixItems` e `$defs`/`$ref` na raiz
  aceites pelo Gemini via `parameters_json_schema` (F19 confirmado). Exemplos 01, 03–07, 09, 10,
  13, 14, 20–23, 32, 33, 36, 37 e 47 correram sem erros.
- Achado ao vivo e corrigido: um modelo pydantic aninhado como parâmetro de tool gerava
  `$defs` dentro da propriedade, com `#/$defs/...` a apontar para a raiz — o Gemini respondia
  `400 reference to undefined schema` (bug anterior ao plano; o F19 só o tornou visível).
  `_schema.py` passa a resolver as referências (`_inline_local_refs`) e a içar para a raiz os
  `$defs` que sobram em modelos recursivos (`_hoist_definitions`). Verificado ao vivo no Gemini
  pelo caminho `parameters`. Testes: `test_pydantic_model_references_are_inlined`,
  `test_recursive_model_definitions_are_hoisted_to_the_tool_root`.
- Achado ao vivo e corrigido: `gemini-2.5-flash` cortado por `max_tokens` enquanto pensa devolve
  `content.parts = None` e `_parse_sdk_response` rebentava com `TypeError`; passa a resposta vazia
  (o caminho de streaming já se protegia). Teste: `test_candidate_with_no_parts_is_an_empty_response`.
- Teste `live_api` já partido em `HEAD` (`tests/integration/test_e2e.py::test_tool_calling_round_trip`
  comparava um `ToolResult` com `"7"`); actualizado ao contrato e a passar.
- Novo `tests/integration/test_provider_contracts_live.py` (7 testes `live_api`, saltados sem
  chaves): fusão de prompts de sistema nos três providers, usage do streaming da Anthropic, cache do
  prompt de sistema na Anthropic, `prefixItems`/`$defs` no Gemini, `tuple`/`Any` pelo `run_tools`.
  `conftest.py` ganha `skip_no_anthropic` e `skip_no_gemini`.
- `cache()` em conteúdo de sistema e de utilizador confirmado como texto no Gemini
  (`gemini-2.5-flash-lite`, depois de a quota do `2.5-flash` se esgotar a meio).
- Não verificado ao vivo: xAI (conta sem créditos, 403).

