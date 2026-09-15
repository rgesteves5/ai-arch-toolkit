# C05 · Server tools: config no wire e `server_tools=` nas estratégias

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (acerta `server_tools` com C06)
- **Origem:** `docs/internal/agentes-app-toolkit-review.md` (L13, "O que escapou" §10, ponto D,
  dívida 13); `docs/internal/toolkit-fix-plan.md` §4 itens 3 e 10
- **Decisões:** por fixar (ver abaixo)

## Problema

Verificado em `main` @ `7ebf7ef` com clientes SDK simulados, sem rede (caminhos em
`src/ai_arch_toolkit/`), e na doc oficial lida em 2026-09-15:
- Anthropic `platform.claude.com/docs/en/`: `agents-and-tools/tool-use/web-search-tool`,
  `…/server-tools`, `…/code-execution-tool`, `…/tool-reference`, `about-claude/pricing`.
- OpenAI `developers.openai.com/api/docs/`: `guides/tools-web-search`, `pricing`.
- Gemini `ai.google.dev/gemini-api/docs/`: `generate-content/google-search`,
  `generate-content/tool-combination`, `pricing`; `discuss.ai.google.dev/t/174074`.
- Meta `dev.meta.ai/docs/`: `search-grounding.md`, `pricing-rate-limits.md`; xAI
  `docs.x.ai/developers/cost-tracking`.

1. **Config descartada.** TODO em `core/_server_tools.py:16-19`; `prepare_tools` espalha a config
   (`core/_tools/__init__.py:110-111`), os adaptadores só lêem `st["type"]`, não há o aviso de §4.3
   e `web_search(max_results=5)` passa. `web_search(max_uses=3, allowed_domains=[…])` +
   `code_execution()` saem assim:
   ```
   _anthropic.py:529-537  [{"type": "web_search_20250305"}, {"type": "code_execution_20250522"}]
   _openai.py:485-495     [{"type": "web_search"}, {"type": "code_interpreter"}]
   _gemini.py:477-492     [{"google_search": {}}, {"code_execution": {}}]
   _meta.py:260-270       [{"type": "web_search"}]  + UserWarning (code_execution descartado)
   _xai.py:404-419        NotImplementedError
   ```
2. **Anthropic.** Falta `name` (`Required` no SDK 0.116): 400 esperado, a confirmar ao vivo;
   `code_execution_20250522` é legado. `_parse_sdk_response` (`:351-406`) ignora os blocos de
   servidor e cola o preâmbulo (`"I'll search.Claude…"`); `usage.server_tool_use` perde-se
   (`:317-327`); `_messages_to_sdk` (`:182-235`) não reenvia os blocos que a doc manda devolver
   verbatim; `pause_turn` não é tratado.
3. **OpenAI: mapeamento partido.** No SDK `openai` 2.45, `tools` do Chat Completions só aceita
   `function`/`custom`; a doc: "The Chat Completions API supports only specialized search models
   for web search". `code_interpreter` é da Responses; o exemplo 25 usa `gpt-4.1-nano`.
4. **Gemini.** Sem `max_uses` nem domínios (`exclude_domains` é só Vertex); com function calling
   pede `include_server_side_tool_invocations=True` (Gemini 3, preview), que o adaptador não põe;
   `web_search_queries` falta quando o modelo pesquisa a raciocinar (fórum Google, 2026-07-08).
5. **Custo.** `has_server_tools` (`core/_llm.py:487-491`) dá `unknown` (`core/_pricing.py:239`) e
   o controller nega a própria chamada (`toolkit/budget/_controller.py:102-114`): com
   `BudgetPolicy(max_cost=5.0)`, `web_search()` → `BudgetExceeded`, 0 chamadas (`docs/safety.md:275`
   diz "depois"). `Response.cost` fica só com tokens; no Meta, `code_execution()` nem sai e é
   `unknown`. Achado: um erro do adaptador ao montar o pedido já conta como chamada `unknown`.
6. **Estratégias (ponto D).** Tools só chegam ao LLM em `react_flow` (`flows/_react.py:80-88`), como
   `ToolGroup`; `ReasoningSpec` não tem campo; `llm_kwargs={"tools": […]}` dá "got multiple values
   for keyword argument 'tools'"; o manifesto só tem `tools.factory`/`manifest` (`_manifest.py:97`).

## Objectivo

Server tools declaradas chegam ao fornecedor com as restrições pedidas ou falham antes de gastar,
têm o custo por uso medido quando reportado e juntam-se ao `ToolGroup` nas estratégias com tools.

## API proposta

```python
ws = web_search(max_uses=3, allowed_domains=("docs.python.org",))
agent = Agent(ReasoningSpec(strategy="react", server_tools=(ws,)), llm, ToolGroup(read_note))
```

- **Tipos.** `ServerTool(type, config)` mantém o fio e rejeita chaves reservadas; `web_search(*,
  max_uses, allowed_domains, blocked_domains, user_location: UserLocation)` valida (`max_uses >= 1`,
  listas exclusivas, domínio ASCII sem esquema nem `*` no host); `from_dict`/`to_dict`.
- **Fio.** O `LLM` chama `provider.check_server_tools(wire_tools)` após o middleware e antes do
  metering (complete, streams, fallbacks; no-op em `BaseProvider`). Anthropic aplica tudo, com
  `name`; Gemini só sem config; Meta só `user_location`; o resto e o OpenAI → `ValueError`; o xAI
  mantém `NotImplementedError`, agora antes do meter.
- **Resposta.** Server tool calls nunca vão para `Response.tool_calls`; `Usage.server_tool_uses` com
  os usos reportados (Anthropic `server_tool_use`; Gemini 3 `len(web_search_queries)`, 2.5 um por
  prompt; Meta só depois do probe). Anthropic junta com linha em branco o texto separado por blocos
  de servidor e reenvia `_raw.content` verbatim enquanto coincide com texto e tool calls (como D12).
- **Custo.** TOML por modelo `server_tools = { web_search = 10.0 }` (USD por 1 000 usos) →
  `ModelPricing.server_tools`; `price()` soma `tarifa × usos / 1000`; sem tarifa ou sem usos
  reportados → `unknown` com motivo. `OperationRequest.server_tools` (tipo → `max_uses`); com
  `fail_closed` o controller só nega sem tarifa ou, em `reserve="strict"`, sem `max_uses`.
- **Governança.** Só canais declarados (`tools=[…]`, a spec, `tools.server` no manifesto, já coberto
  por `override_policy.deny: [tools]`); restrição não aplicável → erro; auditoria em
  `UsageEvent.metadata["server_tools"]` e nos usos; veto por pedido com `Middleware.abefore`.
- **Estratégias.** `FlowStrategy.supports_server_tools` (`rewoo`/`tot` rejeitam). `react` pede com
  `[group, *server_tools]`, mantém-nas em `strip_tools_on_final`, com server tools põe contador e
  aviso final no `system` (após turno misto só pode vir `tool_result`) e continua em `pause_turn`;
  `completion` idem; as factories com react interno recebem `server_tools=` e passam-no.

## Decisões a fixar antes de codificar

1. **Config:** (a) kwargs tipados e erro no que o fornecedor não aplica; (b) `**config` livre; (c)
   (a) com aviso (§4.3). Recomendo (a), porque uma allowlist descartada é um buraco de governança.
2. **Canal:** (a) `ReasoningSpec.server_tools` + `tools.server`; (b) `Agent(server_tools=)` + deps;
   (c) `ToolSet`. Recomendo (a): são dados serializáveis (fronteira knobs/deps), chegam à spec sem
   mexer em `agent_from_manifest`, e (c) muda o tipo de `tools` nas nove factories e no `Agent`.
3. **Custo:** (a) usos em `Usage` + tarifa por modelo, `unknown` se o uso não vier; (b) `Pricer`
   recebe a `Response`; (c) uso em falta conta zero. Recomendo (a), porque o protocolo `Pricer` não
   muda, o preço varia por modelo e (c) subcontaria em silêncio quando a Gemini omite a contagem.
4. **Governança:** (a) declaração + validação + auditoria + middleware; (b) `ServerToolPolicy`.
   Recomendo (a), porque não há aprovação por chamada e a app aprova a capacidade no agente.
5. **Versões Anthropic:** (a) `web_search_20250305` + `code_execution_20250825`; (b) as mais
   recentes; (c) por modelo via C06. Recomendo (a) e depois (c): (a) serve todos os modelos e é
   elegível para ZDR; (b) exige Claude 4.6+ ou `allowed_callers: ["direct"]`.
   *Nota do coordenador (referência da Claude API, 2026-09-15):* os tipos actuais são
   `web_search_20260209` (Opus 5/4.8/4.7/4.6, Sonnet 5 e 4.6; o básico `web_search_20250305` fica para
   modelos anteriores e é o único no Vertex) e `code_execution_20260521`; o `20260120` é o de REPL e
   programmatic tool calling. O `web_search_20260209` já corre code execution por baixo e não deve ir
   com `code_execution` declarado no mesmo pedido. Confirmar a versão de code execution antes de fixar (a).
6. **OpenAI:** (a) `ValueError`; (b) `web_search_options` nos modelos `*-search-*`; (c) Responses.
   Recomendo (a), porque D11 mantém Chat Completions e (b) não tem domínios nem `max_uses`.
7. **`pause_turn`:** (a) nas estratégias; (b) ciclo em `LLM.complete`. Recomendo (a), porque o
   `core` fica sem estado e cada continuação é uma operação medida.

## Sub-tarefas, por ordem

- **C05a** (core): tipos e exports → `check_server_tools` antes do metering → Anthropic (fio,
  parse, texto, usos em complete e stream, reenvio de `_raw`) → Gemini, Meta, OpenAI, xAI → custo
  (`Usage`, `ModelPricing` + TOML, `price()`, `OperationRequest`, estimator, controller, evento).
- **C05b** (toolkit): `ReasoningSpec.server_tools` e `build_flow` → `react_flow` e `completion` →
  factories com react interno → `tools.server` e `ai-arch agent validate` → docs, exemplo, probes.

## Ficheiros

- `core/_server_tools.py`, `core/_metering/_operation.py`, `core/_pricing.py`,
  `core/_default_pricing.toml`; `core/_tools/{__init__,_group}.py` (com C02); `core/_llm.py` (com
  C01, C06); `core/_response.py` (com C01);
  `core/_providers/{_base,_anthropic,_gemini,_meta,_openai,_xai}.py` (com C01 e C06)
- `toolkit/budget/_estimator.py` (com C08), `_controller.py`, `_policy.py`;
  `toolkit/agents/_spec.py`, `_compile.py`, `_manifest.py` (com C04), `_builders.py` e os nove
  `flows/_*.py` (com C01; cinco com C02, `_lats.py` com C04); `_cli.py`; exports `core/__init__.py`,
  `__init__.py` (com todas); se C06 entrar antes, `core/_default_catalog.toml` e
  `tests/test_model_catalog_data.py` (com C06)
- Testes em `tests/`: `test_server_tools.py`, `test_stream_middleware.py`, `test_pricing.py`,
  `test_cli_agent.py`, `test_llm_metering.py` (com C01), `test_core_exports.py` (com C02, C04, C06),
  `test_{anthropic,gemini,openai,meta,xai}_provider.py` (com C06), `metering/test_pricer.py`,
  `budget/test_budget.py` (com C08), `agents/flows/test_*_flow.py`, `agents/test_manifest.py`,
  `agents/test_configurable.py` e `integration/test_provider_contracts_live.py` (com C02)
- Docs: `docs/tools.md` (com C02–C04, C07, C08), `docs/agents.md` (com C01, C02, C04),
  `docs/safety.md` (com C02, C04, C07, C08), `docs/llm.md` (com C01, C06), `docs/pricing.md` e
  `docs/model-compatibility.md` (com C06), `docs/configuring-agents.md`,
  `examples/25_server_tools.py`, `examples/README.md` (com C03, C08)

## Prova

- **Fio.** Anthropic `web_search(max_uses=3, allowed_domains=["example.com"])` → `{"type":
  "web_search_20250305", "name": "web_search", "max_uses": 3, "allowed_domains": ["example.com"]}`;
  listas exclusivas, `"https://x.com"`, `"*.x.com"`, `max_results=5` → erro; Gemini `max_uses=3`,
  Meta `code_execution()`, OpenAI `web_search()` → `ValueError` sem chamar o cliente e snapshot com
  `llm_calls == 0`, `unknown_cost_count == 0`; texto `"I'll search.\n\nClaude…"`.
- **Custo.** Anthropic com `web_search_requests=2` → evento `known` = tokens + `$0.02`, igual em
  `Response.cost` e `stream_events`; `BudgetPolicy(max_cost=5.0)` + `web_search()` → chamada feita;
  `reserve="strict"` sem `max_uses` → negada, com `max_uses=3` → a reserva inclui `$0.03`; Gemini
  sem `web_search_queries` → `unknown`.
- **Estratégias.** `Agent(ReasoningSpec(server_tools=(ws,)), llm, ToolGroup(read_note))` → o
  fornecedor recebe as duas; só server tool → fim sem `tool_results`; o pedido seguinte leva os
  blocos com `encrypted_content`; `pause_turn` → nova chamada; turno misto → só `tool_result` a
  seguir, mesmo com `show_turn_counter`; `strip_tools_on_final` mantém a server tool; `rewoo`/`tot`
  → `ValueError`; `tools.server` → `reasoning_spec()`; override negado → `AgentOverrideError`.
- **`live_api`, só local.** Anthropic sem `name` (confirma o 400) e com `max_uses=1`; turno misto
  via `Agent`; Gemini 3 `google_search` + function tool; Meta `web_search_call`; OpenAI (400).

## Fora do âmbito

- Adaptador OpenAI Responses; server tools do xAI (o `xai-sdk` 1.17 já tem `xai_sdk.tools`, com
  custo exacto em `cost_in_usd_ticks`); tipos novos (`web_fetch`, `url_context`, `x_search`);
  parâmetros específicos (`search_context_size`, `allowed_callers`); filtragem dinâmica (após C06).
- Server tools por fase no manifesto; eventos de stream (C01); quotas gratuitas (o preço de tabela
  sobrestima); tarifa de code execution Anthropic (por hora de contentor); batch.

## Riscos

- O reenvio verbatim engorda o histórico (resultados cobrados de novo como input) e põe mais
  objectos SDK em `_raw` a serializar (partilhado com C04).
- Turno misto Anthropic: texto depois dos `tool_result`, ou tool ausente no pedido seguinte → 400;
  Gemini 3: a flag está em preview e força `VALIDATED` (sem `AUTO`), em choque com `tool_choice`.
- Quebras (*Changed*): `web_search` sem chaves livres; Meta `code_execution` e OpenAI levantam;
  campos novos em `Usage`/`OperationRequest`; fallback que não aplica a config falha. Restrições de
  domínio da organização Anthropic só falham no servidor; `lats` multiplica pedidos.

## Registo do dono

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
