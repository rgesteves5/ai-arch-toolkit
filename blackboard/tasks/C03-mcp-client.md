# C03 · Cliente MCP

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** C02
- **Origem:** `agentes-app-toolkit-review.md` L1 e "Respostas às perguntas de intenção" 1 (item 3);
  `toolkit-fix-plan.md` §4.10
- **Decisões:** por fixar (ver abaixo)

## Problema

Não há cliente, transporte nem conversão MCP em `src/`, nem extra `mcp` (`pyproject.toml:16-72`). Ligar
o SDK oficial à mão tem armadilhas, verificadas com `mcp` 2.2.0, Python 3.13 e asyncio (scripts no
scratchpad; servidores em processo, stdio e ASGI):

- Fechar o `mcp.Client` noutra task: `RuntimeError: Attempted to exit cancel scope in a different task
  than it was entered in`. A app (registo de servidores) e o `Flow` (D9) usam tasks diferentes.
- `group.execute()` síncrono numa tool MCP por stdio (`read_timeout_seconds=5`): `runtime_error:
  Request 'tools/call' timed out` ao fim de 10,01 s, porque o caminho síncrono usa outro loop
  (`_executor.py:86-98`, `docs/tools.md:94`); sem timeout espera o `sync_timeout` (300 s).
- O `Client` não tem timeout por omissão. Timeout → `MCPError(-32001)`; servidor morto → `MCPError(-32000,
  "Connection closed")`; comando inexistente → `FileNotFoundError`; HTTP inacessível → `ExceptionGroup`.
- Resultados: `ToolError` e tool desconhecida → `is_error=True` com texto; outra excepção → só "Error
  executing tool crashes"; imagens em `ImageContent` base64; escalares em `{"result": 5.0}`. O toolkit só
  devolve texto ao modelo (`_result.py:69-78`, `_react.py:143`, `_openai.py:163-169`, `_xai.py:85`).
- Mudanças de tools: na era 2026-07-28 só chegam por `client.listen(tools_list_changed=True)`; na
  2025-11-25 chegam ao `message_handler` e `listen` levanta `ListenNotSupportedError`.

Fontes (2026-09-15): PyPI `mcp` 2.2.0 (2026-09-07, Python ≥ 3.10; 2.0.0 a 2026-07-28; 1.x em
manutenção, 1.30.0). `github.com/modelcontextprotocol/python-sdk` @ `v2.2.0`: `client/client.py`
(`Client`, `call_tool`, `list_tools(cursor=, cache_mode=)`, `listen`), `shared/jsonrpc_dispatcher.py`
(timeout e `notifications/cancelled`), `client/stdio.py` (ambiente só `HOME LOGNAME PATH SHELL TERM USER`
mais `env=`; `open_process([command, *args])` sem shell), `docs/client/transports.md` (headers e OAuth
num `httpx2.AsyncClient`), `docs/get-started/testing.md` (`Client(server)` em memória). Spec
`modelcontextprotocol.io/specification/2026-07-28/server/tools`: "clients MUST consider tool annotations
to be untrusted unless they come from trusted servers"; prefixar nomes ao agregar, sem `serverInfo.name`;
timeouts. `uv lock` numa cópia com o extra resolve: +9 pacotes, nenhuma versão existente muda.

## Objectivo

`toolkit.mcp` liga a um servidor MCP por stdio ou streamable HTTP com o SDK oficial (extra `mcp`) e
converte as tools, com C02, em tools governadas: nomes portáteis, risco alto com aprovação por omissão,
validação, gates, `max_calls`, metering e `ToolResult`. A ligação vive numa task própria, fecha de
qualquer task e avisa quando as tools mudam. Listas brancas, credenciais, política e UI ficam na app.

## API proposta

```python
github = StdioTransport(command="npx", args=("-y", "@modelcontextprotocol/server-github"),
                        env={"GITHUB_TOKEN": token})          # env e headers fora do repr
async with MCPClient(github, name="github", timeout=60.0, on_tools_changed=refresh) as mcp:
    offered = await mcp.list_tools()     # list[MCPTool]: name, remote_name, description,
                                         # input_schema, output_schema, annotations (não confiáveis)
    group = await mcp.tool_group(include={"search_issues"}, approval_handler=ask_user)
    extra = await mcp.tools(include={"get_issue"}, policies={"get_issue": read_only})
```

- `MCPClient(transport, *, name, timeout=60.0, protocol="auto", policy=None, on_tools_changed=None)`;
  `protocol="legacy"` força o `initialize`. Transportes stdlib: `StdioTransport(command, args, env, cwd)`,
  `HttpTransport(url, headers=None, http_client=None)` (`http_client` da app, p. ex. com
  `OAuthClientProvider`) e `InProcessTransport(server)` para testes e servidores embebidos.
- `__aenter__`/`connect()` cria a task dona, que entra no `mcp.Client` e espera; `__aexit__`/`aclose()`
  (idempotente, de qualquer task) sinaliza e aguarda. Falha a ligar → `MCPConnectionError(ConnectionError)`
  com a causa desembrulhada; erro ao listar → `MCPProtocolError(code)`. `list_tools()` pagina até
  `next_cursor is None`, com limite de páginas e `cache_mode="refresh"`. `tools()` e `tool_group()` aceitam
  `include`, `exclude`, `policy` e `policies`; `tool_group()` também `approval_handler`, `gates`, `max_calls`.
- Nomes `<name>_<remoto>`, o que não for `[A-Za-z0-9_-]` passa a `_`; acima de 64 ou em colisão, sufixo
  `_` + 8 hex do sha256 do nome remoto. Política: `policies[remoto]` > `policy` >
  `ToolRuntimePolicy(capability=f"mcp:{name}", risk_level="high", requires_approval=True)`.
- Cada tool é `tool_from_schema(handler, ...)` (C02); o handler chama `call_tool` e devolve:

| MCP | `ToolResult` |
|---|---|
| `is_error=False` | `success(texto)`: blocos `text` e recursos de texto unidos por `\n`; só `structured_content` → JSON; imagem, áudio, blob → `[image image/png, 12 KB]`; `resource_link` → `[resource <uri>]` |
| `is_error=True` | `failure("mcp_tool_error", texto, retryable=True)` |
| `MCPError` -32001 · -32000 · outro | `mcp_timeout` (retryable) · `mcp_connection_closed` · `mcp_protocol_error` (`details["code"]`) |
| resultado fora do output schema ou do protocolo | `mcp_invalid_result` |
| outro loop (`execute()`) · ligação fechada | `mcp_requires_async` · `mcp_connection_closed`, sem esperar |

- `metadata["mcp"]` leva `server`, `tool` (remoto), `structured_content` e `content` serializado.
  `CancelledError` propaga (o SDK avisa o servidor). Metering só no executor (`_executor.py:223-252`):
  uma cobrança por chamada, `tools/list` não conta. Sem wrappers `_sync`.
- `on_tools_changed` (sync ou async; excepções só no log) dispara com `ToolListChangedNotification`
  (2025) ou por `listen` se o servidor anuncia `tools.listChanged` (2026). O grupo não muda sozinho: a
  app relista e usa `group.add(..., replace=True)`/`remove` (C02).

## Decisões a fixar antes de codificar

1. **SDK.** (a) `mcp>=2.2,<3`; (b) `mcp>=1.28,<2`; (c) as duas. Recomendo (a): é a linha estável, fala as
   duas eras (`mode="auto"` cai para `initialize`), a 1.x só recebe correcções e 2.2 é a versão verificada.
2. **Âmbito.** Recomendo só tools; resources, prompts, elicitação, sampling e roots depois. Um servidor
   que pede input recebe `mcp_protocol_error`.
3. **Dono da ligação.** (a) `async with` na task de quem abre; (b) task dona. Recomendo (b), porque o SDK
   recusa fechar noutra task e a app e o `Flow` abrem e fecham em tasks diferentes.
4. **Nomes.** (a) nome remoto; (b) prefixo, saneamento e hash. Recomendo (b): `.` e mais de 64
   caracteres falham no OpenAI, e a spec manda desambiguar sem `serverInfo.name`.
5. **Política.** (a) risco alto com aprovação, anotações só expostas; (b) risco derivado de
   `readOnlyHint`. Recomendo (a), porque a spec as declara não confiáveis; a app passa `policies=`.
6. **Resultados.** Recomendo texto para o modelo e o resto em `metadata`, porque os adaptadores só
   enviam texto em tool results.
7. **`tools/list_changed`.** (a) avisar; (b) sincronizar o grupo. Recomendo (a): a lista branca é da
   app, e uma tool nova ou alterada a meio de um run é superfície de ataque.
8. **Síncrono.** (a) sem `_sync`, com a guarda; (b) portal anyio numa thread. Recomendo (a): a ligação
   pertence a um loop e a app é assíncrona. Excepção documentada à regra do `AGENTS.md`.
9. **Timeout.** Recomendo 60 s por pedido (`None` desliga): o `Client` não tem nenhum por omissão, a spec
   pede timeouts, e o timeout de step (D1) continua a cancelar.
10. **Extra.** `mcp = ["mcp>=2.2,<3"]` em `optional-dependencies`, `all` e `dev`, depois `uv lock`;
    `toolkit/mcp/__init__.py` exporta os transportes e dá `MCPClient`, `MCPTool` e erros por
    `__getattr__` (como `toolkit/moderation/__init__.py:10-22`), com `require_sdk("mcp", "mcp")`
    (`core/_providers/_imports.py:6-12`). Nada entra em `toolkit/__init__.py`.

## Sub-tarefas, por ordem

- **C03a** Extra, `uv lock`, esqueleto (transportes, exports preguiçosos, erro sem extra). **C03b**
  `MCPClient`: task dona, `connect`/`aclose`, erros de ligação, `list_tools` paginado.
- **C03c** Tools: nomes, políticas, resultados, erros, guarda síncrona. **C03d** `on_tools_changed`.
  **C03e** Transportes stdio e HTTP com testes herméticos. **C03f** Docs e exemplo.

## Ficheiros

- `pyproject.toml`, `uv.lock`; `src/ai_arch_toolkit/toolkit/mcp/__init__.py`, `_client.py`,
  `_transports.py`, `_tools.py`, `_errors.py` (novos)
- Testes novos: `tests/mcp/__init__.py`, `conftest.py`, `test_client.py`, `test_tools.py`,
  `test_list_changed.py`, `test_transports.py`, `servers/stdio_server.py`
- Docs: `docs/mcp.md` (novo); `mkdocs.yml`, `README.md`, `docs/index.md`, `AGENTS.md` (partilhados com
  C06); `docs/framework-overview.md` (partilhado com C06, C07); `docs/tools.md` (partilhado com C02,
  C04, C05, C07, C08); `docs/api.md` (partilhado); `docs/getting-started.md`;
  `examples/<n>_mcp_client.py`, `examples/README.md`, `docs/examples.md` (o número é o próximo livre, atribuído ao aplicar)

## Prova

Com `pytest.importorskip("mcp")`, sem rede (servidor em processo, subprocesso Python, ASGI em memória):

- `test_client.py`: abrir numa task e fechar noutra; `aclose()` duas vezes; comando inexistente e
  `http://127.0.0.1:9/mcp` → `MCPConnectionError`, nunca `ExceptionGroup`; 150 tools em páginas de 10.
- `test_tools.py`: `geo.distance`, `a.b` e `a_b` → nomes distintos com ≤ 64; sem handler →
  `approval_denied` e o servidor não é chamado; com handler, `{"a": "1", "b": 2}` → o servidor recebe
  `1` e o texto é `3`; `ToolError` → `mcp_tool_error`; `timeout=0.2` numa tool lenta → `mcp_timeout` e a
  chamada seguinte funciona; `Policy(timeout=)` num step cancela e a ligação continua; imagem →
  marcador e `metadata`; `execute()` numa thread → `mcp_requires_async` em < 1 s; depois de `aclose()`
  → `mcp_connection_closed`; ReAct com LLM falso → `MeterScope().snapshot().tool_calls == 1`.
- `test_list_changed.py`: servidor que acrescenta uma tool com `ctx.notify_tools_changed()` (2026) ou
  `ctx.session.send_tool_list_changed()` e `protocol="legacy"` → `on_tools_changed` uma vez, grupo igual.
- `test_transports.py`: stdio com `sys.executable` → variável do pai invisível, `env=` visível, `repr`
  sem valores; HTTP por `httpx2.ASGITransport` (lifespan da app aberto,
  `TransportSecuritySettings(enable_dns_rebinding_protection=False)`) → os `headers` chegam; `headers`
  com `http_client` → `ValueError`; sem extra (`sys.modules["mcp"] = None`) o módulo importa e
  `MCPClient` dá `ImportError` com `pip install ai-arch-toolkit[mcp]`.

## Fora do âmbito

- Resources, prompts, completions, elicitação, sampling, roots, logging, progresso; SSE antigo; servir
  tools do toolkit por MCP; resultados multimodais; reconexão; vários servidores num objecto.
- Da app: fluxo OAuth (constrói o `OAuthClientProvider`), sincronizar o grupo, listas brancas,
  "sempre/perguntar/nunca", UI, `agent_as_tool`, servidores MCP em manifestos (`tools.factory`).

## Riscos

- SDK v2 recente (2.0.0 → 2.2.0 em seis semanas), fixado em `<3`; superfície usada pequena (`Client`,
  `StdioServerParameters`, `streamable_http_client`, `MCPError`, `mcp.types`, `mcp.client.subscriptions`).
- Árvore pesada só para cliente (starlette, uvicorn, pyjwt/cryptography, opentelemetry-api); `httpx2`
  valida TLS contra o sistema (truststore) e falha em contentores sem CA.
- Descrições, schemas e resultados são texto de terceiros no prompt (tool poisoning), sem limite de
  tamanho no v1; o comando stdio é código arbitrário da configuração; `env` e `headers` levam segredos.
- `InProcessTransport` em `protocol="auto"` não passa por JSON-RPC (stdio e ASGI cobrem o fio). Com um
  tracer OpenTelemetry na app, o SDK envia `traceparent` ao servidor.

## Registo do dono

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
