# A03 · Correcção: tools, ids, clientes, pedidos longos, `cwd` e ligações (G-21, G-23, G-17, G-18, G-26, G-27)

- **Dono:** Claude (2026-10-04) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 3 (G-21, G-23, G-15, G-17, G-18, G-26, G-27) ·
  **Decisões:** nenhuma nova (o briefing diz o resultado de cada uma) · **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-04 no `565fca4`)

- **G-15** já fechou na frente O: as regras da OpenAI vivem no adaptador (a temperatura cai
  enquanto o modelo raciocina, e na Responses as tools vão a qualquer esforço), e cada fallback
  aplica as do seu modelo no seu `prepare()`. Fica fora desta ficha.
- **G-21.**
  - `ToolGroup.add` substitui uma tool com o mesmo nome só com um `warnings.warn`;
  - `functools.wraps` copia o `__dict__` da tool embrulhada, e com ele o `__tool_definition__`,
    cujo `fn` é a função de dentro: o executor corre-a e salta o invólucro;
  - o `DangerousToolGate` responde "requires --allow-dangerous-tools", uma flag de linha de
    comandos que o modelo repete à pessoa.
- **G-23.** O adaptador compatível (Chat Completions: Ollama, LM Studio, vLLM) põe na `ToolCall` o
  id que o servidor mandar, também vazio ou nenhum, e o `tool_result()` recusa-o depois de a
  ferramenta ter corrido. Só o Gemini dava um id seu.
- **G-17.** O `XAIProvider` constrói o `AsyncClient` gRPC no construtor, e o canal `grpc.aio`
  precisa de um loop: sem loop a correr, `RuntimeError`. Reproduzido no `asyncio.to_thread`, numa
  thread simples e na thread principal depois de um `asyncio.run` (Python 3.13).
- **G-18.**
  - O orçamento já soma ao `max_tokens` nos modelos com `budget_tokens` (14 096 por omissão), e
    os de thinking adaptativo não têm orçamento: essa parte está feita;
  - falta a outra: sem stream, o SDK da Anthropic recusa um pedido que espera passar de 10
    minutos (`max_tokens` acima de cerca de 21 333, ou do limite do modelo), antes de enviar. O
    orçamento conta: um `max_tokens` de 16 000 com um orçamento de 8 000 já é recusado. Havia um
    teste que afirmava a recusa (`test_claude_a_request_the_sdk_refuses_was_never_sent`).
- **G-26.** O `run_command` não aceita `cwd`.
- **G-27.**
  - O `search_files` percorre com `rglob("*")` e lê cada ficheiro com `is_file()`, que segue a
    ligação: um ficheiro que seja uma ligação para fora da pasta lê-se;
  - o `list_directory` tem a mesma classe: um padrão como `ligação/*` ou `../*` lista fora da
    pasta (o ai-network contorna-o com um padrão que só pode ser um nome).

## Nota de desenho (antes do código)

- **G-21:**
  - `ToolGroup.add`: outra tool com um nome que o grupo já tem é um `ValueError`. A mesma tool
    outra vez não muda nada;
  - `_definition_for`: uma definição cujo `fn` não é a função que a traz foi copiada (pelo
    `functools.wraps`) e passa a correr essa função, com o mesmo schema e a mesma política;
  - o `DangerousToolGate` responde numa frase para uma pessoa: a tool não correu porque está
    marcada como perigosa e esta execução não permite tools perigosas.
- **G-23:**
  - o `BaseProvider._answer`, por onde passa toda a resposta (complete, stream, lotes), dá um id
    `call_<24 hex>` a uma chamada que chegue sem id, ou com o id de outra chamada da mesma resposta;
  - o Gemini deixa de dar o seu;
  - os eventos `tool_call` de um stream saem da resposta já com os ids.
- **G-17:**
  - o `LoopAwareClientCache` guarda a fábrica e constrói o cliente no primeiro uso, que num
    pedido é dentro do loop. Vale para todos os adaptadores, porque a classe do problema é
    "construir um provider não faz trabalho preso a um loop";
  - o `close()` passa para o cache e só fecha um cliente que exista.
- **G-18:**
  - o `send()` da Anthropic pergunta ao SDK: se ele recusar o pedido sem stream (um
    `ValueError` local, antes de enviar), o pedido vai por stream, e devolve a mensagem final;
  - assim o limite é o do SDK, mesmo que mude. Um `timeout` dado pela pessoa desliga a recusa do
    SDK, e o pedido vai sem stream como ela escolheu.
- **G-26:** o `run_command` aceita `cwd`, uma pasta que tem de existir, e passa-a ao
  `subprocess.run` (muda só a pasta do processo filho).
- **G-27:**
  - o `search_files` lê só os ficheiros cujo caminho resolvido fica dentro da pasta resolvida;
  - o `list_directory` mostra só as entradas cuja pasta-mãe resolvida fica dentro da pasta
    pedida. Uma ligação dentro da pasta aparece pelo nome, mas o que ela aponta já não se
    percorre.
- **Provas:**
  - um nome repetido num `ToolGroup` levanta, e a mesma tool duas vezes não;
  - uma tool embrulhada com `functools.wraps` corre o invólucro;
  - o bloqueio de uma tool perigosa não fala de flags;
  - o adaptador compatível dá um id a uma chamada sem id, no complete e no stream, e dois ids
    iguais ficam diferentes;
  - o `XAIProvider` (e cada adaptador) constrói-se numa thread sem loop e depois responde no
    loop, contra o servidor gRPC local;
  - na Anthropic, um `max_tokens` de 64 000 sem stream chega ao servidor local por stream e
    devolve a resposta, e um pequeno vai sem stream;
  - o `run_command` com `cwd` corre lá, e com uma pasta que não existe devolve um erro;
  - o `search_files` não lê um ficheiro ligado para fora, e o `list_directory` não lista através
    de uma ligação para fora nem de `..`.

## Ficheiros

- `src/ai_arch_toolkit/core/_tools/_group.py`, `_executor.py`, `_governance.py`
- `src/ai_arch_toolkit/core/_providers/_base.py`, `_gemini.py`, `_anthropic.py`, `_xai.py`,
  `_openai_compatible.py`, `_responses.py` (os `close()` passam para o cache)
- `src/ai_arch_toolkit/toolkit/tools/_shell.py`, `_filesystem.py`
- `src/ai_arch_toolkit/toolkit/moderation/_openai.py` (da revisão: o mesmo cliente preso a um loop)
- os testes de tools, de providers, de loop e do toolkit que mudarem; os testes novos
- `docs/tools.md`, `docs/safety.md`, `docs/tools-catalog.md`, `docs/llm.md` (se falar de ids),
  `AGENTS.md`, `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-04), pela nota de desenho e pela revisão independente.
- **G-21:**
  - `ToolGroup.add` levanta `ValueError` para outra tool com um nome que o grupo tem. A
    comparação é por igualdade e não por identidade, porque cada leitura de `obj.metodo` dá um
    método ligado novo (um teste);
  - `_definition_for` religa ao invólucro a definição copiada pelo `functools.wraps`; vale também
    para o `execute_tool` com uma lista;
  - a frase do `DangerousToolGate`.
- **G-23:**
  - `named_calls` no `_base.py`, chamado pelo `_answer` (complete, stream e lotes da OpenAI e da
    Anthropic) e pelo `chat_batch_response` (as linhas de lote de Chat Completions, lidas sem
    adaptador);
  - o Gemini deixou de dar o seu id.
- **G-17:**
  - o `LoopAwareClientCache` constrói o cliente na primeira leitura de `_client`;
  - o `close()` passou para o cache: só fecha um cliente que exista, e larga, sem o fechar, um
    cliente cujo loop já fechou (o de uma chamada sync). O Gemini sobrepõe o `_close_client`;
  - os `close()` da Anthropic, do compatível, do xAI e da Responses saíram;
  - o `prepare()` do xAI deixou de mexer no cliente. O pedido é construído pelo `chat.create` do
    SDK sobre nenhum canal (`_OfflineChat`), e o `send()`/`open_stream()` copiam-no para um pedido
    do cliente (`_bound`), dentro do loop. O `mark_dispatched()` vem depois disso: um cliente que
    não se construa já não conta como pedido enviado;
  - o `OpenAIModerator` passou para o mesmo cache;
  - nos testes, o `fakegrpc` ganhou o `point()` e o `serving_in_thread()`, e o `fakeserver` o
    `KeepAlive` (uma ligação aberta entre pedidos, como numa API a sério).
- **G-18:** o `send()` da Anthropic tenta o `create`; com um `ValueError` antes de enviar
  (`dispatched()`, novo no `_base.py`), vai por stream. Com o pedido já enviado, levanta: um
  pedido nunca se manda duas vezes (o teste dos 200 sem JSON continua a dar `ResponseError`).
- **G-26:** `run_command(cwd=...)`, com `~` expandido; uma pasta que não exista, um ficheiro ou
  um `~nome` sem utilizador devolvem "Not a directory".
- **G-27:**
  - o filtro no `search_files` e no `list_directory`. O Python 3.13 já não descia por uma pasta
    ligada no `rglob`, mas lia um ficheiro ligado, e o `glob` do `list_directory` saía por `../*`
    e por `ligação/*`;
  - no `list_directory`, o limite de 1000 entradas conta todas as que o padrão encontra, dentro
    ou fora, para um padrão que sai da pasta não andar mais do que antes.
- **Revisão independente** (um agente, sobre o diff, com scripts próprios contra servidores
  locais). Cinco achados:
  1. **Regressão:** o `close()` depois de chamadas sync levantava "Event loop is closed"
     (`with LLM(...) as llm:` à volta de `complete_sync()`). Corrigido: o `close()` larga um
     cliente cujo loop fechou, como o antigo, que o trocava por um novo e fechava esse. Provas:
     `test_an_llm_closes_after_sync_calls`, com o servidor `KeepAlive`, no compatível e na
     Anthropic, e um teste unitário do cache;
  2. **G-17 só em parte:** num `stream_sync` do xAI, o `prepare()` corria na thread de quem chama
     e construía lá o cliente gRPC. Depois de um `complete_sync`, reutilizava o cliente do loop
     fechado e contava o pedido como enviado. Corrigido com o `_OfflineChat` e o `_bound`. Prova:
     `test_an_xai_sync_stream_needs_no_loop_of_its_callers` (um stream, um complete e outro
     stream, na thread principal depois de um `asyncio.run`);
  3. **Regressão menor:** o filtro do `list_directory` corria antes do limite, e um `../**/*`
     percorria tudo (30 063 entradas contra 1 001). Corrigido; o teste falha contra o código de
     antes;
  4. **A mesma classe noutro sítio:** o `OpenAIModerator` construía o seu `AsyncOpenAI` à parte do
     cache, e o segundo `moderate_sync()` falhava. Corrigido. Prova:
     `test_the_openai_moderator_survives_a_second_sync_call`;
  5. **Fora do âmbito:** "um nome, uma tool" só vale no `ToolGroup`; o `prepare_tools` e o
     `execute_tool` com listas não o verificam. Fica em `FINDINGS.md`, porque os fluxos dos
     agentes usam um `ToolGroup`.
- **Testes que afirmavam o contrato antigo, corrigidos:**
  - `tests/integration/test_provider_transport.py`: o
    `test_claude_a_request_the_sdk_refuses_was_never_sent` afirmava a recusa. Passou a quatro
    casos: vai por stream (dois modelos), um pedido pequeno vai sem stream, e um timeout próprio
    mantém um grande sem stream;
  - sete testes da construção dos clientes leem o `_client` antes de ver os argumentos dados ao
    SDK, porque o cliente nasce no primeiro uso. São dois em `test_anthropic_provider.py`, dois
    em `test_xai_provider.py` e um em cada um de `test_gemini_provider.py`,
    `test_openai_compatible_provider.py` e `test_openai_provider.py`;
  - `test_provider_loop_safety.py::test_injected_client_is_never_replaced`: a fábrica não chega a
    correr;
  - `test_gemini_provider.py::test_a_call_without_an_id_gets_one` passa pela base (`assembled`);
  - `tests/moderation/test_openai.py::test_context_manager` usa o moderador antes de o fechar;
  - `tests/nanope/test_advanced_configurable_agent.py` afirmava a flag na mensagem do bloqueio.
    Este ficheiro e o anterior não estavam na lista da ficha.
- **Testes novos (43):**
  - `test_tools_group.py`: `TestOneToolPerName` (4), `TestWrappedTools` (3) e a frase do
    bloqueio (1);
  - `tests/integration/test_tool_call_ids.py` (3), contra o servidor local: sem id, com o id de
    outra chamada, e num stream;
  - `test_provider_loop_safety.py::TestBuiltWithoutALoop` (10): os cinco fornecedores num
    `asyncio.to_thread`, o xAI numa thread simples, o xAI construído numa thread a responder no
    loop (contra o servidor gRPC local), e o `close()`;
  - `tests/integration/test_clients_across_loops.py` (4);
  - `test_provider_transport.py` (4);
  - `test_shell.py::TestWorkingDirectory` (4);
  - `test_filesystem.py::TestLinksOutOfTheFolder` (9);
  - `tests/moderation/test_openai.py`: um moderador que não chegou a ser usado não constrói
    cliente (1).
- **Desvios à nota de desenho:**
  - o `list_directory` entrou na G-27, por ser a mesma classe;
  - o `chat_batch_response` também dá ids;
  - o `prepare()` do xAI e o `OpenAIModerator` vieram da revisão.
- **Achados** (em `FINDINGS.md`):
  - um `~nome` sem utilizador faz o `read_file`, o `list_directory` e o `search_files` levantar
    `RuntimeError`;
  - o "um nome, uma tool" fora do `ToolGroup`.
- **Docs:** `docs/tools.md`, `docs/llm.md`, `docs/tools-catalog.md`, `docs/safety.md`,
  `AGENTS.md`, `CHANGELOG` (uma nota de actualização; Added, Changed e Fixed).
- Gate: 6298 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Não verificado ao vivo** (pediria chamadas pagas): o pedido longo da Anthropic e o xAI, que
  continua sem créditos. As provas são os SDKs reais contra os servidores locais.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 3 fechou (a G-15
  já tinha fechado na frente O):
  - um `ToolGroup` recusa outra tool com um nome que já tem, e uma tool embrulhada com
    `functools.wraps` corre o invólucro;
  - o `DangerousToolGate` responde numa frase para a pessoa;
  - toda a chamada de ferramenta chega com um id próprio, em todos os fornecedores;
  - um `LLM` constrói-se sem loop (o xAI também), e os streams sync do xAI correm de qualquer
    thread;
  - na Anthropic, um pedido longo sem stream vai por stream;
  - o `run_command` aceita `cwd`;
  - o `search_files` não lê por uma ligação para fora da pasta, e o `list_directory` não lista
    fora dela.

  Na app, saem estes contornos:
  - a recusa de nomes repetidos no `offer()` e a definição refeita das capacidades oferecidas;
  - os ids `call_<volta>_<n>` do `chat/turns.py`;
  - o `open_llm()` a construir no loop;
  - o padrão `_glob_name` do `filesystem.list`.

  E a E03-03 já pode dar um `max_tokens` grande a um Claude com thinking: o orçamento soma-se,
  e o pedido vai por stream quando o SDK o pede."
