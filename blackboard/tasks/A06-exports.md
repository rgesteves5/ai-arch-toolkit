# A06 · As exportações: o encaminhamento de modelos (G-14) e o backend de memória público (G-24)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 6 (G-14, G-24) · **Decisões:** nenhuma nova ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `6bc7038`)

- **G-14.** O `resolve_provider_name` está no `__all__` de `core/_providers`, mas não sai de
  `ai_arch_toolkit.core`. A tabela de prefixos (`_MODEL_PREFIXES`, `_MODEL_IDS`) e a regra de
  loopback (`_is_local_url`) são privadas. O ai-network copia-as, e à cópia já falta o
  `muse-spark-`.
- **G-24.**
  - O `NetworkXBackend` da memória não está em nenhum `__all__`: os docs e os exemplos 29 e 30
    importam-no do módulo privado;
  - e o pyright recusa `GraphStore(NetworkXBackend())`: o backend do core devolve o `Node[Any]`
    genérico, e o `MemoryBackend` pede o `Node` da memória (reproduzido).

## Nota de desenho (antes do código)

- **G-14:**
  - `MODEL_PREFIXES` e `MODEL_IDS` são vistas só de leitura (`MappingProxyType`) das tabelas que o
    `create_provider` usa, por isso não podem divergir dela;
  - `_is_local_url` passa a `is_local_url`;
  - os quatro nomes, com o `resolve_provider_name`, saem de `ai_arch_toolkit.core`.
- **G-24:**
  - o `NetworkXBackend` do core passa a genérico no tipo de nó
    (`NetworkXBackend[N: Node[Any] = Node[Any]]`). O da memória é `NetworkXBackend[Node]` (o
    `Node` da memória), e satisfaz o `MemoryBackend`. O `subgraph` e o `ego_graph` devolvem um
    backend da mesma classe;
  - o `NetworkXBackend` sai, preguiçoso, de `ai_arch_toolkit.toolkit.memory.graph`, de
    `ai_arch_toolkit.toolkit.memory` e de `ai_arch_toolkit.core.graph` (por `__getattr__`, para o
    `networkx` continuar opcional), e entra no `__all__` de cada um;
  - os docs, os exemplos e os testes passam a importar do caminho público.
- **Provas:**
  - os nomes da G-14 importam-se de `ai_arch_toolkit.core`; cada prefixo da tabela encaminha para
    o seu fornecedor; a tabela não aceita escrita; a regra de loopback em cada caso;
  - `GraphStore(NetworkXBackend())`, importado do caminho público, passa no pyright (um teste
    corre-o sobre um ficheiro), e funciona: um nó guardado volta como `Node` da memória;
  - sem o `networkx`, pedir o `NetworkXBackend` dá o `ImportError` com a dica do extra.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/__init__.py`, `core/__init__.py`
- `src/ai_arch_toolkit/core/graph/_networkx.py`, `core/graph/__init__.py`
- `src/ai_arch_toolkit/toolkit/memory/graph/_networkx.py`, `toolkit/memory/graph/__init__.py`,
  `toolkit/memory/__init__.py`
- os testes que mudarem e os novos; `examples/29_memory_middleware.py`,
  `examples/30_memory_agent_tools.py`
- `docs/memory.md`, `docs/getting-started.md`, `docs/graph.md`, `docs/api.md`,
  `docs/framework-overview.md`, `docs/agents-and-capabilities.md`, `AGENTS.md`, `CHANGELOG.md`
- `uv.lock` (o `virtualenv` dos alertas do Dependabot, num commit à parte)

## Registo do dono

- Estado: done (Claude, 2026-10-05), pela nota de desenho.
- **G-14:**
  - `MODEL_PREFIXES` e `MODEL_IDS` (vistas `MappingProxyType` das tabelas do registo), o
    `is_local_url` (era o `_is_local_url`) e o `resolve_provider_name` saem de
    `ai_arch_toolkit.core`;
  - o `tests/test_provider_registry.py` passou a usar o nome novo.
- **G-24:**
  - o `NetworkXBackend` do core é `NetworkXBackend[N: Node[Any] = Node[Any]]`, e o `subgraph` e o
    `ego_graph` devolvem um backend da mesma classe (`type(self)()`);
  - o da memória é `NetworkXBackend[Node]` e satisfaz o `MemoryBackend`;
  - sai, preguiçoso por `__getattr__`, de `toolkit.memory`, de `toolkit.memory.graph` e de
    `core.graph`, e está no `__all__` de cada um;
  - os docs, os exemplos 28 a 30 e os testes que não testam o próprio módulo importam do caminho
    público.
- **Testes novos (20):**
  - `tests/test_provider_routing_exports.py` (14): cada prefixo encaminha, a tabela tem a família
    `muse-spark-`, não aceita escrita, o adaptador compatível, e dez casos da regra de loopback;
  - `tests/memory/test_public_backend.py` (5): o nome nos três `__all__`, o protocolo em
    execução, um nó que volta como `Node` da memória, o pyright sobre um ficheiro que o usa (falha
    com os tipos antigos, verificado), e o `ImportError` com a dica sem o `networkx`.
- **À parte** (os alertas do Dependabot no push da A05, três altos): `virtualenv` de 21.7.11 para
  21.14.5 no `uv.lock`. É uma dependência de desenvolvimento, pelo `pre-commit`.
- Gate: 6352 passed, 42 skipped; ruff, formatação e pyright limpos; `uv lock --check` em dia.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 6 fechou:
  - `resolve_provider_name`, `MODEL_PREFIXES`, `MODEL_IDS` e `is_local_url` saem de
    `ai_arch_toolkit.core`, e as tabelas são as do registo (com o `muse-spark-` que faltava à
    cópia);
  - o `NetworkXBackend` sai de `ai_arch_toolkit.toolkit.memory`, e `GraphStore(NetworkXBackend())`
    passa no pyright.

  Na app, saem a cópia dos prefixos e da regra de loopback do `nodes/providers.py`, e o import
  privado com a excepção de tipo do `agents/memory.py`."
