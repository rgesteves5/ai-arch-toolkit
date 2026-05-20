# AI-ARCH-TOOLKIT REWRITE BRIEFING

## Para o Claude Code

Este documento é o briefing completo para reescrever o `ai-arch-toolkit`. Foi produzido após uma exploração profunda de first principles sobre o que são LLMs e como todo o seu ecossistema se reduz a um conjunto pequeno de primitivas.

**O código atual funciona.** Usa-o como referência de implementação — especialmente a lógica de providers, parsing de respostas, SSE streaming, token counting, e tool schema generation. Mas a arquitetura, as interfaces públicas, e o paradigma devem ser redesenhados de raiz.

**Rewrite total. Sem backwards compatibility. Projeto pessoal, só eu uso.**

---

## Decisões Técnicas (já tomadas)

### Naming
- A classe principal para LLM calls chama-se **`LLM`**, não Transform, não Model.
- O conceito de "Transform" (content → content) é o paradigma conceptual. `LLM` é o nome prático.
- Tudo o resto segue o mesmo princípio: nomes práticos nas classes, conceitos nos docs.

### Sync / Async
- **Async-first + wrappers sync como _sync methods.**
- Uma única implementação async. Métodos sync são wrappers finos.
- NÃO duplicar ficheiros (o código atual tem `_client.py` + `_async_client.py` — eliminar essa duplicação).

```python
# Async (a implementação real)
result = await llm.complete(messages)
async for chunk in llm.stream(messages):
    print(chunk)

# Sync (wrappers — mesma classe, métodos com _sync suffix)
result = llm.complete_sync(messages)
for chunk in llm.stream_sync(messages):
    print(chunk)
```

Internamente, os _sync methods usam `asyncio.run()` ou equivalente. Zero duplicação de lógica.

### Code Conventions (manter do projeto atual)
- Python 3.12+, `from __future__ import annotations` em todos os ficheiros
- Ruff line length: 99
- `ruff format` depois de edits
- Frozen dataclasses onde faz sentido (Response, Step) com `slots=True`
- Internal modules com `_` prefix
- Public API via top-level re-exports
- uv para package management
- pytest com asyncio_mode = "auto"

---

## O Paradigma

### O que é um LLM

Um LLM é uma função stateless: content in, content out. Não tem goals, memória, nem agency. Tudo o resto — tools, agents, orchestration, safety — é arquitetura que humanos constroem à volta desta função.

### Quatro Primitivas

Tudo no ecossistema LLM reduz-se a:

1. **Content** — o átomo de informação. Na prática: messages (dicts com role + content). Aceitar sempre raw dicts. Fornecer construtores de conveniência.

2. **Transform** — content → content. A shape universal. Um LLM call é um transform. Um tool é um transform. Um agent é um transform. De fora, todos têm a mesma forma.

3. **Identity** — (name, schema, trust). Discovery e addressability.

4. **Memory** — (content, scope, access_control). Persistência. access_control: read_only | append_only | read_write.

### Composição

Cinco operadores: Sequence, Parallel, Conditional, Loop, Recursion.

### Padrão Fundamental: Boundary

Boundary = Memory com access_control=read_only que constrains composição. É a constitution, o frozen prompt, o budget, as safety rules.

Para referência completa do paradigma, ler `docs/from_claude_chat/FIRST_PRINCIPLES.md`.

---

## Estratégia de Rewrite

Começar pelo core e ir expandindo. Não reescrever tudo de uma vez.

```
Fase 1: Core — LLM (calls), Response, content constructors
Fase 2: Tools — @tool decorator, Tools.from_* sources
Fase 3: Memory + Prompt
Fase 4: Agent — o loop (run, stream, start/Session, Step)
Fase 5: Guardians, orchestration, cycles (futuro)
```

**Fase 1 concretamente:**
1. Ler o código atual em `src/ai_arch_toolkit/llm/` para perceber como os providers funcionam
2. Criar nova estrutura: content constructors, Response, LLM
3. LLM wraps os providers internamente (reutilizar lógica existente)
4. Provar com UM provider primeiro (Anthropic) — `.complete()`, `.stream()`, sync wrappers
5. Generalizar para os outros providers
6. Testar

---

## Contratos da API Pública

### Content Constructors

Funções que retornam dicts. Não são classes. São conveniência — raw dicts sempre funcionam.

```python
def system(content: str) -> dict:
    return {"role": "system", "content": content}

def user(content: str | list) -> dict:
    return {"role": "user", "content": content}

def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}

def tool_result(content: Any, tool_use_id: str) -> dict:
    return {"role": "tool", "content": content, "tool_use_id": tool_use_id}
```

### Response

O que volta de qualquer call. Comporta-se como string no caso simples. Rico quando preciso.

```python
class Response:
    text: str                    # o texto gerado
    tool_calls: list[dict]       # tool calls (se houver)
    input_tokens: int
    output_tokens: int
    cost: float                  # estimativa USD (pode estar desatualizado)
    stop_reason: str
    model: str                   # que modelo correu
    raw: Any                     # resposta original do provider, intocada

    # Comporta-se como string
    __str__  → self.text
    __repr__ → self.text
    __bool__ → bool(self.text or self.tool_calls)
    __contains__(item) → item in self.text
    __add__(other) → self.text + other

    @property tokens → input_tokens + output_tokens
    @property has_tool_calls → len(self.tool_calls) > 0
```

Frozen dataclass com `slots=True`.

### LLM

A classe principal. Provider auto-detectado pelo model string.

```python
class LLM:
    def __init__(
        self,
        model: str,                    # "claude-sonnet-4-5-20250929", "gpt-4o", "gemini-2.0-flash", etc.
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: str | None = None,    # default: env var
        base_url: str | None = None,   # para providers custom
        **kwargs,
    ): ...

    # ── Async (implementação real) ──

    async def complete(
        self,
        messages: list[dict] | str,    # str é shorthand para [user("string")]
        *,
        temperature: float | None = None,   # override per-call
        max_tokens: int | None = None,
        tools: list | None = None,
        **kwargs,
    ) -> Response: ...

    async def stream(
        self,
        messages: list[dict] | str,
        **kwargs,
    ) -> AsyncIterator[str]: ...

    async def batch(
        self,
        message_sets: list[list[dict] | str],
        **kwargs,
    ) -> list[Response]: ...

    # ── Sync (wrappers) ──

    def complete_sync(self, messages, **kwargs) -> Response: ...
    def stream_sync(self, messages, **kwargs) -> Iterator[str]: ...
    def batch_sync(self, message_sets, **kwargs) -> list[Response]: ...

    # ── Alias ──

    async def __call__(self, messages, **kwargs) -> Response:
        return await self.complete(messages, **kwargs)
```

### Tool (Fase 2)

```python
@tool
async def search(query: str) -> str:
    '''Search the web.'''
    ...

@tool(retry=3, timeout=30, cache=True)
async def search(query: str, limit: int = 10) -> list[dict]: ...

# Fontes externas (definir interface agora, implementar depois)
Tools.from_function(fn, description="...")
Tools.from_mcp(server_uri)          # futuro
Tools.from_agent(agent, description="...")
Tools.from_openapi(spec_url)        # futuro

# Agrupamento opcional
db = ToolGroup("database", [query, insert, update])
```

Schema auto-inferido de type hints + docstring (reutilizar lógica de `tools/_decorator.py`).

### Memory (Fase 3)

```python
memory = Memory({"methodology": "Start broad.", "learned": ""})

# Dict-like
memory["learned"] = "PubMed > Scholar"

# Access control (enforced mecanicamente, não por prompt)
memory.lock("budget")        # read_only — código impede escrita
memory.unlock("budget")      # só o user pode unlock, não o agent

# Observability
memory.history()             # edições com timestamp + reason
memory.rollback("learned")
memory.snapshot() / memory.diff(snapshot)

# Persistência
memory.save("./memory.json")
Memory.load("./memory.json")
```

**Importante:** lock() é enforced pelo código, não é uma sugestão no prompt. O agent loop verifica locks antes de executar self_modify. Se locked, a escrita falha e o agent recebe erro.

### Prompt (Fase 3)

```python
# Simple
instructions = "You are a researcher."

# Structured
prompt = Prompt(
    role="You are a researcher.",
    rules=["Always cite", "Never fabricate"],
    context="Focus on 2024-2025.",
)

# Composable
combined = base + safety + domain

# Templates
prompt = Prompt(role="You are a {specialty} expert.").bind(specialty="bio")

# From file
prompt = Prompt.from_file("./prompts/researcher.md")
```

**Prompt é frozen (quem o agent é). Memory é mutable (o que aprendeu). São objetos separados.**

### Agent (Fase 4)

```python
agent = Agent(
    model="claude-sonnet-4-5-20250929",       # ou LLM instance
    tools=[search, read],
    instructions="You are a researcher.",  # str ou Prompt
    memory={"methodology": "Start broad."},  # dict ou Memory
    max_steps=50,
    max_time=300.0,
    max_cost=None,
    allow_self_modify=False,
    allow_spawn=False,
    name="researcher",
)

# ── Três verbos (async) ──

result = await agent.run("Research protein folding") -> Response
async for step in agent.stream("Research protein folding") -> AsyncIterator[Step]
session = agent.start("Research protein folding") -> Session

# ── Sync wrappers ──

result = agent.run_sync("Research protein folding")
for step in agent.stream_sync("Research protein folding"):
    print(step)

# ── Session (manual control) ──

session = agent.start("Research protein folding")
while not session.done:
    step = await session.step()
    session.inject(user("also check arxiv"))
    if session.cost > 1.0:
        session.stop()
result = session.result
```

**Step:**
```python
class Step:
    kind: str          # "tool_call" | "response" | "self_modify" | "spawn" | "error"
    content: str
    tool: str
    tool_input: Any
    tool_output: Any
    cost: float
    done: bool
```

**Nota:** Fase 4 implementa SÓ o Agent base (ReAct loop). Os 8 agent types atuais (ToT, LATS, etc.) são reimplementados depois.

**Agent como tool:**
```python
researcher = Agent(model="...", tools=[search], instructions="Research.")
research_tool = Tools.from_agent(researcher, description="Deep research")
orchestrator = Agent(model="...", tools=[research_tool, write_tool])
```

---

## Mapeamento: Código Atual → Novo

| Atual | Novo | Notas |
|---|---|---|
| `Client` + `AsyncClient` | `LLM` | Uma classe, async-first + sync wrappers |
| `Response` (frozen) | `Response` | + comporta-se como string |
| `Message` (frozen) | `dict` + construtores | Simplificar |
| `Tool` (dataclass) | Schema inferido pelo `@tool` | User não cria Tool objects |
| `ToolRegistry` | Não existe | Pass list to Agent |
| `BaseProvider` + 5 providers | Internal | Manter lógica, esconder interface |
| `Middleware` pipeline | Hooks (futuro, Fase 5) | |
| `_templates.py` | `Prompt` class | Composição + templates |
| `_memory.py` | `Memory` class | Access control + history |
| `BaseAgent` + 8 impls | `Agent` (ReAct) | Um agent fundamental |
| `AgentStep` | `Step` | Simplificado |
| `AgentResult` | `Response` | Unificar com LLM Response |
| `AgentConfig` | Params no Agent() | Sem config object separado |
| `_http.py` + `_async_http.py` | Um módulo HTTP async-first | + sync wrapper |

## O que reutilizar

**Reutilizar a lógica, não a interface:**
- `_providers/*.py` — HTTP calls, response parsing, tool call handling
- `_http.py` / `_async_http.py` — SSE, NDJSON, retry (unificar num módulo async-first)
- `_tokens.py` — token estimation
- `_cost.py` — cost calculation (integrar no Response)
- `tools/_decorator.py` — schema generation

**Reescrever:**
- `_types.py` — simplificar para Response, Step, dicts
- `_client.py` / `_async_client.py` — substituir por LLM
- `_middleware.py` — redesenhar como Hooks (Fase 5)
- `agents/_base.py` + impls — novo Agent
- `_memory.py` — novo Memory
- `_templates.py` — novo Prompt

---

## Referência

- `docs/from_claude_chat/FIRST_PRINCIPLES.md` — paradigma conceptual completo
- `docs/from_claude_chat/transform_api.py` — contrato LLM (nota: usa nome Transform, o nome real é LLM)
- `docs/from_claude_chat/agent_api.py` — contrato Agent
- `docs/from_claude_chat/tools_memory_prompt_api.py` — contrato Tools/Memory/Prompt
