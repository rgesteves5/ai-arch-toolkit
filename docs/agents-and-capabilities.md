# Agents & Capabilities

This page is the capabilities index — each subsystem now has its own focused page. Below the index are end-to-end recipes that combine several of them, plus a quick-reference table.

## Capability index

| Capability | Page |
|---|---|
| Agent flows (ReAct, Reflexion, ReWOO, Plan-Execute, ToT, LATS, Self-Discovery, LLM Compiler, Generate-Review) and the Flow engine | [Flow Architecture](flow-architecture.md) |
| High-level agent facade — `Agent` + `ReasoningSpec`, named strategies, config-driven specs, `AgentResult` | [Agent & ReasoningSpec](agents.md) |
| LLM facade — completion, streaming, structured output, extended thinking, fallback, retry, token counting | [LLM Facade](llm.md) |
| Defining tools, `ToolGroup`, server tools, `run_tools` | [Tools](tools.md) |
| The 132 pre-built tools | [Tools Catalog](tools-catalog.md) |
| Risk levels, approval gates, dangerous-tool blocking, trace redaction, budgets | [Tool Governance & Safety](safety.md) |
| Middleware (before/after hooks, async, execution order) | [Middleware](middleware.md) |
| Graph-backed agent memory | [Memory](memory.md) |
| Reusable file and structured-data loading | [Resources](resources.md) |
| Prompt-injectable reference data | [Knowledge Registry](knowledge.md) |
| Files, templates, manifests, layouts, stability, and fingerprints | [Prompts](prompts.md) |
| Messages and multimodal content | [Content & Messages](content.md) |
| Cost estimation, the pricing registry, run-wide budgets | [Pricing & Cost Tracking](pricing.md) |
| Input/output content moderation | [Moderation](moderation.md) |
| The general-purpose graph layer | [Graph](graph.md) |

---

## Putting It Together

### Research agent with memory and knowledge

```python
from ai_arch_toolkit import (
    LLM, ToolGroup, KnowledgeRegistry, GraphStore, MemoryMiddleware, Prompt, PromptSection,
    SimilarityView, TemporalView,
)
from ai_arch_toolkit.core import State
from ai_arch_toolkit.toolkit.tools import get_weather, wikipedia_search, datetime_now
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.memory import memory_tools
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

# Knowledge base
knowledge = KnowledgeRegistry()
knowledge.register("research_guidelines", content="Always cite sources. Verify claims.")
system_prompt = Prompt(
    sections=(
        PromptSection(name="role", content="You are a research assistant.", order=100),
        PromptSection.from_knowledge(
            knowledge, ["research_guidelines"], name="knowledge", order=200
        ),
    )
).render()

# Memory
store = GraphStore(NetworkXBackend())
mem = memory_tools(store)               # ToolGroup with remember/recall/explore_memory/forget_memory
memory_mw = MemoryMiddleware(           # auto-inject relevant memories on every call
    find=SimilarityView(store, node_type="fact").find,
    record=TemporalView(store, node_type="interaction").append,
    k=5,
)

# LLM with memory middleware
llm = LLM(
    "claude-sonnet-4-20250514",
    middleware=[memory_mw],
    fallback="gpt-4o",
)

# Agent flow with tools + memory tools
tools = ToolGroup(get_weather, wikipedia_search, datetime_now, *mem.tools)
flow = react_flow(
    llm, tools,
    system=system_prompt.text,
    max_iterations=15,
)

state = State(operational=react_initial_state(
    "Research the history of the Eiffel Tower and remember key facts"
))
result = await flow.run(state)
```

### Multi-model pipeline with cost control

```python
from ai_arch_toolkit import LLM, ToolGroup
from ai_arch_toolkit.core import State
from ai_arch_toolkit.toolkit.agents.flows import plan_execute_flow, plan_execute_initial_state

fast = LLM("claude-haiku-4-5-20251001")
smart = LLM("claude-opus-4-0-20250514")

flow = plan_execute_flow(
    fast,  # default model
    tools,
    planner_llm=fast,     # cheap model plans
    exec_llm=fast,        # cheap model executes
    solver_llm=smart,     # smart model synthesizes
)

state = State(operational=plan_execute_initial_state("Complex research task"))
result = await flow.run(state)
print(f"Total cost: ${result.total_cost:.4f}")
```

### Parallel research with Flow composition

```python
from ai_arch_toolkit.core import LLM, State, Step, Result, StateSnapshot
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state

llm = LLM("claude-sonnet-4-20250514")

async def research_topic(topic_key: str):
    """Create a step that researches a topic via inner ReAct."""
    async def _research(snap: StateSnapshot) -> Result:
        task = snap.require("task")
        inner = react_flow(llm, tools, system=f"Research: {task}", max_iterations=5)
        state = State(operational=react_initial_state(task))
        await inner.run(state)  # metered under the shared run scope; no manual cost threading
        response = state.get("response")
        return Result(
            value=response.text if response else "",
            artifacts={topic_key: response.text if response else ""},
        )
    return Step(name=topic_key, fn=_research)

flow = Flow(
    FlowStep(step=await research_topic("tech")),
    FlowStep(step=await research_topic("market")),
    FlowStep(
        step=Step(name="synthesize", fn=synthesize),
        after=("tech", "market"),  # runs after both complete
    ),
    name="parallel_research",
)

state = State(operational={"task": "Electric vehicle batteries"})
result = await flow.run(state)
```

---

## Quick Reference: What Enhances Agent Flows

| Capability | How it connects | Example |
|---|---|---|
| **Agent facade** | Declarative wrapper over the factories | `Agent(ReasoningSpec(strategy="react"), llm, tools).run_sync(task)` |
| **Fallback chains** | LLM-level, transparent to flows | `LLM("opus", fallback="sonnet")` |
| **Retry** | LLM-level, exponential backoff | `LLM("opus", retry=RetryConfig(max_retries=3))` |
| **Middleware** | Hooks into every LLM call | Cost tracking, logging, memory injection |
| **Memory** | `MemoryMiddleware` + `memory_tools()` | Agents remember across conversations |
| **Knowledge** | Injected into system prompts | Domain context, style guides |
| **Structured prompts** | Rendered into `ReasoningSpec.system` or an LLM system prompt | Ordered sections and experiment fingerprints |
| **Pre-built tools** | 132 ready-to-use tools | Weather, Wikipedia, math, papers, public data |
| **Server tools** | Provider-hosted web search, code execution | `tools=[web_search()]` |
| **Tool governance** | Risk levels, approval, blocking | `@tool(requires_approval=True)` |
| **Structured output** | `output_schema` on LLM call | Pydantic models as output |
| **Extended thinking** | `thinking=True` on LLM call | Anthropic reasoning traces |
| **Multimodal input** | `Content` accepts images, PDFs | Vision + tools agents |
| **Per-phase models** | `Agent` deps (`deps={"planner_llm": haiku}`), factory kwargs, or manifest `strategy.phases` | Cheap planner, smart solver |
| **Per-phase prompts** | Knobs (`planner_system`, …) or manifest `strategy.phases.*.system`; `{tools}` token renders the phase's tool catalog | Custom planner instructions |
| **Cost tracking** | Automatic on every Response + FlowResult | `result.total_cost` |
| **Budgets** | Run-wide caps on a flow | `Flow(budget_policy=BudgetPolicy(...))` |
| **Token counting** | `llm.count_tokens()` | Budget estimation before running |
| **Pricing registry** | Built-in model pricing | `pricing.estimate_cost(...)` |
| **Flow streaming** | `flow.iter()` / `flow.iter_sync()` | Real-time progress updates |
| **Flow composition** | `flow.as_step()` / nested Flows | Agents inside agents |
| **Policy** | Per-step retry, timeout, confidence | `Step(policy=Policy(...))` |
| **Trace** | Full execution history | `result.trace.steps` |
