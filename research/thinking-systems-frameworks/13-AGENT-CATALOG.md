# 13 — Agent Catalog: How Each Variation Uses the Thinking Systems

A mechanical breakdown of every agent architecture — what triggers thinking, how systems are invoked, and where the intelligence lives.

---

## At a Glance

| # | Agent Variation | Thinking Mechanism | Systems Live As | Selection Method | LLM Calls | Best For |
|---|----------------|-------------------|-----------------|-----------------|-----------|----------|
| A | Single Agent | Prompt chaining | Prompt templates | LLM decides inline | 10–20 | Prototyping |
| B | Multi-Agent Ensemble | Agent-per-system | Separate agents | Conductor agent | 10–30 (parallel) | High-stakes, diversity |
| C | SEDA Pipeline | Pipeline stages | Stage handlers | Fixed pipeline config | 8–15 | Batch processing |
| D | Tool-Use Agent | Tool calls | Registered tools | LLM tool selection | 5–15 | MVP, fastest to build |
| E | Hybrid (Recommended) | Prompt + Router + Tools | Mixed (prompt/code/tools) | Heuristic + LLM | 1–18 (adaptive) | Production |
| F | Dual-Process Gated | Entry gate + escalation | Tiered (fast/medium/deep) | Heuristic router | 1–18 | Cost-optimized production |

---

## Variation A: Single Agent, Dynamic Prompt Chaining

### How It Works

One LLM, one conversation. The agent receives a mega-prompt that contains ALL thinking system instructions. The LLM reads the problem and decides — within a single generation — which systems to apply and in what order. Each "system" is a section of the prompt output.

### Mechanical Detail

```
┌──────────────────────────────────────────┐
│  ONE LLM CALL (or a sequential chain)    │
│                                          │
│  System Prompt:                          │
│  "You have access to these 24 thinking   │
│   systems: [full descriptions].          │
│   For the given problem:                 │
│   1. Classify domain (Cynefin)           │
│   2. Select appropriate systems          │
│   3. Apply each in sequence              │
│   4. Synthesize a final answer"          │
│                                          │
│  The LLM outputs:                        │
│  ## Domain Classification: Complex       │
│  ## First Principles Analysis: ...       │
│  ## Inversion Check: ...                 │
│  ## Pre-mortem: ...                      │
│  ## Synthesis: ...                       │
└──────────────────────────────────────────┘
```

### Thinking Systems Live As: **Sections in a prompt template**

Each system is described in the system prompt as instructions. The LLM executes them inline as labeled sections of its output. There are no separate calls — everything happens in one (long) generation.

```python
SYSTEM_PROMPT = """You are a structured thinker. For every problem:

1. CLASSIFY (Cynefin): Is this clear, complicated, complex, chaotic, or novel?

2. Based on the domain, apply these systems IN ORDER:
   - If Complex: First Principles → Divergent → Pre-mortem → Synthesis
   - If Complicated: Systems Thinking → Critical Thinking → Bayesian → Synthesis
   - If Chaotic: OODA (Act first) → Sensemaking → Reflect
   ...

3. For each system, use this format:
   ### [System Name]
   **Insights:** ...
   **Confidence:** X%
   **Suggested next:** ...

4. End with a SYNTHESIS that integrates all system outputs.
"""

# Single call
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=8192,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": problem}]
)
```

### Selection Method: **LLM decides inline**

The LLM itself picks which systems to use based on the prompt instructions. No external routing logic. The Cynefin classification and system selection all happen within the same generation.

### Strengths and Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Simplest to build (one prompt) | LLM may skip systems or apply them shallowly |
| Full context coherence (one conversation) | No structured output guarantee |
| No orchestration code needed | Can't tune temperature per system |
| Good for prototyping the framework | System 1 of the LLM drives selection (ironic) |
| | Context window limit caps depth |
| | Single model's biases affect everything |

### When to Use

Prototyping. Testing which system combinations work. Quick experiments. NOT for production.

---

## Variation B: Multi-Agent Ensemble

### How It Works

Each thinking system is a **separate agent** with its own system prompt, temperature, and optionally its own model. A **conductor agent** (running Cynefin + OODA + Metacognition) decides which agents to invoke, in what order, and how to synthesize their outputs. Agents can run in parallel where their phases are independent.

### Mechanical Detail

```
┌──────────────────────────────────────────────────┐
│  CONDUCTOR AGENT                                  │
│  (System prompt: Cynefin + OODA + Metacognition)  │
│  Temperature: 0.3 (analytical, precise)           │
│                                                    │
│  1. Classifies domain                              │
│  2. Selects which agents to invoke                 │
│  3. Dispatches calls (parallel where possible)     │
│  4. Receives structured outputs                    │
│  5. Synthesizes (or invokes Dialectical agent      │
│     if outputs conflict)                           │
│  6. Decides: loop back or finalize                 │
└──────────┬──────────┬──────────┬─────────────────┘
           │          │          │
    ┌──────▼───┐ ┌────▼─────┐ ┌─▼──────────┐
    │First     │ │Pre-mortem│ │Lateral     │
    │Principles│ │Agent     │ │Thinking    │
    │Agent     │ │          │ │Agent       │
    │temp: 0.3 │ │temp: 0.5 │ │temp: 0.9  │
    │model:    │ │model:    │ │model:      │
    │sonnet    │ │sonnet    │ │sonnet      │
    └──────────┘ └──────────┘ └────────────┘
```

### Thinking Systems Live As: **Independent agents (separate LLM instances)**

Each agent has:
- Its own system prompt (the thinking system's personality and instructions)
- Its own temperature (analytical systems: low temp; creative systems: high temp)
- Its own model (could use cheaper models for simple systems)
- A standard JSON output schema (common envelope from Doc 12)

```python
class ThinkingAgent:
    def __init__(self, name, system_prompt, temperature, model):
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = model

    def invoke(self, problem: str, context: dict) -> dict:
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": f"Problem: {problem}\nContext: {context}"}],
            tools=[{
                "name": f"output_{self.name}",
                "input_schema": COMMON_OUTPUT_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": f"output_{self.name}"},
        )
        return self._extract_tool_output(response)

# Agent registry
agents = {
    "first_principles": ThinkingAgent(
        name="first_principles",
        system_prompt="You are a First Principles analyst. Decompose everything to bedrock truths...",
        temperature=0.3,
        model="claude-sonnet-4-5-20250929",
    ),
    "lateral_thinking": ThinkingAgent(
        name="lateral_thinking",
        system_prompt="You are a Lateral Thinking provocateur. Break patterns, introduce randomness...",
        temperature=0.9,  # HIGH — maximize creativity
        model="claude-sonnet-4-5-20250929",
    ),
    "critical_thinking": ThinkingAgent(
        name="critical_thinking",
        system_prompt="You are a rigorous Critical Thinking evaluator. Apply intellectual standards...",
        temperature=0.1,  # LOW — maximize precision
        model="claude-sonnet-4-5-20250929",
    ),
    "pre_mortem": ThinkingAgent(
        name="pre_mortem",
        system_prompt="You are a Pre-mortem specialist. This plan has already failed...",
        temperature=0.5,
        model="claude-haiku-4-5-20251001",  # Cheaper model for structured task
    ),
}

# Conductor dispatches
async def run_ensemble(problem, selected_agents):
    tasks = [agents[name].invoke(problem, context) for name in selected_agents]
    results = await asyncio.gather(*tasks)  # Parallel execution
    return results
```

### Selection Method: **Conductor agent decides**

The conductor agent is itself an LLM with Cynefin + OODA baked into its system prompt. It classifies, selects agents, dispatches, and synthesizes. If two agents disagree, the conductor can invoke the Dialectical Synthesizer agent to resolve the conflict.

### Strengths and Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Each agent has tuned parameters | Context sync is hard (each agent has partial view) |
| Parallel execution (faster wall-clock) | More expensive (N parallel calls) |
| Genuine diversity of perspective | Conductor is a single point of failure |
| Can use different models per system | Integration/synthesis quality depends on conductor |
| Natural scaling | Complex orchestration code |

### When to Use

High-stakes decisions where diversity of perspective matters. Research contexts. When you need genuine creative tension between analytical and creative systems.

---

## Variation C: SEDA Pipeline (Staged Event-Driven Architecture)

### How It Works

Thinking systems are organized as **stages in a processing pipeline**. The problem enters at stage 1, flows through each stage, and each stage enriches the context before passing it forward. Stages within a phase can run in parallel. The pipeline topology is configured at startup based on the Cynefin domain.

### Mechanical Detail

```
Problem
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Stage 0: ORIENT                                  │
│ ┌──────────┐ ┌────────────┐ ┌──────────────┐   │
│ │ Cynefin  │ │Sensemaking │ │ Metacog      │   │  ← Parallel
│ │Classifier│ │            │ │ Check        │   │
│ └────┬─────┘ └─────┬──────┘ └──────┬───────┘   │
│      └──────────────┼──────────────┘            │
│                     ▼ merge                      │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│ Stage 1: FRAME (conditional on domain)           │
│ If novel: [First Principles → Inversion]         │
│ If complex: [Design Thinking → Socratic]         │
│ If complicated: [Systems Thinking → TOC]         │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│ Stage 2: GENERATE                                │
│ ┌──────────┐ ┌────────────┐ ┌──────────────┐   │
│ │Divergent │ │Lateral     │ │Latticework   │   │  ← Parallel
│ └──────────┘ └────────────┘ └──────────────┘   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│ Stage 3: EVALUATE (sequential)                   │
│ Convergent → Bayesian → Pre-mortem → 2nd-Order  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│ Stage 4: DECIDE                                  │
│ Bounded Rationality Governor → Final Selection   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
                   Output
```

### Thinking Systems Live As: **Pipeline stage handlers**

Each stage is a class with a `process()` method. Stages can be parallel (multiple systems within a stage) or sequential (ordered chain). The pipeline configuration is a data structure, not hardcoded logic.

```python
@dataclass
class PipelineStage:
    name: str
    systems: list[str]
    parallel: bool = False
    condition: str = None  # e.g., "domain == 'novel'"

@dataclass
class PipelineConfig:
    stages: list[PipelineStage]
    max_cycles: int = 3

# Example: Complex domain pipeline
COMPLEX_PIPELINE = PipelineConfig(stages=[
    PipelineStage("orient", ["cynefin", "sensemaking", "metacog"], parallel=True),
    PipelineStage("frame", ["design_thinking", "socratic"], parallel=False),
    PipelineStage("generate", ["divergent", "lateral", "latticework"], parallel=True),
    PipelineStage("evaluate", ["convergent", "bayesian", "pre_mortem", "second_order"], parallel=False),
    PipelineStage("decide", ["bounded_rationality"], parallel=False),
    PipelineStage("reflect", ["metacog"], parallel=False),
])

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.context = {}

    def run(self, problem: str) -> dict:
        self.context["problem"] = problem

        for cycle in range(self.config.max_cycles):
            for stage in self.config.stages:
                if stage.parallel:
                    results = self._run_parallel(stage.systems)
                else:
                    results = self._run_sequential(stage.systems)

                self.context[stage.name] = results

            # Check if we should cycle again
            if self.context.get("confidence", 0) > 0.85:
                break

        return self.context
```

### Selection Method: **Static pipeline configuration**

The Cynefin classification (which can use the fast heuristic from Doc 12) selects the pipeline topology at startup. Once selected, the pipeline is fixed for that run. Systems don't dynamically select each other — the order is predetermined.

### Strengths and Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Clear, predictable data flow | Less adaptive mid-run |
| Easy to parallelize stages | Hard to loop back (pipeline goes forward) |
| Easy to test (each stage independently) | Fixed topology may not fit every problem variant |
| Great for batch processing | Over-processes simple problems |
| Natural logging/observability | Configuration complexity for many domains |

### When to Use

Batch processing (analyze 100 proposals). Workflow automation. When you need predictability and auditability. When problems within a category are structurally similar.

---

## Variation D: Tool-Use Agent

### How It Works

Each thinking system is registered as a **tool** (function) that the LLM can call. The agent's system prompt encodes the Master Sequence and integration rules as guidelines. The LLM dynamically decides which tools to call, in what order, based on the problem. The orchestration logic is in the LLM's reasoning, not in external code.

### Mechanical Detail

```
┌──────────────────────────────────────────────────┐
│  LLM AGENT                                        │
│  System prompt: Kernel + Integration Rules         │
│                                                    │
│  Available tools:                                  │
│  ┌──────────────────┐ ┌──────────────────┐        │
│  │cynefin_classify  │ │first_principles  │        │
│  │                  │ │                  │        │
│  │input: {problem}  │ │input: {problem,  │        │
│  │output: {domain,  │ │        depth}    │        │
│  │  confidence}     │ │output: {truths,  │        │
│  │                  │ │  reframed}       │        │
│  └──────────────────┘ └──────────────────┘        │
│  ┌──────────────────┐ ┌──────────────────┐        │
│  │inversion         │ │pre_mortem        │        │
│  │                  │ │                  │        │
│  │input: {goal}     │ │input: {plan}     │        │
│  │output: {failures,│ │output: {failure_ │        │
│  │  risks}          │ │  modes, mitigations}     │
│  └──────────────────┘ └──────────────────┘        │
│  ┌──────────────────┐ ┌──────────────────┐        │
│  │bayesian_update   │ │divergent_generate│        │
│  │                  │ │                  │        │
│  │input: {hyp,      │ │input: {problem,  │        │
│  │  prior, evidence}│ │  count}          │        │
│  │output: {posterior,│ │output: {ideas}   │        │
│  │  impact}         │ │                  │        │
│  └──────────────────┘ └──────────────────┘        │
│  ... (one tool per thinking system)                │
│                                                    │
│  AGENT LOOP:                                       │
│  1. LLM reads problem + system prompt              │
│  2. LLM decides: "I should classify first"         │
│  3. Calls cynefin_classify tool                    │
│  4. Reads result: "Complex domain"                 │
│  5. LLM decides: "For complex, I need probing"     │
│  6. Calls first_principles tool                    │
│  7. Reads result, decides next tool...             │
│  8. Eventually: stop_reason == "end_turn"          │
└──────────────────────────────────────────────────┘
```

### Thinking Systems Live As: **Registered tools (functions)**

Each tool has a name, description (crucial — this is how the LLM decides to use it), and an input/output schema. The tool implementation calls the LLM again with a specialized prompt for that thinking system.

```python
tools = [
    {
        "name": "cynefin_classify",
        "description": (
            "Classify a problem into a Cynefin domain "
            "(clear/complicated/complex/chaotic/novel). "
            "ALWAYS call this first before other thinking tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to classify"}
            },
            "required": ["problem"]
        }
    },
    {
        "name": "first_principles",
        "description": (
            "Decompose a problem to its fundamental truths, stripping "
            "away assumptions and conventions. Then rebuild novel solutions "
            "from those truths. Best for novel or complex problems where "
            "conventional approaches have failed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "depth": {
                    "type": "integer",
                    "description": "How many layers to decompose (1-5)",
                    "default": 3
                }
            },
            "required": ["problem"]
        }
    },
    {
        "name": "inversion",
        "description": (
            "Ask 'how would I guarantee failure at this goal?' to identify "
            "risks and reframe the problem. Use at the START of analysis "
            "(to understand the problem) and at the END (to stress-test "
            "the solution). Bookend everything with inversion."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The goal to invert"}
            },
            "required": ["goal"]
        }
    },
    {
        "name": "pre_mortem",
        "description": (
            "Imagine a plan has already FAILED and generate specific "
            "reasons why. Use before committing to any significant "
            "decision. Produces ranked failure modes with mitigations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "The plan to stress-test"}
            },
            "required": ["plan"]
        }
    },
    {
        "name": "bayesian_update",
        "description": (
            "Update confidence in a hypothesis given new evidence. "
            "Tracks prior → posterior shift. Use whenever new data "
            "arrives that should change your belief."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "prior_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "new_evidence": {"type": "string"}
            },
            "required": ["hypothesis", "prior_confidence", "new_evidence"]
        }
    },
    {
        "name": "divergent_generate",
        "description": (
            "Generate maximum quantity and variety of ideas/solutions. "
            "No evaluation, no judgment — pure creative expansion. "
            "ALWAYS use before convergent_evaluate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "minimum_ideas": {"type": "integer", "default": 10}
            },
            "required": ["problem"]
        }
    },
    {
        "name": "convergent_evaluate",
        "description": (
            "Filter and rank ideas against explicit criteria. "
            "Narrow from many options to the best few. "
            "ALWAYS use after divergent_generate, never before."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ideas": {"type": "array", "items": {"type": "string"}},
                "criteria": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["ideas"]
        }
    },
    {
        "name": "second_order_thinking",
        "description": (
            "Trace consequences of consequences. Asks 'and then what?' "
            "recursively. Reveals hidden ripple effects and unintended "
            "consequences. Use for decisions with systemic impact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "depth": {"type": "integer", "default": 3, "description": "How many orders to trace"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "systems_mapping",
        "description": (
            "Map the key variables, causal relationships, and feedback "
            "loops in a system. Identifies reinforcing loops, balancing "
            "loops, and leverage points. Use for complicated/complex "
            "problems with many interacting parts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "system_description": {"type": "string"}
            },
            "required": ["system_description"]
        }
    },
    {
        "name": "socratic_questioning",
        "description": (
            "Probe a claim or assumption through structured questioning. "
            "Surfaces hidden contradictions and unstated assumptions. "
            "Use when something 'feels obvious' but hasn't been examined."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The claim or assumption to probe"}
            },
            "required": ["claim"]
        }
    },
    {
        "name": "dialectical_synthesis",
        "description": (
            "Resolve a contradiction between two positions by finding "
            "a higher-order synthesis that integrates both. Use when "
            "two thinking systems or perspectives produce conflicting outputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thesis": {"type": "string"},
                "antithesis": {"type": "string"}
            },
            "required": ["thesis", "antithesis"]
        }
    },
    {
        "name": "sensemaking",
        "description": (
            "Construct a plausible narrative from ambiguous or incomplete "
            "data. Extract patterns, build coherence. Use when the "
            "situation is unclear and you need a working story to act on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["observations"]
        }
    },
    {
        "name": "critical_evaluation",
        "description": (
            "Apply intellectual standards (clarity, accuracy, logic, "
            "relevance, evidence quality) to evaluate a claim or "
            "argument. Identifies fallacies and weak evidence. "
            "Use as a quality gate on any output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "evidence": {"type": "string"}
            },
            "required": ["claim"]
        }
    },
    {
        "name": "lateral_provocation",
        "description": (
            "Introduce a random stimulus or deliberate provocation "
            "to break established thinking patterns. Forces novel "
            "associations. Use when stuck in conventional solutions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "provocation_type": {
                    "type": "string",
                    "enum": ["random_entry", "reversal", "exaggeration", "distortion"],
                    "default": "random_entry"
                }
            },
            "required": ["problem"]
        }
    },
    {
        "name": "theory_of_constraints",
        "description": (
            "Identify the single bottleneck limiting system performance. "
            "Focus all improvement effort there. Use for process, "
            "operational, or throughput problems."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "system_description": {"type": "string"},
                "desired_outcome": {"type": "string"}
            },
            "required": ["system_description"]
        }
    },
]

# The agent loop
def run_tool_agent(problem: str):
    messages = [{"role": "user", "content": problem}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=KERNEL_SYSTEM_PROMPT,  # Integration rules + metacognition
            tools=tools,
            messages=messages,
        )

        # Collect assistant message
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_thinking_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "user", "content": tool_results})

    return extract_final_answer(messages)
```

### Selection Method: **LLM chooses tools based on descriptions**

The tool descriptions encode when to use each system. The system prompt encodes the integration rules (classify first, diverge before converge, inversion bookends). The LLM reads the problem, the rules, and the tool descriptions, then decides which to call. The orchestration is emergent from the LLM's reasoning.

### Strengths and Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Most flexible — agent adapts per problem | Relies on LLM judgment for tool selection |
| Leverages native tool-use infrastructure | May under-use unfamiliar tools |
| Easy to add/remove systems (just add/remove tools) | Tool description quality determines selection quality |
| Natural conversation flow | No parallelism (sequential tool calls) |
| Fastest to build | May not follow integration rules consistently |

### When to Use

**MVP.** Fastest path to a working prototype. Start here, then graduate to Variation E once you understand which systems are actually valuable for your use case.

---

## Variation E: Hybrid — Kernel as Prompt + OS as Router + Apps as Tools

### How It Works

The three-layer model implemented literally:
- **Kernel** = always-on monitoring baked into the system prompt (not a tool, not optional)
- **OS** = code-level router that runs BEFORE the LLM (heuristic classifier, playbook selector)
- **Applications** = thinking systems registered as tools, invoked within the selected playbook

### Mechanical Detail

```
Problem arrives
      │
      ▼
┌──────────────────────────────────────────┐
│  OS LAYER (Python code, no LLM)          │
│                                           │
│  1. DualProcessRouter.route(problem)      │  ← Heuristic, 0ms
│     → System 1? → single LLM call, done  │
│     → System 2 Light? → select 3-4 tools │
│     → System 2 Deep? → full playbook     │
│                                           │
│  2. FastCynefinClassifier.classify()      │  ← Heuristic first,
│     → confident? → use heuristic domain  │     LLM fallback if <55%
│     → uncertain? → one LLM call          │
│                                           │
│  3. PlaybookSelector.select(domain)       │  ← Code lookup
│     → Returns ordered list of tools       │
│       + phase sequence                    │
│       + max iterations                    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  KERNEL (system prompt, every LLM call)   │
│                                           │
│  "You are a structured thinking agent.    │
│   ALWAYS ACTIVE:                          │
│   - Monitor: System 1 vs System 2?       │
│   - Monitor: Confidence calibrated?      │
│   - Monitor: Effort proportional?        │
│   - Before each tool: brief metacog note │
│   - After each tool: should I loop back? │
│                                           │
│   INTEGRATION RULES:                      │
│   - Classify before choose               │
│   - Diverge before converge              │
│   - Inversion bookends                   │
│   ..."                                    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  APPLICATION TOOLS (same as Variation D)  │
│                                           │
│  The OS provides a "suggested_tools"      │
│  list, but the LLM can deviate if its     │
│  Kernel monitoring says to.               │
│                                           │
│  OS says: [first_principles, inversion,   │
│            pre_mortem, bayesian]           │
│                                           │
│  LLM might: call first_principles, notice │
│  the problem was misframed, loop back to  │
│  socratic_questioning before continuing   │
└──────────────────────────────────────────┘
```

### Thinking Systems Live As: **Three different things simultaneously**

| Layer | Systems | Implemented As |
|-------|---------|---------------|
| Kernel | Metacognition, Dual-Process awareness, Bounded Rationality, Bias Detection | **System prompt text** — injected into every LLM call |
| OS | Cynefin, OODA rhythm, Playbook selection, Latticework | **Python code** — heuristic classifiers, state machines, configuration |
| Application | First Principles, Inversion, Pre-mortem, Bayesian, etc. | **Registered tools** — same as Variation D |

```python
class HybridAgent:
    def __init__(self):
        # OS Layer (code)
        self.router = DualProcessRouter()
        self.classifier = FastCynefinClassifier()
        self.memory = MemoryManager()

        # Application Layer (tools)
        self.tools = THINKING_TOOLS  # Same tool definitions as Variation D

        # Kernel Layer (prompt)
        self.kernel_prompt = KERNEL_SYSTEM_PROMPT

    def solve(self, problem: str, stakes: str = None) -> dict:
        # ── OS: Route ──────────────────────────────────
        route = self.router.route(problem, stakes)

        if route.route == CognitiveRoute.SYSTEM_1:
            # Single call, kernel still active in system prompt
            return self._single_call(problem)

        # ── OS: Classify ───────────────────────────────
        domain = self.classifier.classify(problem)

        # ── OS: Select playbook ────────────────────────
        playbook = self._get_playbook(domain.domain, route.route)

        # ── OS: Build context with tiered memory ───────
        context = self.memory.build_context_window(phase="orient")

        # ── Agent loop with Kernel + Tools ─────────────
        messages = [{
            "role": "user",
            "content": f"""Problem: {problem}

Domain classification: {domain.domain} ({domain.confidence:.0%})
Suggested tool sequence: {playbook['tools']}

{context}

Apply the suggested tools in order, but deviate if your
metacognitive monitoring tells you to. Start now."""
        }]

        # Standard tool-use loop (same as Variation D)
        # but with:
        # - Kernel always in system prompt
        # - Memory managed between tool calls
        # - OS providing guardrails (max iterations, budget)
        result = self._run_agent_loop(messages, playbook)

        return result

    def _get_playbook(self, domain: str, route) -> dict:
        playbooks = {
            "clear": {
                "tools": ["critical_evaluation"],
                "max_iterations": 1,
            },
            "complicated": {
                "tools": ["systems_mapping", "critical_evaluation",
                          "bayesian_update", "inversion"],
                "max_iterations": 3,
            },
            "complex": {
                "tools": ["first_principles", "divergent_generate",
                          "convergent_evaluate", "pre_mortem",
                          "second_order_thinking", "inversion"],
                "max_iterations": 5,
            },
            "chaotic": {
                "tools": ["sensemaking", "inversion"],
                "max_iterations": 2,
            },
            "novel": {
                "tools": ["first_principles", "lateral_provocation",
                          "divergent_generate", "convergent_evaluate",
                          "pre_mortem", "inversion"],
                "max_iterations": 5,
            },
        }

        book = playbooks.get(domain, playbooks["complex"])

        # If S2 Light, trim to top 3-4 tools
        if route == CognitiveRoute.SYSTEM_2_LIGHT:
            book["tools"] = book["tools"][:4]
            book["max_iterations"] = 2

        return book
```

### Selection Method: **Heuristic routing → playbook suggestion → LLM execution with override**

Three layers of selection:
1. **Heuristic** (code): Routes to System 1/2 Light/2 Deep
2. **Playbook** (code): Suggests a tool sequence based on domain
3. **LLM** (runtime): Follows the suggestion but can deviate if metacognition flags an issue

### Strengths and Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Best cost efficiency (heuristic gates) | Most complex to design |
| Kernel always on (no opt-out) | Three layers of logic to maintain |
| Playbooks provide structure | Requires tuning the heuristic thresholds |
| LLM retains flexibility to deviate | |
| Tiered memory controls context | |
| Scales from 1 call to 18 calls adaptively | |

### When to Use

**Production.** This is what you ship. It combines the cost efficiency of heuristic routing, the structure of playbooks, and the flexibility of tool-use agents.

---

## Variation F: Dual-Process Gated (Cost-Optimized Production)

### How It Works

Variation E with the Dual-Process Entry Gate from Doc 12 fully implemented. The key insight: **most problems don't need the full thinking pipeline**. The gate decides in <1ms, with zero LLM calls, whether to think fast or slow.

### Mechanical Detail

```
Problem arrives
      │
      ▼
┌──────────────────────────────────────────┐
│  DUAL-PROCESS GATE (code, 0 LLM calls)   │
│                                           │
│  Scans for complexity signals:            │
│  - "strategy" "risk" "uncertain" → HIGH   │
│  - "analyze" "compare" "plan" → MEDIUM    │
│  - "what is" "how to" "list" → LOW        │
│                                           │
│  Routes:                                  │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ │
│  │System 1 │ │S2 Light  │ │S2 Deep    │ │
│  │1 call   │ │3-5 calls │ │10-18 calls│ │
│  │~65% of  │ │~25% of   │ │~10% of    │ │
│  │problems │ │problems  │ │problems   │ │
│  └────┬────┘ └────┬─────┘ └────┬──────┘ │
└───────┼───────────┼────────────┼─────────┘
        │           │            │
        ▼           ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌─────────────┐
   │ Single  │ │ Targeted│ │ Full Hybrid │
   │ LLM     │ │ 3-4     │ │ Pipeline    │
   │ call    │ │ tools   │ │ (Var. E)    │
   │ with    │ │ with    │ │             │
   │ Kernel  │ │ Kernel  │ │             │
   │ prompt  │ │ prompt  │ │             │
   └─────────┘ └─────────┘ └─────────────┘
```

### The Three Paths

**Path 1 — System 1 (single call, ~65% of problems)**

Simple questions, well-defined tasks, factual lookups. The Kernel system prompt is still active (metacognition never turns off), but no thinking tools are invoked. Just a high-quality direct response.

```python
def _system_1_response(self, problem: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=self.kernel_prompt,  # Kernel always on
        messages=[{"role": "user", "content": problem}],
        # No tools provided — fast, direct response
    )
    return {"answer": response.content[0].text, "calls": 1, "route": "system_1"}
```

**Path 2 — System 2 Light (3-5 calls, ~25% of problems)**

Moderate analysis. The heuristic router picks 3-4 relevant tools based on problem keywords. No full Cynefin classification needed.

```python
def _system_2_light(self, problem: str, selected_tools: list[str]) -> dict:
    # Filter tool registry to only the selected tools
    active_tools = [t for t in self.tools if t["name"] in selected_tools]

    messages = [{"role": "user", "content": f"""Analyze this problem using the available tools.
Problem: {problem}

Apply each tool once, then synthesize a final answer."""}]

    return self._run_agent_loop(messages, active_tools, max_iterations=5)
```

**Path 3 — System 2 Deep (10-18 calls, ~10% of problems)**

Full Variation E pipeline: Cynefin classify → playbook → full tool sequence → synthesis → reflection.

### Thinking Systems Live As: **Tiered deployment**

| Path | Kernel | Thinking Systems | Memory |
|------|--------|-----------------|--------|
| System 1 | System prompt (always) | None (direct response) | None needed |
| S2 Light | System prompt (always) | 3-4 tools (heuristic-selected) | Working memory only |
| S2 Deep | System prompt (always) | Full tool registry + playbook | Full tiered memory |

### Cost Profile

| Metric | System 1 | S2 Light | S2 Deep |
|--------|----------|----------|---------|
| LLM calls | 1 | 3-5 | 10-18 |
| Latency | ~2s | ~10s | ~45s |
| Cost (est.) | $0.003 | $0.015 | $0.05 |
| Frequency | ~65% | ~25% | ~10% |
| **Weighted avg cost** | | | **~$0.009/problem** |

Compared to running the full pipeline for everything (~$0.05/problem), the gated architecture is **~5x cheaper** on average.

### When to Use

**Cost-optimized production.** Same quality as Variation E for hard problems, but doesn't waste resources on easy ones. This is the final evolution.

---

## Decision Matrix: Which Agent to Build

```
Are you prototyping?
├─ Yes → Variation A (single prompt) or D (tool-use)
└─ No
   ├─ Do you need diversity of perspective?
   │  ├─ Yes → Variation B (multi-agent ensemble)
   │  └─ No
   │     ├─ Is this batch processing?
   │     │  ├─ Yes → Variation C (SEDA pipeline)
   │     │  └─ No
   │     │     ├─ Do you care about cost?
   │     │     │  ├─ Yes → Variation F (dual-process gated)
   │     │     │  └─ No → Variation E (hybrid)
   │     │     └─
   │     └─
   └─
```

### Recommended Build Path

```
Week 1:  Variation D (tool-use MVP, 6 Tier-1 tools)
         → Validate which systems are useful

Week 2:  Variation E (add Kernel prompt + OS router + memory)
         → Add Tier-2 tools as needed

Week 3:  Variation F (add Dual-Process gate)
         → Optimize cost, tune heuristic thresholds

Week 4+: Variation B components (spin out creative systems
         as separate high-temperature agents)
         → Only for systems that benefit from parameter tuning
```

---

## Appendix: Complete Tool Registry for Copy-Paste

All tools from Variation D/E/F in one list, with tier annotations:

| Tool Name | Tier | Phase | Description (for LLM) |
|-----------|------|-------|----------------------|
| `cynefin_classify` | OS | Orient | Classify problem domain. ALWAYS call first. |
| `first_principles` | 1 | Frame/Generate | Decompose to bedrock truths, rebuild from scratch. |
| `inversion` | 1 | Frame/Evaluate | How to guarantee failure? Bookend everything. |
| `pre_mortem` | 1 | Evaluate | This plan failed — why? Ranked failure modes. |
| `bayesian_update` | 1 | Evaluate/Reflect | Update confidence given new evidence. |
| `critical_evaluation` | 1 | Evaluate | Apply intellectual standards. Quality gate. |
| `divergent_generate` | 1 | Generate | Maximum ideas, no judgment. Before convergent. |
| `convergent_evaluate` | 1 | Evaluate | Filter and rank against criteria. After divergent. |
| `systems_mapping` | 2 | Perceive/Frame | Map variables, feedback loops, leverage points. |
| `socratic_questioning` | 2 | Frame/Evaluate | Probe assumptions through structured questions. |
| `sensemaking` | 2 | Perceive/Frame | Construct narrative from ambiguous data. |
| `second_order_thinking` | 2 | Evaluate | And then what? Recursive consequence tracing. |
| `lateral_provocation` | 3 | Generate | Random stimulus to break patterns. |
| `dialectical_synthesis` | 3 | Frame/Generate | Resolve contradictions via thesis-antithesis-synthesis. |
| `theory_of_constraints` | 3 | Frame/Decide | Find the one bottleneck. Focus there. |
