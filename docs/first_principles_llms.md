# First Principles: LLMs and Their Usage

## What an LLM is

An LLM is a stateless function: text in, text out. It has no goals, no memory, no agency, no persistence. It predicts the next tokens given a context window. Everything else — tools, agents, orchestration, safety — is architecture humans build around this function.

## The Four Primitives

Everything in the LLM ecosystem reduces to four irreducible concepts:

### 1. Content
The atom of information. Everything flowing through the system is content: text, structured data, images, errors, observations.

Content always carries:
- **Substance** — the information itself
- **Type** — what kind of information (instruction, response, observation, signal)
- **Intent** — what is it for (request, inform, delegate, report)
- **Source** — where it came from (which identity produced it)
- **Timestamp** — when it was produced

In practice, current LLM APIs encode this as `{"role": "...", "content": "..."}` which is a lossy compression of these dimensions. The "role" field conflates source and intent into one value.

### 2. Transform
The atom of computation. Every operation is: content in → content out.

An LLM completion is a transform. A tool call is a transform. An agent loop is a transform. A guardian audit is a transform. An orchestrator is a transform. From the outside, they all have the same shape.

Transforms have attributes:
- **Pure vs effectful** — does it change the world, or just produce new content?
- **Deterministic vs nondeterministic** — same input, same output?
- **Bounded vs unbounded** — does it have a time/cost/step limit?

Pure transforms can be cached, retried, replayed. Effectful transforms cannot. An LLM completion is pure (stateless prediction). A tool that writes to a database is effectful. This distinction matters for safety, testing, and optimization.

### 3. Identity
The atom of addressability and discovery. Without identity, you have anonymous functions. With identity, you have discoverable, trustable, routable entities.

Identity contains:
- **Name** — how to refer to it
- **Schema** — self-description of what it does and what it expects (the "Agent Card", the tool definition, the MCP manifest — all the same concept)
- **Trust** — how much should other entities trust its outputs

Trust is critical and absent from most frameworks. The output of a verified database query and the output of an LLM hallucination have fundamentally different epistemic status, but current systems treat them identically — both are just strings. Trust should propagate: if a high-trust agent delegates to a low-trust agent, the result inherits the lower trust level.

### 4. Memory
The atom of persistence. Without memory, every transform is stateless — no history, no learning, no context.

Memory is:
- **Content** — what's being remembered
- **Scope** — who can access it (private, shared, global)
- **Access control** — read_only, append_only, read_write

This maps directly to real concepts:
- Conversation history = memory with access_control=append_only
- The frozen part of a system prompt = memory with access_control=read_only
- Learned preferences / techniques = memory with access_control=read_write
- A budget or safety constitution = memory with access_control=read_only (a constraint)

The frozen/mutable distinction in prompts is just two memory regions with different access controls. This is how you get safe self-modification: the agent can write to read_write memory (learning) but cannot touch read_only memory (its core values and constraints).

## Composition

Primitives compose through five operators:

- **Sequence** — A then B. Output of A becomes input of B.
- **Parallel** — A and B simultaneously. Both get same input, outputs merge.
- **Conditional** — A or B depending on content.
- **Loop** — A repeatedly until a condition is met.
- **Recursion** — A invokes things that can invoke A.

Every architecture in the LLM ecosystem is some combination:
- **Chatbot** = single Transform, no loop
- **Tool-using LLM** = Transform + Conditional(tool needed?) + Sequence(call tool, feed back)
- **Agent** = Loop(Transform → Conditional → side-effect Transform) with Memory
- **Pipeline orchestrator** = Sequence of Transforms
- **Parallel orchestrator** = Parallel Transforms + Aggregate
- **Dynamic orchestrator** = Loop(Transform decides next Transform) — the LLM is the control flow
- **Multi-agent** = Recursion of Transforms communicating via Content
- **Guardian + Agent** = Loop(Agent Transform → Guardian Transform with Conditional: continue/halt)
- **Cycle runner** = Loop with time boundaries and Guardian checkpoint between iterations

## Fundamental Pattern: Boundary

A Boundary is read_only Memory that constrains composition.

This is the constitution, the sandbox, the frozen prompt, the budget limit, the safety rules. All the same concept: persistent, immutable constraints that shape what the system can and cannot do.

Boundaries are what make autonomous systems controllable. Without boundaries, a Loop with Memory and self-modification is unbounded — it can drift anywhere. With boundaries, it can learn and adapt within defined limits.

## Cross-Cutting Dimension: Time

Time is not a fifth primitive but a dimension that lives on all four:
- Content + time = ordering, freshness
- Transform + time = duration, deadline, timeout
- Memory + time = TTL, history, staleness
- Identity + time = lifecycle (created, active, retired)

Time matters because an LLM system is a distributed system. Multiple transforms run concurrently, produce results at different speeds, and need coordination. A transform that takes 1ms and one that takes 30 minutes have the same shape but fundamentally different operational semantics.

## How Everything Maps

### APIs
An API is just a network boundary around a Transform. You serialize Content, send it over HTTP, the Transform executes remotely, and Content comes back. The API adds Identity (authentication, model selection) and often constrains Memory (context window limits, rate limits).

### Tools / Function Calling
A Tool is a Transform with a Schema (its Identity's self-description). The LLM reads the Schema, produces structured Content requesting a tool call, the system executes the Transform, and feeds the result Content back. The LLM is still just doing text → text. The feedback loop with tools is what gives it agency.

### MCP (Model Context Protocol)
MCP is a standardization of Identity + Schema for tools. Instead of every application inventing its own tool interface, MCP defines one universal way to describe tools, resources, and prompts. It's the interoperability layer — build a tool once, any MCP-compatible system can use it. MCP assumes an asymmetric relationship: a smart caller (agent) invoking tools (servers). The tool does what it's told and returns a result.

### A2A (Agent-to-Agent Protocol)
A2A standardizes communication between peers — Transforms that are both autonomous, both nondeterministic, both potentially long-running. It adds: task lifecycle (submitted → working → needs_input → completed), async patterns, negotiation (back-and-forth), and discovery (Agent Cards = Identity with Schema). A2A exists because MCP's assumptions break down when both sides have autonomy.

### Agents
An agent is: Loop(LLM Transform → Conditional(done?) → Tool Transforms) with Memory. From the outside, it's just another Transform: content in, content out. Inside, it's an autonomous loop. The loop is what creates agency. The LLM decides which Transforms to invoke and when.

### Self-Modification
An agent that can modify its own read_write Memory (mutable prompt sections). It learns techniques, preferences, knowledge. The read_only Memory (frozen prompt) prevents it from modifying its core objectives. This is safe self-improvement within Boundaries.

### Agent Spawning
An agent that creates new Identities (sub-agents) with their own Transforms, Memory, and Boundaries. The parent composes them via Sequence, Parallel, or Delegation. The spawned agent is, from the parent's perspective, just another Transform it can invoke.

### Tool Creation
An agent that creates new Transforms at runtime. The architecture becomes a runtime decision rather than a design-time decision. A tool is just a Transform with an Identity — creating one is writing a function and registering its Schema.

### Guardians
A Guardian is a Transform that reads another entity's Memory (trace, prompt edits, outputs) and produces a verdict Content (ok/warning/drift/halt). It enforces Boundaries. It should have independent Identity (different model, different prompt) so the agent it watches cannot influence its judgment.

### Orchestration Spectrum
Orchestration ranges from tight to loose:
- **Tight**: Hardcoded Sequence of specific Transforms. Predictable, rigid.
- **Config-driven**: Sequence/Parallel/Conditional defined in data, not code.
- **LLM-driven**: An LLM Transform reads available Identities (via Schema/Registry) and decides the composition at runtime. Flexible, unpredictable.

The more you let an LLM decide the composition, the more powerful and less predictable the system becomes.

### Same-hardware vs Cross-network
When Transforms are in the same process, composition is direct function calls with shared Memory. When they're on the same machine but different processes, you need a protocol (even lightweight). When they're across networks, you need full serialization, discovery, and trust. The primitives are the same — Content, Transform, Identity, Memory — but the overhead of composition increases with distance and trust boundaries.

## The Hierarchy of Capability

Everything is content → content with feedback loops of increasing complexity:

```
No loop              = completion (chatbot)
One loop             = tool use (function calling, MCP)
Persistent loop      = agent (Claude Code style)
Loop + Memory write  = learning agent (self-modification)
Loop + spawning      = expanding agent (creates tools and agents)
Nested loops         = multi-agent system
Loops + Guardian     = controlled autonomous system
```

Each level adds capability and reduces predictability. The role of Boundaries (read_only Memory) is to constrain each level so it remains controllable.

## Key Insight

The LLM never changed. It's still just a stateless Transform: text in, text out. Everything else — agency, memory, tools, collaboration, self-improvement — is composition of primitives around it. The model is just the engine. The architecture is the vehicle. The boundaries are the road.
