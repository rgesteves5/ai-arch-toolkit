# 12 — Agent Refinements

Engineering-grade responses to six architectural critique points. This document is a companion to `11-AGENT-ARCHITECTURE.md` — it addresses what needs to change before Variation E ships.

---

## Critique 1: LLM Call Count — The Dual-Process Entry Gate

**Problem:** A full pipeline (classify → frame → generate → evaluate → reflect) with multiple systems per phase could be 10–20+ LLM calls for one problem. Most problems don't need that.

**Solution:** Build a Dual-Process Router as the first gate. System 1 handles the common case in a single call. System 2 escalates to the full thinking pipeline only when warranted.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class CognitiveRoute(Enum):
    SYSTEM_1 = "system_1"   # Single call, fast, heuristic
    SYSTEM_2_LIGHT = "s2_light"  # 3-5 calls, targeted
    SYSTEM_2_DEEP = "s2_deep"    # Full pipeline, 10-20 calls


@dataclass
class RoutingDecision:
    route: CognitiveRoute
    reasoning: str
    estimated_calls: int
    selected_systems: list[str]
    budget_tokens: int


class DualProcessRouter:
    """The entry gate. Decides whether to think fast or slow.
    This is NOT an LLM call — it's a fast heuristic classifier."""

    # Keyword/pattern signals for complexity
    COMPLEXITY_SIGNALS = {
        "high": [
            "strategy", "tradeoff", "trade-off", "consequences",
            "stakeholders", "long-term", "systemic", "unintended",
            "risk", "uncertainty", "ambiguous", "novel", "unprecedented",
            "crisis", "competitive", "bet-the-company", "irreversible",
        ],
        "medium": [
            "compare", "analyze", "evaluate", "design", "plan",
            "optimize", "improve", "why", "how should", "recommend",
            "complex", "multiple", "factors",
        ],
        "low": [
            "what is", "define", "list", "summarize", "explain",
            "how to", "steps", "simple", "quick", "just",
        ]
    }

    def route(self, problem: str, user_stakes: Optional[str] = None) -> RoutingDecision:
        """Fast heuristic routing — NO LLM call. O(n) string scan."""

        problem_lower = problem.lower()
        problem_length = len(problem.split())

        # Score complexity signals
        high_hits = sum(1 for k in self.COMPLEXITY_SIGNALS["high"] if k in problem_lower)
        med_hits = sum(1 for k in self.COMPLEXITY_SIGNALS["medium"] if k in problem_lower)
        low_hits = sum(1 for k in self.COMPLEXITY_SIGNALS["low"] if k in problem_lower)

        # Length heuristic: longer problem statements usually mean more complexity
        length_score = min(problem_length / 100, 1.0)

        # Composite score
        complexity = (high_hits * 3 + med_hits * 1.5 + low_hits * 0.5 + length_score * 2)

        # User override: if they say "critical", escalate regardless
        if user_stakes == "critical":
            complexity = max(complexity, 6)

        # Route decision
        if complexity < 2:
            return RoutingDecision(
                route=CognitiveRoute.SYSTEM_1,
                reasoning=f"Low complexity ({complexity:.1f}). Single-call response.",
                estimated_calls=1,
                selected_systems=[],  # No thinking systems — direct answer
                budget_tokens=2048,
            )
        elif complexity < 5:
            # Pick 2-3 targeted systems based on what the problem needs
            systems = self._select_targeted_systems(problem_lower, high_hits > 0)
            return RoutingDecision(
                route=CognitiveRoute.SYSTEM_2_LIGHT,
                reasoning=f"Medium complexity ({complexity:.1f}). Targeted analysis.",
                estimated_calls=len(systems) + 2,  # systems + classify + synthesize
                selected_systems=systems,
                budget_tokens=8192,
            )
        else:
            return RoutingDecision(
                route=CognitiveRoute.SYSTEM_2_DEEP,
                reasoning=f"High complexity ({complexity:.1f}). Full pipeline.",
                estimated_calls=15,  # estimate, capped by governor
                selected_systems=[],  # full pipeline selects dynamically
                budget_tokens=32768,
            )

    def _select_targeted_systems(self, problem: str, has_risk_signals: bool) -> list[str]:
        """For System 2 Light: pick 2-3 systems that match the problem."""
        systems = []

        if any(k in problem for k in ["why", "assumption", "root cause", "fundamental"]):
            systems.append("first_principles")
        if any(k in problem for k in ["risk", "fail", "wrong", "danger"]):
            systems.append("pre_mortem")
        if any(k in problem for k in ["consequence", "then what", "ripple", "long-term"]):
            systems.append("second_order")
        if any(k in problem for k in ["option", "choose", "compare", "alternative"]):
            systems.extend(["divergent", "convergent"])
        if any(k in problem for k in ["evidence", "confidence", "likely", "probability"]):
            systems.append("bayesian")
        if has_risk_signals:
            systems.append("inversion")

        # Default: if nothing matched, use the generalist trio
        if not systems:
            systems = ["critical_thinking", "inversion"]

        return systems[:4]  # Cap at 4 systems for light mode
```

### Call Budget Per Route

| Route | Estimated Calls | Latency (est.) | When to Use |
|-------|----------------|-----------------|-------------|
| System 1 | 1 | < 3s | Factual questions, simple requests, well-defined tasks |
| System 2 Light | 3–5 | 10–20s | Moderate analysis, targeted evaluation, comparison tasks |
| System 2 Deep | 10–20 | 30–90s | High-stakes strategy, novel problems, crisis analysis |

### Integration with the Agent

```python
class ThinkingAgent:
    def __init__(self):
        self.router = DualProcessRouter()
        self.kernel = Kernel()
        # ... rest of init

    def solve(self, problem: str, stakes: str = None) -> dict:
        # GATE: Dual-Process Router (no LLM call)
        route = self.router.route(problem, user_stakes=stakes)

        if route.route == CognitiveRoute.SYSTEM_1:
            # Single call. Kernel monitoring baked into system prompt.
            return self._system_1_response(problem)

        elif route.route == CognitiveRoute.SYSTEM_2_LIGHT:
            # Targeted: classify + run selected systems + synthesize
            return self._system_2_light(problem, route.selected_systems)

        else:
            # Full pipeline from 11-AGENT-ARCHITECTURE
            return self._system_2_deep(problem)
```

---

## Critique 2: Context Window — Tiered Memory Architecture

**Problem:** By phase 4–5, accumulated context (all prior insights, observations, OODA state) blows up the prompt. Each system dumps everything into the next call.

**Solution:** Three-tier memory, inspired by human memory architecture (and the research's own layered model).

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class MemoryTiers:
    """Three-tier memory system. Total context budget is managed explicitly."""

    # ── Tier 1: Working Memory ────────────────────────────────
    # Circular buffer. Holds the CURRENT phase's context only.
    # Oldest items get evicted when buffer is full.
    # This is what gets injected into every prompt.
    working: deque = field(default_factory=lambda: deque(maxlen=8))
    working_token_budget: int = 2048

    # ── Tier 2: Semantic Memory ───────────────────────────────
    # Extracted facts, conclusions, and decisions.
    # Compressed from working memory at phase transitions.
    # Consulted when the agent needs prior conclusions.
    semantic: list[dict] = field(default_factory=list)
    semantic_token_budget: int = 1024

    # ── Tier 3: Episodic Log ──────────────────────────────────
    # Compressed narrative of what happened.
    # Only accessed during the Reflect phase or when
    # the agent needs to understand its own history.
    episodic: list[dict] = field(default_factory=list)
    episodic_token_budget: int = 512


class MemoryManager:
    """Manages the three tiers. Handles promotion, compression, and eviction."""

    def __init__(self, total_context_budget: int = 4096):
        self.memory = MemoryTiers()
        self.total_budget = total_context_budget

    def add_to_working(self, item: dict):
        """Add a new item to working memory. Oldest auto-evicts."""
        self.memory.working.append(item)

    def consolidate(self, current_phase: str):
        """Called at phase transitions. Compress working → semantic."""
        if not self.memory.working:
            return

        # Extract the key facts/decisions from working memory
        consolidation_prompt = f"""Compress these working memory items into
key facts, decisions, and conclusions. Be extremely concise.
Discard process details, keep only results.

Items from phase '{current_phase}':
{list(self.memory.working)}

Return a JSON list of extracted facts, each under 20 words."""

        # This is one of the few "overhead" LLM calls — but it keeps
        # all subsequent calls smaller.
        extracted = self._call_llm_for_compression(consolidation_prompt)
        self.memory.semantic.extend(extracted)

        # Log the phase to episodic memory
        self.memory.episodic.append({
            "phase": current_phase,
            "summary": f"Completed {current_phase}. "
                       f"Key outputs: {len(extracted)} facts extracted.",
            "fact_count": len(extracted),
        })

        # Clear working memory for next phase
        self.memory.working.clear()

    def build_context_window(self, phase: str) -> str:
        """Build the context string for the current LLM call.
        Budget-aware: never exceeds total_context_budget."""

        sections = []

        # Always include: semantic memory (compressed facts)
        if self.memory.semantic:
            semantic_str = "\n".join(
                f"- {fact['text']}" for fact in self.memory.semantic[-20:]
            )
            sections.append(f"## Known Facts & Decisions\n{semantic_str}")

        # Include: current working memory
        if self.memory.working:
            working_str = "\n".join(
                f"- [{item.get('source', '?')}] {item.get('content', '')}"
                for item in self.memory.working
            )
            sections.append(f"## Current Working Context\n{working_str}")

        # Only during Reflect: include episodic log
        if phase == "reflect" and self.memory.episodic:
            episodic_str = "\n".join(
                f"- Phase {ep['phase']}: {ep['summary']}"
                for ep in self.memory.episodic
            )
            sections.append(f"## Process History\n{episodic_str}")

        return "\n\n".join(sections)

    def _call_llm_for_compression(self, prompt: str) -> list[dict]:
        """Compress working memory to semantic facts."""
        # Implementation: call LLM with low max_tokens, parse JSON
        raise NotImplementedError
```

### Memory Flow Per Phase

```
Phase N starts
  → Build context: semantic (compressed) + working (current)
  → Run thinking system(s)
  → Results go into working memory
  → Phase N ends

Phase transition
  → consolidate() extracts facts from working → semantic
  → Log summary to episodic
  → Clear working memory

Phase N+1 starts
  → Build context: semantic (now includes Phase N facts) + working (empty, filling)
  → ...
```

### Token Budget Example

| Tier | Budget | Contains | Persistence |
|------|--------|----------|-------------|
| Working | 2048 tokens | Current phase raw outputs | Cleared each phase |
| Semantic | 1024 tokens | Compressed facts from all prior phases | Grows, pruned by relevance |
| Episodic | 512 tokens | Phase summaries, process narrative | Grows, only accessed in Reflect |
| System prompt | ~1000 tokens | Kernel + integration rules | Fixed |
| Thinking system prompt | ~500 tokens | Current system instructions | Changes per call |
| **Total per call** | **~5000 tokens** | | **Bounded** |

---

## Critique 3: Structured Output — JSON Schema with Common Envelope

**Problem:** `_parse_output` is `raise NotImplementedError` but it's actually the hardest part. Each system asks for different output.

**Solution:** Common envelope schema that wraps every system's output, with a `system_specific` field for per-system data. Use JSON schema enforcement (Anthropic tool_use / function calling) to guarantee structure.

```python
# The universal output envelope — every system returns this shape.
COMMON_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        # ── Common fields (every system) ──────────────────────
        "confidence": {
            "type": "number",
            "minimum": 0, "maximum": 1,
            "description": "Calibrated confidence in this output"
        },
        "insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key insights, each under 30 words"
        },
        "metacog_assessment": {
            "type": "string",
            "description": "Brief self-assessment of reasoning quality"
        },
        "suggested_next": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which thinking system(s) should run next"
        },
        "should_loop_back": {
            "type": "boolean",
            "description": "Should we return to an earlier phase?"
        },
        "loop_target": {
            "type": "string",
            "enum": ["orient", "perceive", "frame", "generate",
                     "evaluate", "decide", "reflect"],
            "description": "If looping back, which phase"
        },

        # ── System-specific fields ────────────────────────────
        "system_specific": {
            "type": "object",
            "description": "Output specific to this thinking system"
        }
    },
    "required": ["confidence", "insights", "metacog_assessment"]
}

# Per-system schemas for the system_specific field:

FIRST_PRINCIPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "assumptions_found": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "assumption": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["bedrock_truth", "convention",
                                 "habit", "analogy", "uncertain"]
                    }
                }
            }
        },
        "bedrock_truths": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reframed_problem": {"type": "string"},
    }
}

PRE_MORTEM_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_modes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                    "detectability": {
                        "type": "string",
                        "enum": ["early", "late", "never"]
                    },
                    "mitigation": {"type": "string"}
                }
            }
        },
        "success_factors": {
            "type": "array",
            "items": {"type": "string"}
        },
        "critical_mitigations": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

BAYESIAN_SCHEMA = {
    "type": "object",
    "properties": {
        "prior": {"type": "number"},
        "likelihood": {"type": "number"},
        "posterior": {"type": "number"},
        "evidence_impact": {
            "type": "string",
            "enum": ["strongly_confirms", "weakly_confirms",
                     "neutral", "weakly_disconfirms",
                     "strongly_disconfirms"]
        },
        "key_uncertainties": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# Registry: system name → system_specific schema
SYSTEM_SCHEMAS = {
    "first_principles": FIRST_PRINCIPLES_SCHEMA,
    "pre_mortem": PRE_MORTEM_SCHEMA,
    "bayesian": BAYESIAN_SCHEMA,
    "inversion": {  # ... etc
        "type": "object",
        "properties": {
            "failure_guarantees": {"type": "array", "items": {"type": "string"}},
            "current_risks": {"type": "array", "items": {"type": "string"}},
            "inverted_insights": {"type": "array", "items": {"type": "string"}},
        }
    },
    # ... one per system
}
```

### Using It in the Agent (Anthropic tool_use)

```python
def _execute_system(self, system_name: str, problem: str, context: str) -> dict:
    """Execute a thinking system with enforced JSON output."""

    system_prompt = self.systems[system_name].get_prompt(problem, context)
    specific_schema = SYSTEM_SCHEMAS.get(system_name, {"type": "object"})

    # Merge common envelope with system-specific schema
    full_schema = {**COMMON_OUTPUT_SCHEMA}
    full_schema["properties"]["system_specific"] = specific_schema

    response = self.client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system="You are a structured thinking module. Always respond "
               "with valid JSON matching the provided schema.",
        messages=[{"role": "user", "content": system_prompt}],
        # Force structured output via tool_use
        tools=[{
            "name": f"thinking_output_{system_name}",
            "description": f"Structured output from {system_name} analysis",
            "input_schema": full_schema,
        }],
        tool_choice={"type": "tool", "name": f"thinking_output_{system_name}"},
    )

    # Extract the structured tool_use block — guaranteed valid JSON
    for block in response.content:
        if block.type == "tool_use":
            return block.input  # Already parsed dict, schema-validated

    raise RuntimeError(f"No structured output from {system_name}")
```

---

## Critique 4: Cynefin Chicken-and-Egg — Fast Heuristic with LLM Fallback

**Problem:** Using an LLM to classify the problem before using the LLM is burning budget on overhead.

**Solution:** Two-tier classifier. Fast heuristic first (0ms, no LLM call). LLM fallback only when heuristic confidence is low.

```python
@dataclass
class ClassificationResult:
    domain: str          # clear, complicated, complex, chaotic, novel
    confidence: float    # 0.0 to 1.0
    method: str          # "heuristic" or "llm"
    reasoning: str


class FastCynefinClassifier:
    """Two-tier Cynefin classifier. Heuristic first, LLM fallback."""

    # ── Tier 1: Pattern-based heuristic (no LLM) ─────────────

    DOMAIN_PATTERNS = {
        "clear": {
            "signals": [
                "how to", "steps to", "best practice", "standard",
                "procedure", "recipe", "tutorial", "guide", "setup",
                "install", "configure",
            ],
            "anti_signals": ["uncertain", "complex", "novel", "ambiguous"],
            "structural": lambda p: len(p.split()) < 30 and "?" in p,
        },
        "complicated": {
            "signals": [
                "optimize", "analyze", "debug", "diagnose", "root cause",
                "architecture", "performance", "scale", "technical",
                "engineering", "expert", "calculate",
            ],
            "anti_signals": ["unprecedented", "chaotic", "novel"],
            "structural": lambda p: any(c.isdigit() for c in p),
        },
        "complex": {
            "signals": [
                "stakeholder", "organizational", "culture", "market",
                "strategy", "long-term", "systemic", "emergent",
                "adaptive", "political", "social", "ecosystem",
                "unintended", "relationship",
            ],
            "anti_signals": ["simple", "straightforward", "standard"],
            "structural": lambda p: len(p.split()) > 50,
        },
        "chaotic": {
            "signals": [
                "crisis", "emergency", "outage", "urgent", "immediately",
                "disaster", "breach", "down", "broken", "collapsing",
                "right now", "asap",
            ],
            "anti_signals": ["long-term", "plan", "strategy"],
            "structural": lambda p: "!" in p or p.isupper(),
        },
        "novel": {
            "signals": [
                "never been done", "no precedent", "invent", "create new",
                "first ever", "paradigm", "revolutionary", "from scratch",
                "doesn't exist", "unprecedented", "reimagine",
            ],
            "anti_signals": ["improve", "optimize", "fix"],
            "structural": lambda p: "?" in p and len(p.split()) > 40,
        },
    }

    # Confidence threshold below which we escalate to LLM
    HEURISTIC_CONFIDENCE_THRESHOLD = 0.55

    def classify(self, problem: str) -> ClassificationResult:
        """Two-tier classification. Fast heuristic first."""

        # Tier 1: Heuristic
        result = self._heuristic_classify(problem)
        if result.confidence >= self.HEURISTIC_CONFIDENCE_THRESHOLD:
            return result

        # Tier 2: LLM fallback (one call, low max_tokens)
        return self._llm_classify(problem, heuristic_hint=result)

    def _heuristic_classify(self, problem: str) -> ClassificationResult:
        """Pattern-matching classifier. O(n), no LLM."""
        problem_lower = problem.lower()
        scores = {}

        for domain, patterns in self.DOMAIN_PATTERNS.items():
            signal_hits = sum(
                1 for s in patterns["signals"] if s in problem_lower
            )
            anti_hits = sum(
                1 for s in patterns["anti_signals"] if s in problem_lower
            )
            structural = 1 if patterns["structural"](problem) else 0

            scores[domain] = signal_hits * 2 - anti_hits * 3 + structural
            scores[domain] = max(scores[domain], 0)

        total = sum(scores.values()) or 1
        normalized = {k: v / total for k, v in scores.items()}

        best_domain = max(normalized, key=normalized.get)
        confidence = normalized[best_domain]

        return ClassificationResult(
            domain=best_domain,
            confidence=confidence,
            method="heuristic",
            reasoning=f"Pattern scores: {scores}. "
                      f"Best: {best_domain} ({confidence:.0%})",
        )

    def _llm_classify(
        self, problem: str, heuristic_hint: ClassificationResult
    ) -> ClassificationResult:
        """LLM fallback. Used only when heuristic is uncertain."""
        # Single call, low token budget, structured output
        prompt = f"""Classify this problem into one Cynefin domain.

Problem: {problem}

Heuristic pre-classification suggested '{heuristic_hint.domain}'
with {heuristic_hint.confidence:.0%} confidence.

Domains: clear, complicated, complex, chaotic, novel.
Respond as JSON: {{"domain": "...", "confidence": 0.X, "reasoning": "..."}}"""

        # ~200 tokens max, fast model
        response = self._call_llm(prompt, max_tokens=200)
        parsed = json.loads(response)

        return ClassificationResult(
            domain=parsed["domain"],
            confidence=parsed["confidence"],
            method="llm",
            reasoning=parsed["reasoning"],
        )
```

### Cost Comparison

| Approach | Latency | Cost | Accuracy |
|----------|---------|------|----------|
| LLM-only (original) | ~2s | ~$0.003 | ~90% |
| Heuristic-only | <1ms | $0 | ~70% |
| **Hybrid (heuristic + LLM fallback)** | **<1ms for ~65% of queries, ~2s for remainder** | **~$0.001 avg** | **~88%** |

The heuristic handles the easy cases (clear questions, obvious crises) and the LLM handles the ambiguous ones. In practice, ~65% of inputs are classifiable by heuristic alone.

---

## Critique 5: System Tiering — What to Build When

**Problem:** 24 systems is overkill for v1. Some don't translate to a single-agent LLM context.

**Solution:** Tier by (a) how well it translates to LLM prompting and (b) how much value it delivers per call.

### Tier 1 — Core (Build First, Week 1)

These are the highest-ROI systems for an LLM agent. Each one is a well-defined prompt pattern that reliably produces structured, useful output.

| System | Why Tier 1 | Prompt Complexity |
|--------|-----------|-------------------|
| **First Principles** | Decomposition + reconstruction. Clean two-stage prompt. Universal applicability. | Medium |
| **Inversion** | Single prompt, massive ROI. "How to guarantee failure?" always produces insights. | Low |
| **Pre-mortem** | Structured failure imagination. Produces actionable risk lists. | Low |
| **Bayesian Updater** | Explicit prior→posterior tracking. Keeps the agent calibrated. | Medium |
| **Critical Thinking** | Standards-based evaluation. The quality gate for all other outputs. | Medium |
| **Divergent/Convergent** | The creative heartbeat. Generate broadly, then filter. Temperature-tunable. | Low |

### Tier 2 — High Value (Build in Week 2-3)

These add strategic depth and handle more nuanced problems.

| System | Why Tier 2 | Notes |
|--------|-----------|-------|
| **Systems Thinking** | Causal loop generation. Requires more structured output (graph). | Needs graph schema |
| **Socratic Method** | Recursive questioning. Natural for multi-turn LLM interaction. | Multi-turn chain |
| **Sensemaking** | Narrative construction from ambiguous data. Valuable for complex domains. | Works best with prior observations |
| **Second-Order** | Recursive consequence tracing. Direct prompt pattern with depth parameter. | Set depth=3 default |
| **OODA Conductor** | Multi-cycle orchestration. More of an orchestration pattern than a prompt. | Framework code, not prompt |
| **Design Thinking** | Multi-phase workflow (empathize → define → ideate → prototype → test). | Orchestration pattern |

### Tier 3 — Specialized (Build on Demand)

Valuable in specific contexts but not universally needed.

| System | When to Add | Notes |
|--------|------------|-------|
| **Lateral Thinking** | When creative problems dominate. Random stimulus injection. | Needs external randomness source |
| **Dialectical** | When inputs frequently conflict. Thesis-antithesis-synthesis. | Use for multi-system disagreement |
| **Theory of Constraints** | For process/operational problems specifically. | Narrow but deep |
| **Probabilistic** | When explicit probability distributions are needed. | Overlaps with Bayesian for most uses |
| **Deductive/Inductive/Abductive** | For formal reasoning tasks. | Often implicit in other systems |

### Skip for Single-Agent v1

| System | Why Skip | How to Reintroduce |
|--------|----------|-------------------|
| **Embodied Cognition** | No body. An LLM has no physical state to leverage. | Multi-modal agent with sensors/actuators |
| **Distributed Cognition** | Single agent. No team to distribute across. | Multi-agent ensemble (Variation B) |
| **Six Thinking Hats** | Designed for group facilitation. One agent can't genuinely take 6 adversarial perspectives at the same time — it'll converge. | Multi-agent with 6 specialized agents |
| **Bounded Rationality** | Not a prompt — it's a governor. Already embedded in the Kernel. | Keep as Kernel logic, not a callable tool |
| **Dual-Process Theory** | Not a prompt — it's a routing decision. Already embedded in the Entry Gate. | Keep as Router logic |
| **Embodied Cognition** | *(duplicate removed)* | |

### Build Order

```
Week 1: Tier 1 (6 systems) + Cynefin classifier + Kernel
Week 2: Tier 2 systems + OODA conductor + tiered memory
Week 3: Tier 3 on demand + multi-system synthesis + calibration
Week 4+: Multi-agent for skip-tier systems
```

---

## Critique 6: State Management — Replacing the Flat AgentState

**Problem:** The original `AgentState` dataclass is flat and will grow unbounded. Every field accumulates forever.

**Solution:** Replace with the tiered memory architecture from Critique 2, wrapped in a proper state machine.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import deque


class AgentPhase(Enum):
    ROUTING = "routing"        # Dual-Process gate
    CLASSIFYING = "classifying" # Cynefin
    EXECUTING = "executing"     # Running thinking systems
    SYNTHESIZING = "synthesizing"  # Combining outputs
    REFLECTING = "reflecting"   # Meta-cognitive review
    COMPLETE = "complete"


@dataclass
class WorkingMemory:
    """Current phase context. Circular buffer, auto-evicts oldest."""
    items: deque = field(default_factory=lambda: deque(maxlen=10))
    current_system: Optional[str] = None
    current_phase: Optional[str] = None

    def add(self, source: str, content: str, confidence: float = 0.5):
        self.items.append({
            "source": source,
            "content": content,
            "confidence": confidence,
        })

    def to_context(self) -> str:
        if not self.items:
            return "No working context yet."
        return "\n".join(
            f"[{i['source']}] (conf: {i['confidence']:.0%}) {i['content']}"
            for i in self.items
        )

    def clear(self):
        self.items.clear()


@dataclass
class SemanticMemory:
    """Compressed facts and decisions. Pruned by relevance."""
    facts: list[dict] = field(default_factory=list)  # {text, source, confidence, phase}
    decisions: list[dict] = field(default_factory=list)  # {decision, rationale, confidence}
    max_facts: int = 30  # Hard cap — prune lowest-confidence when exceeded

    def add_fact(self, text: str, source: str, confidence: float, phase: str):
        self.facts.append({
            "text": text, "source": source,
            "confidence": confidence, "phase": phase,
        })
        self._prune()

    def add_decision(self, decision: str, rationale: str, confidence: float):
        self.decisions.append({
            "decision": decision, "rationale": rationale,
            "confidence": confidence,
        })

    def _prune(self):
        """Keep only the top-N facts by confidence."""
        if len(self.facts) > self.max_facts:
            self.facts.sort(key=lambda f: f["confidence"], reverse=True)
            self.facts = self.facts[:self.max_facts]

    def to_context(self) -> str:
        parts = []
        if self.facts:
            facts_str = "\n".join(
                f"- {f['text']} [{f['source']}, {f['confidence']:.0%}]"
                for f in self.facts[-15:]  # Most recent 15
            )
            parts.append(f"Known facts:\n{facts_str}")
        if self.decisions:
            dec_str = "\n".join(
                f"- {d['decision']} ({d['confidence']:.0%})"
                for d in self.decisions
            )
            parts.append(f"Decisions made:\n{dec_str}")
        return "\n\n".join(parts) if parts else "No prior knowledge."


@dataclass
class EpisodicLog:
    """Compressed process history. Append-only, accessed in Reflect."""
    entries: list[dict] = field(default_factory=list)
    max_entries: int = 20

    def log(self, phase: str, system: str, summary: str):
        self.entries.append({
            "phase": phase, "system": system, "summary": summary,
        })
        # Keep only recent entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def to_context(self) -> str:
        if not self.entries:
            return "No process history."
        return "\n".join(
            f"[{e['phase']}/{e['system']}] {e['summary']}"
            for e in self.entries
        )


@dataclass
class AgentState:
    """Replaces the flat state with tiered memory."""

    # Problem definition (immutable after init)
    problem: str = ""
    domain: Optional[str] = None
    stakes: str = "medium"
    route: Optional[str] = None  # system_1, s2_light, s2_deep

    # Phase tracking (state machine)
    phase: AgentPhase = AgentPhase.ROUTING
    master_sequence_phase: Optional[str] = None  # orient, perceive, frame, etc.
    cycle: int = 0

    # Tiered memory
    working: WorkingMemory = field(default_factory=WorkingMemory)
    semantic: SemanticMemory = field(default_factory=SemanticMemory)
    episodic: EpisodicLog = field(default_factory=EpisodicLog)

    # Kernel monitors (small, fixed-size)
    confidence: float = 0.5
    effort_spent: float = 0.0
    effort_budget: float = 0.5
    active_alerts: list[str] = field(default_factory=list)

    def build_context(self) -> str:
        """Build the context string for the current LLM call.
        Always bounded by memory tier budgets."""
        parts = [
            f"## Problem\n{self.problem}",
            f"## Domain: {self.domain or 'unclassified'} | "
            f"Phase: {self.master_sequence_phase or 'routing'} | "
            f"Confidence: {self.confidence:.0%} | "
            f"Budget: {(self.effort_budget - self.effort_spent) / self.effort_budget:.0%} remaining",
        ]

        if self.active_alerts:
            parts.append(f"## ⚠ Alerts\n" + "\n".join(f"- {a}" for a in self.active_alerts))

        parts.append(f"## Prior Knowledge\n{self.semantic.to_context()}")
        parts.append(f"## Current Phase Context\n{self.working.to_context()}")

        # Episodic only in reflect phase
        if self.phase == AgentPhase.REFLECTING:
            parts.append(f"## Process History\n{self.episodic.to_context()}")

        return "\n\n".join(parts)

    def transition_phase(self, new_phase: str):
        """Handle phase transition: consolidate working → semantic."""
        # Log to episodic
        self.episodic.log(
            phase=self.master_sequence_phase or "?",
            system=self.working.current_system or "?",
            summary=f"Completed with {len(self.working.items)} items, "
                    f"confidence {self.confidence:.0%}",
        )
        # Working memory items become semantic memory candidates
        # (In practice, run the consolidation LLM call here)
        self.working.clear()
        self.master_sequence_phase = new_phase
```

### Memory Lifecycle

```
Problem arrives
  → Router classifies (no memory needed)
  → Cynefin classifies → result goes to semantic.add_fact()

Phase: Frame
  → First Principles runs → insights go to working.add()
  → Socratic runs → insights go to working.add()
  → Phase transition → consolidate working → semantic
  → working.clear()

Phase: Generate
  → Divergent runs → ideas go to working.add()
  → Lateral runs → ideas go to working.add()
  → Phase transition → consolidate → semantic

Phase: Evaluate
  → Context window contains: semantic (compressed facts from Frame + Generate)
    + working (current eval outputs). Total: ~3-4K tokens, not 15K.
  → Pre-mortem, Bayesian, etc. run against compact context

Phase: Reflect
  → Episodic log is included (only time)
  → Full process review against compact history
```

---

## Summary: The Revised Architecture Stack

```
┌─────────────────────────────────────────────────────────────┐
│  ENTRY GATE: Dual-Process Router (Critique 1)               │
│  ┌─────────┐  ┌─────────────┐  ┌───────────────────┐       │
│  │System 1 │  │System 2     │  │System 2 Deep      │       │
│  │1 call   │  │Light: 3-5   │  │Full pipeline: 10-20│      │
│  └────┬────┘  └──────┬──────┘  └─────────┬─────────┘       │
│       │              │                    │                  │
│       ▼              ▼                    ▼                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CYNEFIN GATE: Fast Heuristic + LLM Fallback (C4)   │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │  KERNEL: System Prompt (always-on monitoring)         │   │
│  │  Metacognition · Bias Detection · Budget Governor     │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │  TIERED MEMORY (Critique 2 & 6)                       │   │
│  │  Working (current) → Semantic (facts) → Episodic (log)│   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │  THINKING SYSTEMS as Tools (Critique 5 tiering)       │   │
│  │  Tier 1: FP, Inversion, Pre-mortem, Bayes, Crit, D/C │   │
│  │  Tier 2: Systems, Socratic, Sense, 2nd-Order, OODA   │   │
│  │  Tier 3: Lateral, Dialectical, ToC, Probabilistic     │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │  STRUCTURED OUTPUT: JSON Envelope (Critique 3)        │   │
│  │  Common: confidence, insights, metacog, suggested_next│   │
│  │  + system_specific: per-system schema                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Revised Call Budget

| Scenario | Old Architecture | Revised Architecture |
|----------|-----------------|---------------------|
| Simple question | 10-15 calls (full pipeline) | **1 call** (System 1 gate) |
| Moderate analysis | 10-15 calls | **4-6 calls** (S2 Light + targeted) |
| High-stakes strategy | 15-20 calls | **12-18 calls** (S2 Deep, same but bounded) |
| Cynefin classification | 1 LLM call always | **0 calls ~65% of the time** |
| Context window per call | Unbounded growth | **~5K tokens, bounded** |
