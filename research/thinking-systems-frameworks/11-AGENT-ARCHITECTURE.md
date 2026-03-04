# 11 — Agent Architecture: Programming Thinking Systems

How to implement the 24 thinking systems, their integration patterns, and the three-layer operating model as a programmable AI agent.

---

## Core Insight

The three-layer model (Kernel / OS / Application) maps directly to agent architecture:

- **Kernel** → Always-on monitoring hooks (metacognition, bias detection, effort calibration)
- **OS** → Situation classifier + system selector (Cynefin dispatch, OODA rhythm, latticework routing)
- **Application** → Individual thinking system modules invoked per phase

The agent doesn't "pick one framework and run it." It runs a continuous control loop where the Kernel monitors, the OS classifies and routes, and Application-layer modules execute specific reasoning patterns.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     THINKING AGENT                          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  KERNEL (always running)                              │  │
│  │  ┌──────────┐ ┌──────────────┐ ┌──────────────────┐  │  │
│  │  │Metacog   │ │Dual-Process  │ │Bounded Rationality│  │  │
│  │  │Monitor   │ │Router        │ │Governor           │  │  │
│  │  └──────────┘ └──────────────┘ └──────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │ signals                          │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │  OS LAYER (selected per situation)                    │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌────────────┐  │  │
│  │  │Cynefin   │ │OODA      │ │Systems│ │Latticework │  │  │
│  │  │Classifier│ │Conductor │ │Mapper │ │Selector    │  │  │
│  │  └──────────┘ └──────────┘ └──────┘ └────────────┘  │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │ dispatch                         │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │  APPLICATION LAYER (selected per phase)               │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │  │
│  │  │First         │ │Socratic      │ │Pre-mortem    │  │  │
│  │  │Principles    │ │Questioner    │ │Simulator     │  │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │  │
│  │  │Lateral       │ │Bayesian      │ │Inversion     │  │  │
│  │  │Generator     │ │Updater       │ │Analyzer      │  │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │  │
│  │  ... (all 24 systems as callable modules)             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  STATE: problem context, phase, confidence,           │  │
│  │         history, accumulated insights                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. The Thinking System Module

Every thinking system is implemented as a module with a standard interface.

### Interface

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Phase(Enum):
    ORIENT = "orient"
    PERCEIVE = "perceive"
    FRAME = "frame"
    GENERATE = "generate"
    EVALUATE = "evaluate"
    DECIDE = "decide"
    REFLECT = "reflect"

class Domain(Enum):
    CLEAR = "clear"
    COMPLICATED = "complicated"
    COMPLEX = "complex"
    CHAOTIC = "chaotic"
    NOVEL = "novel"

@dataclass
class ThinkingSystemOutput:
    insights: list[str]
    confidence: float  # 0.0 to 1.0
    artifacts: dict  # structured outputs (diagrams, models, lists)
    next_suggested_phase: Optional[Phase] = None
    next_suggested_systems: list[str] = field(default_factory=list)
    should_loop_back: bool = False
    loop_target_phase: Optional[Phase] = None

class ThinkingSystem:
    """Base interface for all thinking system modules."""

    name: str
    phases: list[Phase]          # which phases this system serves
    domains: list[Domain]        # which Cynefin domains it fits
    locus: str                   # individual, dialogical, team, distributed
    cognitive_load: str          # low, moderate, high, very_high
    complements: list[str]       # names of synergistic systems

    def execute(
        self,
        problem: str,
        context: dict,       # accumulated state from prior systems
        phase: Phase,
        depth: int = 1       # how deep to go (bounded rationality)
    ) -> ThinkingSystemOutput:
        """Run this thinking system on the problem."""
        raise NotImplementedError

    def get_prompt(self, problem: str, context: dict, phase: Phase) -> str:
        """Generate the LLM prompt for this system."""
        raise NotImplementedError
```

### Example: First Principles Module

```python
class FirstPrinciplesSystem(ThinkingSystem):
    name = "first_principles"
    phases = [Phase.FRAME, Phase.GENERATE]
    domains = [Domain.NOVEL, Domain.COMPLEX]
    locus = "individual"
    cognitive_load = "very_high"
    complements = ["inversion", "socratic_method", "lateral_thinking"]

    def get_prompt(self, problem: str, context: dict, phase: Phase) -> str:
        if phase == Phase.FRAME:
            return f"""You are performing First Principles analysis.

PROBLEM: {problem}

PRIOR CONTEXT: {context.get('accumulated_insights', 'None yet')}

TASK — Decomposition:
1. List every assumption embedded in how this problem is currently
   understood or how solutions are currently approached.
2. For each assumption, ask: "Is this a fundamental truth (physics,
   math, verified empirical fact) or a convention/analogy/habit?"
3. Discard everything that is convention. Keep only bedrock truths.
4. State the problem ONLY in terms of those bedrock truths.

OUTPUT FORMAT:
- assumptions_found: list of assumptions with classification
- bedrock_truths: the fundamental truths that survive
- reframed_problem: the problem stated purely in terms of bedrock truths
- conventional_baggage: what was discarded and why
"""
        elif phase == Phase.GENERATE:
            return f"""You are performing First Principles reconstruction.

PROBLEM (reframed): {context.get('reframed_problem', problem)}
BEDROCK TRUTHS: {context.get('bedrock_truths', 'Not yet identified')}

TASK — Reconstruction:
Starting ONLY from the bedrock truths above, build up novel solutions.
Do NOT reference existing solutions or conventional approaches.
Reason upward from the truths to generate solutions that may look
completely different from anything that currently exists.

Generate at least 5 distinct solution directions.

OUTPUT FORMAT:
- solutions: list of novel solution directions with reasoning chain
  from bedrock truths to solution
- most_promising: which solution has the strongest logical foundation
- comparison_to_conventional: how these differ from standard approaches
"""
```

### Example: Pre-mortem Module

```python
class PreMortemSystem(ThinkingSystem):
    name = "pre_mortem"
    phases = [Phase.EVALUATE]
    domains = [Domain.COMPLICATED, Domain.COMPLEX]
    locus = "team"
    cognitive_load = "moderate"
    complements = ["inversion", "second_order_thinking", "probabilistic"]

    def get_prompt(self, problem: str, context: dict, phase: Phase) -> str:
        proposed_solution = context.get("proposed_solution", problem)
        return f"""You are conducting a Pre-mortem analysis.

PROPOSED SOLUTION/PLAN: {proposed_solution}

TASK — Prospective Hindsight:
It is 12 months from now. This plan has FAILED SPECTACULARLY.

1. Generate at least 10 distinct, specific reasons why it failed.
   Be vivid and concrete — not vague risks but specific failure stories.
2. For each failure reason, assess:
   - Probability (0.0 to 1.0)
   - Severity (1-10)
   - Detectability: would we see it coming? (early/late/never)
   - Preventability: can we mitigate it now? (easy/hard/impossible)
3. Rank failures by risk score (probability × severity).
4. For the top 5 risks, propose specific mitigations.

Also run the CRYSTAL BALL variant:
It is 12 months from now. This plan SUCCEEDED BRILLIANTLY.
What happened? What did we get right? What lucky breaks occurred?

OUTPUT FORMAT:
- failure_modes: ranked list with probability, severity, mitigations
- success_factors: what would need to go right
- critical_mitigations: actions to take NOW
- revised_confidence: updated confidence in the plan (0.0 to 1.0)
"""
```

---

## 2. The Kernel Layer

The Kernel runs continuously — it wraps every thinking system call with monitoring.

```python
@dataclass
class KernelState:
    current_phase: Phase
    cognitive_budget: float      # 0.0 to 1.0 — how much effort to invest
    confidence: float            # current calibrated confidence
    bias_alerts: list[str]       # detected potential biases
    system_1_flags: list[str]    # moments where System 1 might be driving
    effort_spent: float          # track total computation used
    max_effort: float            # bounded rationality cap
    reasoning_quality: float     # self-assessed reasoning quality

class Kernel:
    """Always-on monitoring layer — Metacognition + Dual-Process +
    Bounded Rationality + Embodied Cognition."""

    def __init__(self, stakes_level: str = "medium"):
        # Bounded Rationality: set effort budget based on stakes
        effort_map = {
            "low": 0.2,     # satisfice quickly
            "medium": 0.5,  # moderate analysis
            "high": 0.8,    # deep analysis
            "critical": 1.0  # full stack, no shortcuts
        }
        self.state = KernelState(
            current_phase=Phase.ORIENT,
            cognitive_budget=effort_map.get(stakes_level, 0.5),
            confidence=0.5,  # start at maximum uncertainty
            bias_alerts=[],
            system_1_flags=[],
            effort_spent=0.0,
            max_effort=effort_map.get(stakes_level, 0.5),
            reasoning_quality=0.5
        )

    def pre_check(self, system: ThinkingSystem, context: dict) -> dict:
        """Run BEFORE every thinking system call."""
        alerts = {}

        # Dual-Process check: is this a System 1 autopilot risk?
        if system.cognitive_load == "low" and self.state.confidence > 0.8:
            alerts["system1_warning"] = (
                "High confidence + low cognitive load = possible "
                "autopilot. Consider whether System 2 should engage."
            )

        # Bounded Rationality check: are we over-investing?
        if self.state.effort_spent > self.state.max_effort * 0.8:
            alerts["budget_warning"] = (
                f"Cognitive budget {self.state.effort_spent:.0%} spent. "
                "Consider satisficing with current best answer."
            )

        # Bias detection: check for common patterns
        if context.get("iterations", 0) > 3 and self.state.confidence < 0.4:
            alerts["analysis_paralysis"] = (
                "Multiple iterations with low confidence. "
                "Consider: act on best current hypothesis and observe."
            )

        if (context.get("solutions_generated", 0) < 3
                and self.state.current_phase == Phase.EVALUATE):
            alerts["premature_convergence"] = (
                "Evaluating with fewer than 3 options. "
                "Consider returning to GENERATE phase."
            )

        return alerts

    def post_check(self, output: ThinkingSystemOutput) -> dict:
        """Run AFTER every thinking system call."""
        # Update confidence
        self.state.confidence = (
            0.7 * self.state.confidence + 0.3 * output.confidence
        )
        self.state.effort_spent += 0.1  # increment effort tracker

        signals = {}

        # Should we loop back?
        if output.should_loop_back:
            signals["loop_back"] = output.loop_target_phase

        # Should we switch systems?
        if output.next_suggested_systems:
            signals["suggested_next"] = output.next_suggested_systems

        return signals

    def should_stop(self) -> bool:
        """Bounded Rationality governor: have we spent enough effort?"""
        if self.state.effort_spent >= self.state.max_effort:
            return True
        if self.state.confidence > 0.9:
            return True  # high confidence — stop refining
        return False

    def metacognitive_prompt(self) -> str:
        """Generate a metacognitive reflection prompt to inject
        into any system call."""
        return f"""
[METACOGNITIVE MONITOR — Phase: {self.state.current_phase.value}]
Current confidence: {self.state.confidence:.0%}
Effort budget remaining: {(self.state.max_effort - self.state.effort_spent) / self.state.max_effort:.0%}
Active alerts: {self.state.bias_alerts or 'None'}

Before proceeding, briefly assess:
- Is the current approach working?
- Am I reasoning carefully or on autopilot?
- What am I most uncertain about?
"""
```

---

## 3. The OS Layer — Situation Classifier & Dispatcher

### Cynefin Classifier

```python
class CynefinClassifier:
    """Classifies problems into Cynefin domains and dispatches
    to appropriate thinking system combinations."""

    DOMAIN_PLAYBOOKS = {
        Domain.CLEAR: {
            "description": "Known solution exists. Apply best practice.",
            "primary_systems": ["critical_thinking", "deductive"],
            "sequence": [Phase.PERCEIVE, Phase.EVALUATE, Phase.DECIDE],
            "max_iterations": 1
        },
        Domain.COMPLICATED: {
            "description": "Experts can solve it. Analyze deeply.",
            "primary_systems": [
                "systems_thinking", "theory_of_constraints",
                "bayesian", "second_order", "critical_thinking"
            ],
            "sequence": [
                Phase.PERCEIVE, Phase.FRAME, Phase.EVALUATE,
                Phase.DECIDE
            ],
            "max_iterations": 3
        },
        Domain.COMPLEX: {
            "description": "Emergent. Probe, sense, respond.",
            "primary_systems": [
                "design_thinking", "sensemaking", "divergent",
                "dialectical", "bayesian", "pre_mortem"
            ],
            "sequence": [
                Phase.PERCEIVE, Phase.FRAME, Phase.GENERATE,
                Phase.EVALUATE, Phase.DECIDE, Phase.REFLECT
            ],
            "max_iterations": 5
        },
        Domain.CHAOTIC: {
            "description": "Act first. Establish order. Then sense.",
            "primary_systems": [
                "ooda", "bounded_rationality", "sensemaking"
            ],
            "sequence": [Phase.DECIDE, Phase.PERCEIVE, Phase.REFLECT],
            "max_iterations": 2
        },
        Domain.NOVEL: {
            "description": "No precedent. Build from scratch.",
            "primary_systems": [
                "first_principles", "lateral_thinking", "divergent",
                "convergent", "inversion", "pre_mortem"
            ],
            "sequence": [
                Phase.FRAME, Phase.GENERATE, Phase.EVALUATE,
                Phase.DECIDE, Phase.REFLECT
            ],
            "max_iterations": 4
        },
    }

    def classify(self, problem: str, context: dict) -> Domain:
        """Use an LLM call to classify the problem domain."""
        prompt = f"""Classify this problem into exactly one Cynefin domain.

PROBLEM: {problem}

DOMAINS:
- CLEAR: The relationship between cause and effect is obvious.
  Best practices exist and are well-known. Example: following a recipe.
- COMPLICATED: Cause and effect require analysis or expertise.
  Multiple right answers exist. Example: engineering a bridge.
- COMPLEX: Cause and effect are only visible in retrospect.
  The situation is emergent and unpredictable. Example: raising a child,
  entering a new market.
- CHAOTIC: No discernible cause and effect. Immediate action needed.
  Example: crisis response, system outage.
- NOVEL: No precedent exists. Conventional wisdom doesn't apply.
  Example: inventing a new technology category.

Respond with:
- domain: one of [clear, complicated, complex, chaotic, novel]
- confidence: 0.0 to 1.0
- reasoning: brief explanation of why this domain
- ambiguity: any parts of the problem that might belong to a
  different domain
"""
        # This would call the LLM and parse the response
        return self._call_llm_and_parse(prompt)

    def get_playbook(self, domain: Domain) -> dict:
        """Return the appropriate playbook for the classified domain."""
        return self.DOMAIN_PLAYBOOKS[domain]
```

### OODA Conductor

```python
class OODAConductor:
    """Manages the observe-orient-decide-act rhythm across iterations."""

    def __init__(self):
        self.cycle_count = 0
        self.observations = []
        self.orientations = []
        self.decisions = []
        self.actions = []

    def observe(self, new_data: dict) -> dict:
        """Gather and integrate new information."""
        self.observations.append(new_data)
        return {
            "all_observations": self.observations,
            "new_signals": new_data,
            "observation_count": len(self.observations)
        }

    def orient(self, observations: dict, mental_models: list[str]) -> str:
        """Generate an orientation prompt that synthesizes observations
        through available mental models."""
        prompt = f"""You are in the ORIENT phase of an OODA cycle
(cycle #{self.cycle_count + 1}).

OBSERVATIONS (all accumulated):
{observations}

MENTAL MODELS AVAILABLE: {mental_models}

PRIOR ORIENTATIONS: {self.orientations[-3:] if self.orientations else 'None'}

TASK:
1. What patterns do you see across all observations?
2. How do the available mental models reframe what you're seeing?
3. What has changed since the last orientation?
4. What are you most uncertain about?
5. What would change your current understanding?

Construct a coherent orientation — a narrative that makes sense of
what you're observing and suggests what to do next.
"""
        return prompt

    def decide(self, orientation: str, options: list[dict]) -> str:
        """Select a course of action based on orientation."""
        prompt = f"""Based on this orientation:
{orientation}

And these available options:
{options}

Select the best course of action. Treat this as a hypothesis,
not a commitment. Be prepared to update on next cycle.
"""
        return prompt

    def complete_cycle(self, action_result: dict):
        """Feed action results back as observations for next cycle."""
        self.cycle_count += 1
        self.observe({"action_result": action_result, "cycle": self.cycle_count})
```

### Latticework Selector

```python
class LatticeworkSelector:
    """Selects which domain models (mental models from different
    disciplines) are most relevant to the current problem."""

    DOMAIN_MODELS = {
        "biology": [
            "natural_selection", "niche_theory", "red_queen",
            "symbiosis", "immune_response", "homeostasis"
        ],
        "physics": [
            "entropy", "conservation_laws", "equilibrium",
            "phase_transitions", "feedback_loops"
        ],
        "economics": [
            "supply_demand", "incentive_structures",
            "comparative_advantage", "externalities",
            "marginal_utility", "game_theory"
        ],
        "psychology": [
            "cognitive_biases", "social_proof", "loss_aversion",
            "motivation_theory", "habit_formation"
        ],
        "engineering": [
            "redundancy", "margin_of_safety", "feedback_control",
            "modularity", "graceful_degradation"
        ],
        "mathematics": [
            "combinatorics", "probability", "optimization",
            "network_theory", "exponential_growth"
        ],
        "history": [
            "analogical_reasoning", "cyclical_patterns",
            "unintended_consequences", "chesterton_fence"
        ],
    }

    def select_models(self, problem: str, domain: Domain) -> list[str]:
        """Use LLM to select the most relevant cross-domain models."""
        prompt = f"""Given this problem and its domain classification:

PROBLEM: {problem}
DOMAIN: {domain.value}

AVAILABLE MENTAL MODELS BY DISCIPLINE:
{self.DOMAIN_MODELS}

Select the 3-5 most relevant models from DIFFERENT disciplines.
For each, explain in one sentence why it's relevant.

Prioritize models that offer non-obvious perspectives — the value
is in cross-pollination, not confirmation.
"""
        return self._call_llm_and_parse(prompt)
```

---

## 4. The Orchestrator — Putting It All Together

```python
class ThinkingAgent:
    """The main orchestrator that ties Kernel, OS, and Application
    layers together into a functioning thinking agent."""

    def __init__(self, stakes: str = "medium"):
        # Kernel layer
        self.kernel = Kernel(stakes_level=stakes)

        # OS layer
        self.cynefin = CynefinClassifier()
        self.ooda = OODAConductor()
        self.latticework = LatticeworkSelector()

        # Application layer — registry of all thinking systems
        self.systems: dict[str, ThinkingSystem] = {}
        self._register_default_systems()

        # State
        self.context = {
            "problem": "",
            "domain": None,
            "phase": Phase.ORIENT,
            "accumulated_insights": [],
            "iterations": 0,
            "solutions_generated": 0,
        }

    def _register_default_systems(self):
        """Register all 24 thinking system modules."""
        self.systems["first_principles"] = FirstPrinciplesSystem()
        self.systems["pre_mortem"] = PreMortemSystem()
        self.systems["inversion"] = InversionSystem()
        self.systems["socratic_method"] = SocraticSystem()
        self.systems["bayesian"] = BayesianSystem()
        self.systems["lateral_thinking"] = LateralThinkingSystem()
        self.systems["systems_thinking"] = SystemsThinkingSystem()
        self.systems["design_thinking"] = DesignThinkingSystem()
        self.systems["critical_thinking"] = CriticalThinkingSystem()
        self.systems["divergent"] = DivergentThinkingSystem()
        self.systems["convergent"] = ConvergentThinkingSystem()
        self.systems["second_order"] = SecondOrderSystem()
        self.systems["dialectical"] = DialecticalSystem()
        self.systems["sensemaking"] = SensemakingSystem()
        self.systems["theory_of_constraints"] = TOCSystem()
        # ... etc for all 24

    def solve(self, problem: str) -> dict:
        """Main entry point. Run the full thinking pipeline."""
        self.context["problem"] = problem

        # ── PHASE 0: ORIENT ──────────────────────────────────
        # Cynefin classification
        domain = self.cynefin.classify(problem, self.context)
        self.context["domain"] = domain
        playbook = self.cynefin.get_playbook(domain)

        # Latticework selection
        relevant_models = self.latticework.select_models(problem, domain)
        self.context["mental_models"] = relevant_models

        # Kernel: set up monitoring based on domain
        self.kernel.state.current_phase = Phase.ORIENT

        # ── EXECUTE PLAYBOOK PHASES ──────────────────────────
        for phase in playbook["sequence"]:
            if self.kernel.should_stop():
                break

            self.kernel.state.current_phase = phase
            self.context["phase"] = phase

            # Get systems appropriate for this phase + domain
            phase_systems = self._select_systems_for_phase(
                phase, playbook["primary_systems"]
            )

            for system_name in phase_systems:
                if self.kernel.should_stop():
                    break

                system = self.systems[system_name]

                # Kernel pre-check
                alerts = self.kernel.pre_check(system, self.context)
                if alerts:
                    self.context["kernel_alerts"] = alerts

                # Execute the thinking system
                output = self._execute_system(system, phase)

                # Kernel post-check
                signals = self.kernel.post_check(output)

                # Accumulate insights
                self.context["accumulated_insights"].extend(output.insights)
                self.context["iterations"] += 1

                # Handle loop-back signals
                if "loop_back" in signals:
                    # Reset to earlier phase
                    break  # will be caught by outer loop logic

                # OODA: feed results back as observations
                self.ooda.observe({
                    "system": system_name,
                    "phase": phase.value,
                    "output": output,
                })

        # ── PHASE 6: REFLECT ─────────────────────────────────
        reflection = self._reflect()

        return {
            "domain": domain.value,
            "insights": self.context["accumulated_insights"],
            "confidence": self.kernel.state.confidence,
            "iterations": self.context["iterations"],
            "reflection": reflection,
            "effort_spent": self.kernel.state.effort_spent,
        }

    def _select_systems_for_phase(
        self, phase: Phase, primary_systems: list[str]
    ) -> list[str]:
        """Select which systems to run for this phase, filtered by
        the playbook's primary systems and the phase compatibility."""
        result = []
        for name in primary_systems:
            if name in self.systems:
                system = self.systems[name]
                if phase in system.phases:
                    result.append(name)
        return result

    def _execute_system(
        self, system: ThinkingSystem, phase: Phase
    ) -> ThinkingSystemOutput:
        """Execute a single thinking system with metacognitive wrapping."""
        # Build the full prompt: system prompt + metacognitive monitor
        system_prompt = system.get_prompt(
            self.context["problem"], self.context, phase
        )
        meta_prompt = self.kernel.metacognitive_prompt()
        full_prompt = meta_prompt + "\n\n" + system_prompt

        # Call LLM
        raw_output = self._call_llm(full_prompt)

        # Parse into structured output
        return self._parse_output(raw_output)

    def _reflect(self) -> str:
        """Final metacognitive reflection on the entire process."""
        prompt = f"""Reflect on the thinking process just completed.

Problem: {self.context['problem']}
Domain: {self.context['domain'].value}
Iterations: {self.context['iterations']}
Final confidence: {self.kernel.state.confidence:.0%}

Key insights accumulated:
{self.context['accumulated_insights'][-10:]}

1. What was the most valuable thinking system used and why?
2. Were there any moments where the approach should have shifted?
3. What remains uncertain?
4. If you could do this again, what would you change?
5. What should be explored next?
"""
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """Call the underlying LLM. Implementation depends on provider."""
        # Anthropic API, OpenAI, local model, etc.
        raise NotImplementedError

    def _parse_output(self, raw: str) -> ThinkingSystemOutput:
        """Parse LLM output into structured ThinkingSystemOutput."""
        raise NotImplementedError
```

---

## 5. Agent Variations

The base architecture supports several specialization strategies.

### Variation A: Single-Agent, Dynamic System Selection

One agent, one LLM, dynamically selects and chains thinking systems based on the Cynefin classification. This is the architecture described above.

**Strengths:** Simple to deploy, maintains coherent context across systems.
**Weaknesses:** Single model's biases affect all systems, sequential bottleneck.

### Variation B: Multi-Agent Ensemble

Each thinking system is a separate agent with its own system prompt, temperature, and personality. A "conductor" agent orchestrates them.

```python
class MultiAgentThinkingSystem:
    """Each thinking system is a separate agent."""

    def __init__(self):
        self.agents = {
            "first_principles": Agent(
                system_prompt=FIRST_PRINCIPLES_PERSONA,
                temperature=0.3,  # precise, analytical
                model="claude-sonnet-4-5-20250929"
            ),
            "lateral_thinking": Agent(
                system_prompt=LATERAL_THINKING_PERSONA,
                temperature=0.9,  # high creativity
                model="claude-sonnet-4-5-20250929"
            ),
            "critical_thinking": Agent(
                system_prompt=CRITICAL_THINKING_PERSONA,
                temperature=0.1,  # rigorous, conservative
                model="claude-sonnet-4-5-20250929"
            ),
            "pre_mortem": Agent(
                system_prompt=PRE_MORTEM_PERSONA,
                temperature=0.5,
                model="claude-sonnet-4-5-20250929"
            ),
            # ... etc
        }
        self.conductor = ConductorAgent(
            system_prompt=CONDUCTOR_PERSONA,  # Cynefin + OODA + Metacognition
            temperature=0.3
        )
```

**Strengths:** Each agent can have tuned parameters (temperature, model). Parallel execution. Natural diversity of perspectives.
**Weaknesses:** Context synchronization overhead. More expensive. Integration complexity.

### Variation C: Layered Pipeline (SEDA-style)

Thinking systems are pipeline stages. The problem flows through stages, each enriching the context. Stages can run in parallel where independent.

```python
pipeline = Pipeline([
    # Phase 0: Orient (parallel)
    ParallelStage([
        CynefinClassifier(),
        SensemakingModule(),
        MetacognitionCheck(),
    ]),
    # Phase 2: Frame (sequential within, selected by Cynefin output)
    ConditionalStage({
        "novel": [FirstPrinciplesSystem(), InversionSystem()],
        "complex": [DesignThinkingSystem(), SocraticSystem()],
        "complicated": [SystemsThinkingSystem(), TOCSystem()],
    }),
    # Phase 3: Generate (parallel)
    ParallelStage([
        DivergentThinkingSystem(),
        LateralThinkingSystem(),
        LatticeworkGenerator(),
    ]),
    # Phase 4: Evaluate (sequential)
    SequentialStage([
        ConvergentThinkingSystem(),
        BayesianUpdater(),
        PreMortemSystem(),
        SecondOrderSystem(),
        InversionSystem(),
    ]),
    # Phase 5: Decide
    BoundedRationalityDecider(),
    # Phase 6: Reflect
    MetacognitiveReflection(),
])

result = pipeline.run(problem, max_cycles=3)
```

**Strengths:** Clear data flow. Easy to parallelize. Natural for batch processing.
**Weaknesses:** Less dynamic — harder to loop back mid-pipeline. Fixed structure.

### Variation D: Tool-Use Agent

The thinking systems are registered as tools that the agent can call. The agent itself decides which tools to use based on its built-in understanding of the framework.

```python
tools = [
    {
        "name": "cynefin_classify",
        "description": "Classify a problem into a Cynefin domain "
                       "(clear/complicated/complex/chaotic/novel). "
                       "ALWAYS call this first.",
        "parameters": {"problem": "string"}
    },
    {
        "name": "first_principles_decompose",
        "description": "Decompose a problem to its fundamental truths, "
                       "stripping away assumptions and conventions. "
                       "Use for novel problems or when conventional "
                       "approaches have failed.",
        "parameters": {"problem": "string", "depth": "integer"}
    },
    {
        "name": "pre_mortem",
        "description": "Imagine a plan has failed and identify why. "
                       "Use before committing to any significant decision.",
        "parameters": {"plan": "string"}
    },
    {
        "name": "bayesian_update",
        "description": "Update confidence in a hypothesis given new "
                       "evidence. Use when new data arrives.",
        "parameters": {
            "hypothesis": "string",
            "prior_confidence": "float",
            "new_evidence": "string"
        }
    },
    {
        "name": "inversion_analysis",
        "description": "Ask 'how would I guarantee failure?' to identify "
                       "risks and reframe problems. Use at start and end "
                       "of any analysis.",
        "parameters": {"goal": "string"}
    },
    # ... one tool per thinking system
]
```

The agent's system prompt encodes the Master Sequence, the three layers, and the integration rules as guidelines for tool selection.

**Strengths:** Most flexible. Agent can adapt dynamically. Leverages native tool-use capabilities.
**Weaknesses:** Relies on the agent's judgment for tool selection. May under-use systems it doesn't "understand."

### Variation E: Hybrid — Kernel as System Prompt + OS as Router + Apps as Tools

The most practical architecture combines all approaches:

- **Kernel** is baked into the system prompt (always-on metacognitive monitoring)
- **OS** is a router/classifier that runs first and selects the playbook
- **Applications** are tools the agent calls within the playbook

```python
SYSTEM_PROMPT = """You are a thinking agent that uses structured
reasoning frameworks to solve problems.

## KERNEL (always active)
You are always monitoring your own reasoning:
- Am I on autopilot (System 1) or deliberately engaged (System 2)?
- Is my current approach working?
- Have I spent proportional effort to the stakes?
- Am I falling into any cognitive biases?

Before EVERY response, briefly assess your reasoning quality.
After EVERY tool use, check: did that help? Should I switch approaches?

## OPERATING PROCEDURE
1. ALWAYS start by classifying the problem domain (use cynefin_classify)
2. Based on the domain, follow the appropriate playbook
3. Use tools from the application layer as needed
4. Monitor your confidence throughout
5. Reflect at the end

## INTEGRATION RULES
- Classify before you choose
- Diverge before you converge
- Inversion bookends everything
- Depth proportional to stakes
- Cycle, don't waterfall — loop back when needed
- Update, don't anchor — each new insight should shift your view
"""
```

---

## 6. State Management

The agent needs a structured state object that accumulates across thinking system calls.

```python
@dataclass
class AgentState:
    # Problem definition
    problem: str
    domain: Optional[Domain] = None
    reframed_problem: Optional[str] = None

    # Phase tracking
    current_phase: Phase = Phase.ORIENT
    phases_completed: list[Phase] = field(default_factory=list)
    cycle_count: int = 0

    # Accumulated knowledge
    assumptions_identified: list[str] = field(default_factory=list)
    bedrock_truths: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    solutions: list[dict] = field(default_factory=list)
    failure_modes: list[dict] = field(default_factory=list)
    mental_models_applied: list[str] = field(default_factory=list)

    # Evaluation state
    confidence: float = 0.5
    confidence_history: list[float] = field(default_factory=list)
    bayesian_priors: dict = field(default_factory=dict)

    # Metacognitive state
    bias_alerts: list[str] = field(default_factory=list)
    approach_effectiveness: list[dict] = field(default_factory=list)
    effort_spent: float = 0.0

    # Decision state
    proposed_solution: Optional[dict] = None
    decision_rationale: Optional[str] = None
    mitigations: list[dict] = field(default_factory=list)

    # OODA tracking
    ooda_observations: list[dict] = field(default_factory=list)
    ooda_orientations: list[str] = field(default_factory=list)
```

---

## 7. Prompt Engineering Patterns

### Pattern: Metacognitive Wrapper

Every thinking system call gets wrapped with metacognitive monitoring:

```
[METACOGNITIVE MONITOR]
Phase: {phase} | Confidence: {confidence} | Budget: {remaining}
Alerts: {any_bias_alerts}

Before responding, briefly note:
- Is this approach working?
- What am I uncertain about?
- Should I suggest switching to a different system?

---

[THINKING SYSTEM: {system_name}]
{actual_system_prompt}

---

[OUTPUT REQUIREMENTS]
In addition to the system output, include:
- metacog_assessment: brief self-assessment of reasoning quality
- confidence: your confidence in this output (0.0 to 1.0)
- suggested_next: which thinking system(s) should run next
- loop_back: should we return to an earlier phase? which one?
```

### Pattern: Dialectical Synthesis

When two systems produce conflicting outputs, use a synthesis prompt:

```
Two thinking systems have produced different conclusions:

SYSTEM A ({system_a_name}):
{system_a_output}

SYSTEM B ({system_b_name}):
{system_b_output}

Apply DIALECTICAL THINKING:
1. What is the thesis (System A's core claim)?
2. What is the antithesis (System B's core claim)?
3. What is each system seeing that the other is missing?
4. Can you construct a SYNTHESIS that integrates the valid
   elements of both while resolving the contradiction?
5. If no synthesis is possible, which system's output is more
   reliable given the problem domain and current evidence?
```

### Pattern: Cross-Domain Provocation (Latticework)

```
You are applying mental models from {discipline} to this problem.

PROBLEM: {problem}
RELEVANT MODELS FROM {discipline}: {models}

For each model:
1. How does this model reframe the problem?
2. What does it predict would happen?
3. What does it suggest as a solution?
4. What would a {discipline} expert find obvious here that
   everyone else is missing?

Be specific and concrete — not just "this is like natural selection"
but "the specific selection pressure here is X, the variation is Y,
and the fitness landscape suggests Z."
```

---

## 8. Evaluation & Calibration

### Measuring Agent Quality

```python
@dataclass
class ThinkingQualityMetrics:
    # Did the agent classify the domain correctly?
    domain_classification_accuracy: float

    # Did it select appropriate systems for the domain?
    system_selection_relevance: float

    # Was the confidence well-calibrated?
    # (Brier score: lower is better)
    calibration_score: float

    # Did it identify the key insights?
    insight_coverage: float

    # Did it avoid the anti-patterns?
    antipattern_avoidance: float

    # Was effort proportional to stakes?
    effort_efficiency: float

    # Did it loop back when needed?
    appropriate_iteration: float
```

### Calibration Loop

Run the agent on problems with known good analyses. Compare its domain classification, system selection, and conclusions against expert benchmarks. Use the results to tune:

- The Cynefin classifier's prompts and thresholds
- The Kernel's bias detection rules
- The Bounded Rationality governor's effort budgets
- Each thinking system's prompt templates

---

## 9. Implementation Roadmap

### Phase 1 — MVP (1-2 weeks)
Build Variation D (tool-use agent) with:
- Cynefin classifier tool
- 5 core thinking system tools (First Principles, Inversion, Pre-mortem, Bayesian, Systems Thinking)
- Kernel as system prompt
- Simple state tracking

### Phase 2 — Full Stack (2-4 weeks)
- Add all 24 thinking system tools
- Implement the Kernel as a proper monitoring layer
- Add the OODA conductor for multi-cycle reasoning
- Implement playbooks as predefined tool sequences

### Phase 3 — Multi-Agent (4-8 weeks)
- Split thinking systems into separate agents
- Build the conductor agent
- Implement parallel execution
- Add dialectical synthesis for conflicting outputs

### Phase 4 — Self-Improving (ongoing)
- Calibration testing suite
- Automatic prompt refinement based on outcome tracking
- A/B testing of system combinations
- User feedback integration to improve system selection
