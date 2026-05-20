"""
Agent: Transform in a Loop

The public surface for agents.
An agent is content → content, like an LLM call.
But it's autonomous, multi-step, and observable.

Users need:
  1. Create an agent     (what model, what tools, what instructions)
  2. Run it              (give task, get result)
  3. Observe it          (watch it work)
  4. Drive it            (step by step, manual control)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# STEP — one iteration of the agent loop
#
# Not a final answer. An intermediate state.
# "Here's what I did, here's what happened."
# ═══════════════════════════════════════════════════════════════════


class Step:
    """
    One iteration of the agent loop.

    Represents: "I decided to do X, and here's what happened."

        step.kind           # "tool_call" | "response" | "self_modify" | "spawn" | "error"
        step.content        # what the LLM said / thought
        step.tool           # tool name (if kind == "tool_call")
        step.tool_input     # what was sent to the tool
        step.tool_output    # what came back
        step.cost           # cost of this step
        step.done           # is the agent finished after this step?

    Usage:
        async for step in agent.stream("do research"):
            if step.kind == "tool_call":
                print(f"Called {step.tool}({step.tool_input})")
                print(f"Got: {step.tool_output}")
            elif step.kind == "response":
                print(f"Final: {step.content}")
    """

    __slots__ = (
        "content",
        "cost",
        "done",
        "kind",
        "tool",
        "tool_input",
        "tool_output",
    )

    def __init__(
        self,
        kind: str,
        content: str = "",
        tool: str = "",
        tool_input: Any = None,
        tool_output: Any = None,
        cost: float = 0.0,
        done: bool = False,
    ):
        self.kind = kind
        self.content = content
        self.tool = tool
        self.tool_input = tool_input
        self.tool_output = tool_output
        self.cost = cost
        self.done = done

    def __str__(self) -> str:
        if self.kind == "tool_call":
            return f"[{self.tool}] {self.tool_input} → {self.tool_output}"
        return self.content

    def __repr__(self) -> str:
        return f"Step({self.kind}, done={self.done})"


# ═══════════════════════════════════════════════════════════════════
# RESPONSE — final result of an agent run
#
# Same as LLM Response but with agent-specific metadata.
# Still behaves like a string.
# ═══════════════════════════════════════════════════════════════════


class Response:
    """
    Final result of an agent run.

    Behaves like a string:
        result = await agent.run("research protein folding")
        print(result)  # the final answer

    Rich when you need it:
        result.text          # the final answer
        result.steps         # list[Step] — everything that happened
        result.total_cost    # total USD spent
        result.total_tokens  # total tokens consumed
        result.total_steps   # how many loop iterations
        result.elapsed       # wall clock seconds
        result.model         # which model ran
    """

    __slots__ = (
        "elapsed",
        "model",
        "steps",
        "text",
        "total_cost",
        "total_steps",
        "total_tokens",
    )

    def __init__(
        self,
        text: str = "",
        steps: list[Step] | None = None,
        total_cost: float = 0.0,
        total_tokens: int = 0,
        total_steps: int = 0,
        elapsed: float = 0.0,
        model: str = "",
    ):
        self.text = text
        self.steps = steps or []
        self.total_cost = total_cost
        self.total_tokens = total_tokens
        self.total_steps = total_steps
        self.elapsed = elapsed
        self.model = model

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text)


# ═══════════════════════════════════════════════════════════════════
# AGENT — the loop
#
# Three verbs:
#   .run()    — do the task, give me the result (blocking)
#   .stream() — do the task, show me each step (async iterator)
#   .step()   — do one iteration, give me control (manual)
#
# Same content → content shape as Transform.
# Different verbs because the operational weight is different.
# ═══════════════════════════════════════════════════════════════════


class Agent:
    """
    A Transform in a loop.

    Create:
        agent = Agent(
            model="claude-sonnet-4-5-20250929",
            tools=[search, read, save],
            instructions="You are a research assistant.",
        )

    Run (most common):
        result = await agent.run("Research protein folding advances")
        print(result)
        print(f"Took {result.total_steps} steps, cost ${result.total_cost:.4f}")

    Stream (watch it work):
        async for step in agent.stream("Research protein folding"):
            print(step)

    Step (manual control):
        session = agent.start("Research protein folding")
        while not session.done:
            step = await session.step()
            print(step)
            # inspect, intervene, modify memory, inject content
            if something_wrong:
                session.stop()

    Batch (multiple tasks):
        results = await agent.batch(["task1", "task2", "task3"])
    """

    def __init__(
        self,
        model: str,
        tools: list | None = None,
        instructions: str = "",
        *,
        # Memory (mutable sections — the learning surface)
        memory: dict[str, str] | None = None,
        # Limits
        max_steps: int = 50,
        max_time: float = 300.0,  # seconds
        max_cost: float | None = None,  # USD
        # Permissions
        allow_self_modify: bool = False,
        allow_spawn: bool = False,
        # Identity
        name: str = "",
        description: str = "",
    ): ...

    # ─── The three verbs ─────────────────────────────────────────

    async def run(
        self,
        task: str | list[dict],
        **kwargs,
    ) -> Response:
        """
        Run to completion. Blocking.

        Input: task (string or messages)
        Output: Response (behaves like a string)

        The agent loops internally until done or limits hit.
        """
        ...

    async def stream(
        self,
        task: str | list[dict],
        **kwargs,
    ) -> AsyncIterator[Step]:
        """
        Run and yield each step.

        async for step in agent.stream(task):
            if step.kind == "tool_call":
                print(f"Calling {step.tool}...")
            elif step.done:
                print(f"Done: {step.content}")
        """
        ...

    def start(
        self,
        task: str | list[dict],
        **kwargs,
    ) -> Session:
        """
        Start a session for manual step-by-step control.

        session = agent.start("do research")
        step = await session.step()     # one iteration
        session.inject(user("also check arxiv"))  # add context mid-run
        step = await session.step()     # next iteration
        session.stop()                  # halt early
        result = session.result         # final response
        """
        ...

    async def batch(
        self,
        tasks: list[str | list[dict]],
        **kwargs,
    ) -> list[Response]:
        """
        Run multiple tasks in parallel.
        Same agent, concurrent runs, independent memory.
        """
        ...

    # ─── Memory access (for inspection / intervention) ───────────

    @property
    def memory(self) -> dict[str, str]:
        """Current mutable memory sections. Read/write."""
        ...

    @property
    def instructions(self) -> str:
        """Frozen instructions. Read only."""
        ...


# ═══════════════════════════════════════════════════════════════════
# SESSION — manual control of the agent loop
#
# Returned by agent.start().
# Gives the user full control: step, inspect, inject, stop.
# This is the power-user interface.
# ═══════════════════════════════════════════════════════════════════


class Session:
    """
    Manual control of an agent run.

        session = agent.start("task")

        while not session.done:
            step = await session.step()
            print(step)

            # Inspect state
            print(session.steps)        # all steps so far
            print(session.cost)         # cost so far
            print(session.memory)       # current mutable memory

            # Intervene
            session.inject(user("also check this source"))
            session.inject(system("reminder: always cite sources"))

            # Modify memory mid-run
            session.memory["methodology"] = "Try deeper search"

            # Halt if needed
            if session.cost > 1.0:
                session.stop("Budget exceeded")

        result = session.result   # final Response
    """

    @property
    def done(self) -> bool:
        """Is the agent finished?"""
        ...

    async def step(self) -> Step:
        """Execute one loop iteration. Returns what happened."""
        ...

    def inject(self, message: dict) -> None:
        """Add a message to the agent's conversation mid-run."""
        ...

    def stop(self, reason: str = "") -> None:
        """Halt the agent."""
        ...

    @property
    def steps(self) -> list[Step]:
        """All steps executed so far."""
        ...

    @property
    def cost(self) -> float:
        """Total cost so far."""
        ...

    @property
    def memory(self) -> dict[str, str]:
        """Current mutable memory. Read/write."""
        ...

    @property
    def result(self) -> Response:
        """Final result (available after done=True or stop())."""
        ...


# ═══════════════════════════════════════════════════════════════════
# TOOL — what agents can use
#
# A tool is just a function with a description.
# The @tool decorator is the public API.
# ═══════════════════════════════════════════════════════════════════


def tool(
    fn=None,
    *,
    name: str = "",
    description: str = "",
):
    """
    Decorator. Turns a function into a tool an agent can use.

    @tool
    async def search(query: str) -> str:
        '''Search the web for information.'''
        return await do_search(query)

    @tool(name="read_pdf", description="Extract text from a PDF")
    async def read(url: str) -> str:
        return await extract_pdf(url)

    # Tools are just functions. They work normally too.
    result = await search("protein folding")

    # Pass to agent
    agent = Agent(model="...", tools=[search, read])
    """
    ...


# ═══════════════════════════════════════════════════════════════════
# FULL USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════
#
#
# --- Simple agent ---
#
#   agent = Agent(
#       model="claude-sonnet-4-5-20250929",
#       tools=[search, read],
#       instructions="You are a research assistant.",
#   )
#   result = await agent.run("What's new in protein folding?")
#   print(result)
#
#
# --- Watch it work ---
#
#   async for step in agent.stream("Research CRISPR advances"):
#       if step.kind == "tool_call":
#           print(f"  🔧 {step.tool}({step.tool_input})")
#       elif step.done:
#           print(f"  ✅ {step.content[:100]}")
#
#
# --- Manual control ---
#
#   session = agent.start("Analyze this dataset")
#   while not session.done:
#       step = await session.step()
#       print(step)
#       if session.cost > 2.0:
#           session.stop("Too expensive")
#   print(session.result)
#
#
# --- Self-modifying agent ---
#
#   agent = Agent(
#       model="claude-sonnet-4-5-20250929",
#       tools=[search, read, save],
#       instructions="You are a research assistant. Always cite sources.",
#       memory={
#           "methodology": "Start broad, then narrow.",
#           "learned": "",
#       },
#       allow_self_modify=True,
#   )
#
#   result = await agent.run("Research protein folding")
#   print(agent.memory["learned"])  # see what it learned
#
#
# --- Agent as a tool for another agent ---
#
#   researcher = Agent(model="...", tools=[search], instructions="Research things.")
#
#   @tool(description="Delegate research to a specialist")
#   async def delegate_research(topic: str) -> str:
#       result = await researcher.run(topic)
#       return str(result)
#
#   orchestrator = Agent(
#       model="...",
#       tools=[delegate_research, write_report],
#       instructions="You coordinate research and writing.",
#   )
#
# ═══════════════════════════════════════════════════════════════════
