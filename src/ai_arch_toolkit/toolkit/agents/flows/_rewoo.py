"""ReWOO as a Flow — Plan → Execute → Solve, three sequential phases."""

from __future__ import annotations

import re
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.flow._flow import Flow

_PLAN_RE = re.compile(r"#E(\d+)\s*=\s*(\w+)\[([^\]]*)\]")


def rewoo_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    system: str = "",
    planner_system: str = (
        "You are a planning agent. Break the task into steps.\n"
        "For each step, specify a tool call in the format:\n"
        "#E1 = ToolName[argument]\n#E2 = ToolName[argument using #E1]\n"
        "You can reference previous evidence with #E{n}."
    ),
    solver_system: str = (
        "You are a solving agent. Given the task and evidence from "
        "executed steps, provide the final answer."
    ),
    timeout: float | None = None,
    policy: Policy | None = None,
    planner_llm: LLM | None = None,
    solver_llm: LLM | None = None,
) -> Flow:
    """Create a ReWOO Flow — Plan with evidence placeholders, Execute, Solve.

    Args:
        llm: Default language model.
        tools: Tool group for execution.
        system: Base system prompt.
        planner_system: System prompt for the planner phase.
        solver_system: System prompt for the solver phase.
        timeout: Overall timeout in seconds.
        policy: Optional execution policy.
        planner_llm: Override LLM for planning.
        solver_llm: Override LLM for solving.
    """
    plan_llm = planner_llm or llm
    solve_llm = solver_llm or llm

    # Build tool descriptions from the group's provider-safe definitions.
    tool_schemas: dict[str, dict[str, Any]] = {}
    if hasattr(tools, "definitions"):
        tool_schemas = {d["name"]: d for d in tools.definitions}

    tool_descriptions = "\n".join(
        f"- {name}: {schema.get('description', 'No description')}"
        for name, schema in tool_schemas.items()
    )

    async def plan(snap: StateSnapshot) -> Result:
        """Generate a plan with #E{n} evidence placeholders."""
        task: str = snap.require("task")

        full_system = planner_system
        if tool_descriptions:
            full_system += f"\n\nAvailable tools:\n{tool_descriptions}"

        response = await plan_llm.complete([user(task)], system=full_system)
        plan_text = response.text

        # Parse plan steps
        plan_steps: list[tuple[str, str, str]] = []
        for match in _PLAN_RE.finditer(plan_text):
            eid, tool_name, raw_args = match.group(1), match.group(2), match.group(3)
            plan_steps.append((eid, tool_name, raw_args))

        return Result(
            value=plan_text,
            artifacts={
                "plan_text": plan_text,
                "plan_steps": plan_steps,
                "evidence": {},
            },
            usage=response.usage,
            cost=response.cost or 0.0,
        )

    async def execute(snap: StateSnapshot) -> Result:
        """Execute plan steps sequentially, substituting evidence references."""
        plan_steps: list[tuple[str, str, str]] = snap.require("plan_steps")
        evidence: dict[str, str] = dict(snap.get("evidence", {}))
        total_cost = 0.0

        for eid, tool_name, raw_args in plan_steps:
            # Substitute #E{n} references
            args = raw_args
            for ref_match in re.finditer(r"#E(\d+)", raw_args):
                ref_key = f"#E{ref_match.group(1)}"
                if ref_key in evidence:
                    args = args.replace(ref_key, evidence[ref_key])

            # Execute tool
            try:
                if tool_name not in tool_schemas:
                    result_str = f"Error: Unknown tool '{tool_name}'"
                else:
                    schema = tool_schemas[tool_name]
                    params = schema.get("parameters", schema.get("input_schema", {}))
                    props = params.get("properties", {})
                    first_param = next(iter(props), "input")

                    tc = ToolCall(
                        id=f"rewoo_e{eid}",
                        name=tool_name,
                        input={first_param: args.strip()},
                    )
                    exec_result = await tools.async_execute(tc)
                    result_str = exec_result.to_model_text()
            except AdmissionDenied:
                raise  # budget denial is terminal — the flow executor converts it
            except Exception as exc:
                result_str = f"Error: {exc}"

            evidence[f"#E{eid}"] = result_str

        return Result(
            value=evidence,
            artifacts={"evidence": evidence},
            cost=total_cost,
        )

    async def solve(snap: StateSnapshot) -> Result:
        """Synthesize final answer from task and evidence."""
        task: str = snap.require("task")
        evidence: dict[str, str] = snap.get("evidence", {})
        plan_text: str = snap.get("plan_text", "")

        evidence_block = "\n".join(f"{k}: {v}" for k, v in sorted(evidence.items()))

        response = await solve_llm.complete(
            [user(f"Task: {task}\n\nPlan:\n{plan_text}\n\nEvidence:\n{evidence_block}")],
            system=solver_system,
        )

        return Result(
            value=response.text,
            artifacts={"answer": response.text, "response": response},
            usage=response.usage,
            cost=response.cost or 0.0,
        )

    flow_policy = policy
    if timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=timeout)

    return Flow(
        Step(name="plan", fn=plan),
        Step(name="execute", fn=execute),
        Step(name="solve", fn=solve),
        name="rewoo",
        policy=flow_policy,
    )


def rewoo_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a rewoo_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {"task": task_str}
