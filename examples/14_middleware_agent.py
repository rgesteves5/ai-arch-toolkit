"""14 — Middleware with ReAct Flow.

Shows how LLM middleware fires on every call inside a react_flow loop.
A CostLogger middleware tracks token usage across all LLM invocations.
"""

from ai_arch_toolkit import LLM, Request, Response, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.tools import geocode, get_weather


class CostLogger:
    """Middleware that logs token usage for every LLM call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def before(self, request: Request) -> Request:
        return request

    def after(self, request: Request, response: Response) -> Response:
        tokens = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        }
        self.calls.append(tokens)
        total = tokens["input"] + tokens["output"]
        print(f"  [middleware] Call #{len(self.calls)}: {total} tokens")
        return response


logger = CostLogger()
tools = ToolGroup(get_weather, geocode)
llm = LLM("gpt-4.1-nano", middleware=[logger])

flow = react_flow(
    llm,
    tools,
    system="You are a helpful assistant. Use tools to answer questions.",
    max_iterations=5,
)

print("Running flow...\n")
state = State(operational=react_initial_state("What's the weather in Tokyo and Paris?"))
result = flow.run_sync(state)

print(f"\nAnswer: {state['response'].text}")
print(f"Steps: {len(result.trace.steps)}")
print("\nMiddleware summary:")
print(f"  Total LLM calls: {len(logger.calls)}")
total_in = sum(c["input"] for c in logger.calls)
total_out = sum(c["output"] for c in logger.calls)
print(f"  Total tokens — in: {total_in}, out: {total_out}")
print(f"  Cost: ${result.total_cost:.4f}")
