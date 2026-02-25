"""06 — Tool Loop with @tool and ToolGroup.

Use the @tool decorator to auto-generate schemas from type hints and
docstrings. ToolGroup manages registration, and run_tools_sync handles
execution + tool_result message creation in a single call.
"""

from ai_arch_toolkit import LLM, run_tools_sync
from ai_arch_toolkit.core import ToolGroup, tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression to evaluate, e.g. "2 + 2".
    """
    try:
        result = eval(expression, {"__builtins__": {}})
    except Exception as e:
        return f"Error: {e}"
    return str(result)


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit (km, miles, kg, lbs, c, f).
        to_unit: Target unit.
    """
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v * 0.453592,
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
    }
    fn = conversions.get((from_unit.lower(), to_unit.lower()))
    if fn is None:
        return f"Unsupported conversion: {from_unit} → {to_unit}"
    return f"{fn(value):.2f} {to_unit}"


# Build a ToolGroup from decorated functions
group = ToolGroup(calculate, unit_convert)

# Show auto-generated definitions
print("Registered tools:")
for t in group.definitions:
    print(f"  {t['name']}: {t['description']}")
print()

# Tool loop — keep calling until the model stops requesting tools
llm = LLM("gpt-4.1-nano")
messages = [{"role": "user", "content": "What is 42 * 17, and convert 100 km to miles?"}]

response = llm.complete_sync(messages, tools=group)

while response.has_tool_calls:
    messages.append(response.to_message())
    tool_results = run_tools_sync(response, group)
    for tr in tool_results:
        print(f"[Tool result: {tr['name']} → {tr['content']}]")
    messages.extend(tool_results)
    response = llm.complete_sync(messages, tools=group)

print("\nAssistant:", response.text)
