"""06 — Tool Registry (xAI).

Use the @tool decorator to auto-generate schemas from type hints and
docstrings, and let the ToolRegistry handle execution.
"""

from ai_arch_toolkit import Client, ToolRegistry, ToolResult, tool

registry = ToolRegistry()


@tool(registry=registry)
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


@tool(registry=registry)
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


# Show auto-generated definitions
print("Registered tools:")
for t in registry.definitions():
    print(f"  {t.name}: {t.description}")
print()

# Use with an LLM
client = Client("xai", model="grok-4-1-fast-reasoning")
messages = [{"role": "user", "content": "What is 42 * 17, and convert 100 km to miles?"}]

response = client.chat(messages, tools=registry.definitions())

while response.tool_calls:
    messages.append(response.to_message())
    for tc in response.tool_calls:
        result = registry.execute(tc)
        print(f"[Tool: {tc.name}({tc.arguments}) → {result}]")
        messages.append(ToolResult(tool_call_id=tc.id, name=tc.name, content=result))
    response = client.chat(messages, tools=registry.definitions())

print("\nAssistant:", response.text)
