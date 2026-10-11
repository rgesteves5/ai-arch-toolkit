import datetime
from ai_arch_toolkit import tool, run_tools_sync, Response, ToolCall, Usage
side = []
@tool
def book(day: str) -> dict:
    """Book a slot."""
    side.append(day)
    return {"booked": datetime.date(2026, 1, 2)}
resp = Response(text="", usage=Usage(), model="m", tool_calls=[ToolCall(id="1", name="book", input={"day": "mon"})])
try:
    print(run_tools_sync(resp, [book]))
except Exception as e:
    print("run_tools_sync RAISED", type(e).__name__, e, "| side effects:", side)
