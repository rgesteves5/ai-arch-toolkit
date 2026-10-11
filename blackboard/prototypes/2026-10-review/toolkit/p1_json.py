import datetime, asyncio
from ai_arch_toolkit.core import tool, ToolGroup
from ai_arch_toolkit.core._response import ToolCall

@tool
def now() -> str:
    """Now."""
    return datetime.datetime(2026, 1, 1)

@tool
def tags() -> list:
    """Tags."""
    return {"a", "b"}

g = ToolGroup(now, tags)
for name in ("now", "tags"):
    try:
        r = g.execute(ToolCall(id="1", name=name, input={}))
        print(name, "->", r)
    except Exception as e:
        print(name, "RAISED", type(e).__name__, e)
