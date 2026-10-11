from ai_arch_toolkit import tool, ToolGroup, run_tools_sync, execute_tool, prepare_tools, Response, ToolCall, Usage, web_search
@tool
def ping() -> str:
    """Ping."""
    return "pong"
group = ToolGroup(ping)
tools = [group, web_search()]
print("prepare_tools sends:", [d.get("name") or d.get("type") for d in prepare_tools(tools)])
resp = Response(text="", usage=Usage(), model="m", tool_calls=[ToolCall(id="1", name="ping", input={})])
print("execute_tool:", execute_tool(resp.tool_calls[0], tools))
try:
    print(run_tools_sync(resp, tools))
except Exception as e:
    print("run_tools_sync RAISED", type(e).__name__, e)
