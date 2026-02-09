"""09 — Server Tools (xAI Responses API).

Use the xai-responses provider to access server-side tools like
web_search and x_search that run on xAI's infrastructure.
"""

from ai_arch_toolkit import Client, ServerTool

# Note: must use "xai-responses" provider (not "xai") for ServerTool support
client = Client("xai-responses", model="grok-4-1-fast-reasoning")

print("=== Web Search ===")
resp = client.chat(
    "What were the major tech news stories this week?",
    server_tools=[ServerTool(type="web_search")],
)
print(resp.text[:500], "...\n")

print("=== X (Twitter) Search ===")
resp2 = client.chat(
    "What are people saying about Python 3.14 on X?",
    server_tools=[ServerTool(type="x_search")],
)
print(resp2.text[:500], "...")
