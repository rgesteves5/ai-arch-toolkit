import sys, asyncio, tempfile, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "tests"))  # the repo root, 4 levels up
from fake_provider import fake_llm
from ai_arch_toolkit.core import State, Response, Usage, ToolGroup, tool, ToolCall
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.agents.flows._common import parse_score
from ai_arch_toolkit.toolkit.resources import Resource, MarkdownHeading

U = Usage(input_tokens=10, output_tokens=5); M = "claude-sonnet-4-6"

# 1. ReAct last turn: tool calls still executed after the final LLM call
ran = []
@tool
def send_email(to: str) -> str:
    """Send."""
    ran.append(to); return "sent"
resp = Response(text="", usage=U, model=M, tool_calls=[ToolCall(id="c1", name="send_email", input={"to": "x@example.com"})])
llm, prov = fake_llm(resp)
flow = react_flow(llm, ToolGroup(send_email), max_iterations=2)
st = State(operational=react_initial_state("go"))
res = flow.run_sync(st)
print("1. react max_iterations=2: llm calls", prov.calls, "| tool executions", len(ran), "| answer", repr(st.get("answer")))

# 2. parse_score
for t in ["Score: 0.7/1.0", "0.4 on a scale from 0 to 1", "Score: 0.3 (1 = best)", "-0.5"]:
    print("2. parse_score(%r) = %s" % (t, parse_score(t)))

# 3. MarkdownHeading stops at a fenced code comment
md = "## Install\n\nRun:\n\n```bash\n# create venv\npython -m venv .venv\n```\n\nThen done.\n\n## Use\nx\n"
r = Resource.from_text(md, media_type="text/markdown")
print("3. MarkdownHeading('Install') ->", repr(MarkdownHeading(heading="Install").select(r)))
