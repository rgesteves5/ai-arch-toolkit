import pathlib
import sys, asyncio
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "tests"))  # the repo root, 4 levels up
from fake_provider import fake_llm
from ai_arch_toolkit.core import State, Response, Usage, user
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.flow import Flow

U = Usage(input_tokens=10, output_tokens=5)
researcher_llm, rp = fake_llm(Response(text="research notes", usage=U, model="claude-sonnet-4-6"))
writer_llm, wp = fake_llm(Response(text="final article", usage=U, model="claude-sonnet-4-6"))

researcher = Agent(ReasoningSpec(strategy="react", system="research"), researcher_llm)
writer = Agent(ReasoningSpec(strategy="react", system="write"), writer_llm)

import dataclasses
pipeline = Flow(dataclasses.replace(researcher.as_step(), name="researcher"), dataclasses.replace(writer.as_step(), name="writer"), name="pipeline")
state = State(operational={"messages": [user("Write about X")]})
res = pipeline.run_sync(state)
print("researcher LLM calls:", rp.calls, "| writer LLM calls:", wp.calls)
print("final answer in state:", state.get("answer"))
print("trace:", [(s.name, s.error, [c.name for c in s.children]) for s in res.trace.steps])
