import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "tests"))  # the repo root, 4 levels up
import inspect
from fake_provider import fake_llm
from ai_arch_toolkit.core import State, Policy, RetryConfig, ProviderTimeout, Response, Usage, ToolGroup
from ai_arch_toolkit.toolkit.agents.flows import plan_execute_flow, plan_execute_initial_state

print(inspect.signature(ProviderTimeout.__init__))
ok = Response(text="1. do it", usage=Usage(input_tokens=1, output_tokens=1), model="claude-sonnet-4-6")
for exc in (ProviderTimeout("read timed out"),):
    llm, prov = fake_llm(exc, ok)
    flow = plan_execute_flow(llm, ToolGroup(), policy=Policy(retry=RetryConfig(max_retries=2, base_delay=0.01)))
    res = flow.run_sync(State(operational=plan_execute_initial_state("t")))
    st = res.trace.steps[0]
    print(type(exc).__name__, "-> step error:", st.error, "| decisions:", st.policy_decisions, "| attempts:", st.attempts, "| provider calls:", prov.calls)
