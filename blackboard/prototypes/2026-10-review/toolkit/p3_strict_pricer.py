import pathlib
import sys, asyncio
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "tests"))  # the repo root, 4 levels up
from fake_provider import fake_llm
from ai_arch_toolkit.core import RunConfig, State, Step, Result
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy, budget_scope
from ai_arch_toolkit.toolkit.flow import Flow

MODEL = "acme-local-7b"

class AppPricer:
    def price(self, request, usage):
        # app prices its own local model at $1/MTok in and out
        return Cost.known(Money.from_usd((usage.input_tokens + usage.output_tokens) * 1e-6))

llm, prov = fake_llm(model=MODEL, base_url="http://127.0.0.1:9/v1")

async def call(snap):
    r = await llm.complete("hi", max_tokens=100)
    return Result(value=r.text, artifacts={"answer": r.text})

flow = Flow(Step(name="call", fn=call))
policy = BudgetPolicy(max_cost=1.0, reserve="strict")
for label, cfg in [
    ("RunConfig(controller=BudgetController(policy), pricer=AppPricer())",
     RunConfig(controller=BudgetController(policy), pricer=AppPricer())),
    ("RunConfig(pricer=AppPricer()) measure-only", RunConfig(pricer=AppPricer())),
]:
    res = flow.run_sync(State(), config=cfg)
    print(label, "->", [(s.name, s.error) for s in res.trace.steps], "provider calls", prov.calls)

with budget_scope(policy, pricer=AppPricer()):
    res = flow.run_sync(State())
print("budget_scope(policy, pricer=AppPricer()) ->", [(s.name, s.error) for s in res.trace.steps], "provider calls", prov.calls)
