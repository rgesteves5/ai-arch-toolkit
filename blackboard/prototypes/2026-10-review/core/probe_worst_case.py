"""The worst case reads the LLM's max_tokens, not what the adapter sends (no network)."""

from __future__ import annotations

from ai_arch_toolkit.core import LLM
from ai_arch_toolkit.core._attempts import Arguments, prepare, request_facts
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._metering._store import SharedMeter
from ai_arch_toolkit.core._metering._admission import ResourceLimits
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._worst_case import worst_case
from ai_arch_toolkit.core._pricing import pricing

OPTIONS = {
    "thinking": False,
    "thinking_effort": None,
    "thinking_budget": None,
    "output_schema": None,
    "tool_choice": None,
    "json_mode": False,
    "logprobs": False,
}


def show(model: str, extra: dict, **options) -> None:
    llm = LLM(model, api_key="k")
    args = Arguments({**OPTIONS, **options}, extra)
    request = llm._prepare_call([{"role": "user", "content": "hi"}], None, None, args)
    prepared = prepare(llm._provider, request)
    params = getattr(prepared, "params", None)
    sent = (
        params.get("max_output_tokens") or params.get("max_tokens")
        if isinstance(params, dict)
        else getattr(params["config"], "max_output_tokens", None)
    )
    shared = SharedMeter(ResourceLimits(max_cost=Money.from_usd(100)))
    with MeterScope(RunConfig(shared=shared)) as scope:
        facts = request_facts(llm, request, "complete", scope)
        bound = worst_case(facts, pricing)
    print(
        f"{model:22} {extra} {options} -> sent max output {sent}, "
        f"facts declare {facts.declared_max_output_tokens}, worst-case output tokens "
        f"{bound.output_tokens if bound else None}"
    )


show("gpt-5.5", {"max_completion_tokens": 100_000})
show("claude-sonnet-4-5", {}, thinking=True)
show("claude-sonnet-4-5", {}, thinking_effort="high")
