"""Run a single BBEH strategy and save results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from ai_arch_toolkit.nanope.bbeh import bbeh_task

OUTPUT_DIR = Path(__file__).parent / "_results"

# Map model name to inspect model string.
# Models whose provider Inspect doesn't support natively use a dummy model;
# our solver handles the actual LLM call via ai-arch-toolkit.
_INSPECT_MODELS: dict[str, str] = {
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-5-mini": "openai/gpt-5-mini",
    "o4-mini": "openai/o4-mini",
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5-20251001",
    "grok-4-1-fast-reasoning": "openai/gpt-5-nano",  # dummy — solver uses grok internally
    "gemini-3.1-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
}


def run_and_save(strategy: str, model: str = "gpt-5-nano", **kwargs: object) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Build filename: strategy for gpt-5-nano, strategy_model for others
    if model == "gpt-5-nano":
        filename = f"{strategy}.json"
    else:
        safe_model = model.replace("/", "_").replace(".", "_")
        filename = f"{strategy}_{safe_model}.json"

    print(f"Running {strategy} with {model}...")
    task = bbeh_task(strategy=strategy, model=model, **kwargs)
    inspect_model = _INSPECT_MODELS.get(model, f"openai/{model}")
    logs = inspect_eval(task, model=inspect_model)
    log: EvalLog = logs[0]
    print(f"Status: {log.status}")

    samples_data = []
    total_cost = 0.0
    correct = 0
    total = 0
    errors = 0
    task_results: dict[str, dict] = {}

    if log.samples:
        for s in log.samples:
            meta = s.metadata or {}
            cost = meta.get("cost", 0.0)
            task_name = meta.get("task", "unknown")
            score_val = None
            if s.scores:
                for _, score in s.scores.items():
                    score_val = score.value
                    break
            is_correct = score_val == "C"
            output_text = str(s.output.completion[:500] if s.output else "")
            is_error = output_text.startswith("Error:")
            total_cost += cost
            total += 1
            if is_correct:
                correct += 1
            if is_error:
                errors += 1
            if task_name not in task_results:
                task_results[task_name] = {
                    "correct": 0,
                    "total": 0,
                    "cost": 0.0,
                    "errors": 0,
                }
            task_results[task_name]["total"] += 1
            task_results[task_name]["cost"] += cost
            if is_correct:
                task_results[task_name]["correct"] += 1
            if is_error:
                task_results[task_name]["errors"] += 1
            samples_data.append(
                {
                    "id": s.id,
                    "task": task_name,
                    "correct": is_correct,
                    "cost": cost,
                    "answer": output_text,
                    "target": str(s.target) if s.target else "",
                    "error": is_error,
                }
            )

    for t in task_results.values():
        t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.0
    result = {
        "strategy": strategy,
        "model": model,
        "total_samples": total,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
        "total_cost": total_cost,
        "cost_per_sample": total_cost / total if total > 0 else 0,
        "per_task": task_results,
        "samples": samples_data,
    }
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {filename}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total}), Cost: ${total_cost:.4f}")
    print(f"Errors: {errors}/{total}")


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5-nano"
    # Extra kwargs from remaining args (key=value format)
    kwargs: dict[str, object] = {}
    for arg in sys.argv[3:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            # Parse booleans
            if v.lower() == "true":
                kwargs[k] = True
            elif v.lower() == "false":
                kwargs[k] = False
            else:
                kwargs[k] = v
    run_and_save(strategy, model, **kwargs)
