"""Run BBEH mini benchmark across all 3 strategies and dump results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from ai_arch_toolkit.nanope.bbeh import bbeh_task

MODEL = "gpt-5-nano"
STRATEGIES = ["baseline", "self_discovery", "react_tools"]
OUTPUT_DIR = Path(__file__).parent / "_results"


def run_strategy(strategy: str) -> EvalLog:
    """Run a single strategy and return the eval log."""
    print(f"\n{'=' * 60}")
    print(f"  Running strategy: {strategy}")
    print(f"{'=' * 60}\n")
    task = bbeh_task(strategy=strategy, model=MODEL)
    logs = inspect_eval(task, model="openai/gpt-5-nano")
    return logs[0]


def extract_results(log: EvalLog, strategy: str) -> dict:
    """Extract key metrics from an eval log."""
    samples = []
    total_cost = 0.0
    correct = 0
    total = 0
    task_results: dict[str, dict] = {}

    if log.samples:
        for s in log.samples:
            cost = (s.metadata or {}).get("cost", 0.0)
            task_name = (s.metadata or {}).get("task", "unknown")
            score_val = None
            if s.scores:
                for _scorer_name, score in s.scores.items():
                    score_val = score.value
                    break

            is_correct = score_val == "C"
            total_cost += cost
            total += 1
            if is_correct:
                correct += 1

            if task_name not in task_results:
                task_results[task_name] = {"correct": 0, "total": 0, "cost": 0.0}
            task_results[task_name]["total"] += 1
            task_results[task_name]["cost"] += cost
            if is_correct:
                task_results[task_name]["correct"] += 1

            samples.append(
                {
                    "id": s.id,
                    "task": task_name,
                    "correct": is_correct,
                    "cost": cost,
                    "answer": str(s.output.completion if s.output else ""),
                }
            )

    accuracy = correct / total if total > 0 else 0.0

    # Per-task accuracy
    for t in task_results.values():
        t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0

    return {
        "strategy": strategy,
        "model": MODEL,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "total_cost": total_cost,
        "cost_per_sample": total_cost / total if total > 0 else 0.0,
        "per_task": task_results,
        "samples": samples,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    strategies = sys.argv[1:] if len(sys.argv) > 1 else STRATEGIES

    for strategy in strategies:
        log = run_strategy(strategy)
        results = extract_results(log, strategy)
        all_results[strategy] = results

        # Save individual result
        out_path = OUTPUT_DIR / f"{strategy}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {strategy} results to {out_path}")
        acc = results["accuracy"]
        print(f"  Accuracy: {acc:.1%} ({results['correct']}/{results['total_samples']})")
        print(f"  Total cost: ${results['total_cost']:.4f}")

    # Save combined results
    combined_path = OUTPUT_DIR / "combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
