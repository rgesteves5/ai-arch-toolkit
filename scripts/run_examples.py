#!/usr/bin/env python3
"""Run all example scripts sequentially and persist their outputs.

This utility executes each Python file matching ``examples/[0-9][0-9]_*.py``
in lexical order and writes captured stdout/stderr to a per-example file under
``scripts/output``.

Output naming convention:
    ``<example_filename>.output.txt``
Example:
    ``examples/01_hello_world.py`` ->
    ``scripts/output/01_hello_world.py.output.txt``

The runner continues through all examples by default and records each command's
exit code and runtime in its output file.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_CALL_START_RE = re.compile(r"^\[TRACE\] \[CALL (\d+)\] START ", re.MULTILINE)
_CALL_END_RE = re.compile(r"^\[TRACE\] \[CALL (\d+)\] END\s+", re.MULTILINE)
_CALL_FAIL_RE = re.compile(r"^\[TRACE\] \[CALL (\d+)\] FAIL\s+", re.MULTILINE)
_CALL_SUMMARY_RE = re.compile(
    r"^\[TRACE\] CALL_SUMMARY total_calls=(\d+) completed_calls=(\d+) failed_calls=(\d+)$",
    re.MULTILINE,
)


def _default_python_executable() -> str:
    """Prefer project virtualenv Python when available."""
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@dataclass(slots=True)
class RunResult:
    """Result of executing a single example script."""

    example_path: Path
    output_path: Path
    exit_code: int
    duration_seconds: float
    validation_errors: list[str] = field(default_factory=list)


def _status_label(exit_code: int) -> str:
    """Return a human-readable status label for a process exit code."""
    if exit_code == 0:
        return "SUCCESS"
    if exit_code == 124:
        return "TIMED OUT"
    return "FAILED"


def _discover_examples(examples_dir: Path) -> list[Path]:
    """Return sorted example scripts matching the numeric naming pattern."""
    return sorted(examples_dir.glob("[0-9][0-9]_*.py"))


def _render_output_header(
    example_path: Path, command: list[str], duration_seconds: float, exit_code: int
) -> str:
    """Build a readable run report header for each output file."""
    status = _status_label(exit_code)
    timeout_hint = (
        "This run hit the configured timeout. Check partial output below."
        if exit_code == 124
        else "A non-zero exit code usually means missing env vars or runtime errors."
    )
    interpretation = "Command completed without process errors." if exit_code == 0 else timeout_hint
    return (
        "=== EXAMPLE RUN REPORT ===\n"
        f"example_script: {example_path}\n"
        f"command: {' '.join(command)}\n"
        f"status: {status}\n"
        f"exit_code: {exit_code}\n"
        f"duration_seconds: {duration_seconds:.3f}\n"
        f"interpretation: {interpretation}\n"
        "\n"
    )


def _render_stream_section(name: str, content: str) -> str:
    """Render a labeled output section, explicitly marking empty streams."""
    normalized = content if content else "<empty>\n"
    return f"=== {name} ===\n{normalized}\n"


def _render_validation_section(errors: list[str]) -> str:
    """Render validation details appended to output reports."""
    if not errors:
        return "=== OUTPUT VALIDATION ===\nPASS\n\n"
    lines = "\n".join(f"- {error}" for error in errors)
    return f"=== OUTPUT VALIDATION ===\nFAIL\n{lines}\n\n"


def _render_output_notes(stdout: str, stderr: str) -> str:
    """Return human-readable notes that help interpret common output patterns."""
    notes: list[str] = []
    assistant_lines = sum(1 for line in stdout.splitlines() if line.strip().startswith("Assistant:"))
    user_lines = sum(1 for line in stdout.splitlines() if line.strip().startswith("User:"))

    if assistant_lines > 0 and user_lines == 0:
        notes.append(
            "This output contains assistant replies only. "
            "The example likely uses hardcoded user prompts and only prints model responses."
        )
    if "[TRACE] [CALL " in stdout:
        notes.append(
            "Trace lines are present. Each [CALL NNN] START/END pair maps output to a specific SDK call."
        )
    if "RESPONSE_TEXT_BEGIN" in stdout or "STREAM_TEXT_BEGIN" in stdout:
        notes.append(
            "Full model text is included between *_BEGIN and *_END trace markers for each call."
        )
    if not stdout.strip() and not stderr.strip():
        notes.append(
            "Both stdout and stderr are empty. The script may have run silently or exited early."
        )
    if stderr.strip():
        notes.append("stderr is non-empty. Check the STDERR section for warnings or errors.")

    if not notes:
        notes.append("No interpretation hints detected.")

    bullet_lines = "\n".join(f"- {note}" for note in notes)
    return f"=== OUTPUT NOTES ===\n{bullet_lines}\n\n"


def _validate_output_content(content: str) -> list[str]:
    """Validate generated output report for trace boundaries and legacy snippets."""
    errors: list[str] = []
    if "preview=" in content:
        errors.append("Found legacy preview snippet (`preview=`). Full text markers are expected.")

    summary_match = _CALL_SUMMARY_RE.search(content)
    if summary_match is None:
        errors.append("Missing call summary marker (`[TRACE] CALL_SUMMARY ...`).")
        return errors

    total_calls = int(summary_match.group(1))
    completed_calls = int(summary_match.group(2))
    failed_calls = int(summary_match.group(3))
    start_ids = set(_CALL_START_RE.findall(content))
    end_ids = set(_CALL_END_RE.findall(content))
    fail_ids = set(_CALL_FAIL_RE.findall(content))
    terminal_ids = end_ids | fail_ids

    if total_calls != len(start_ids):
        errors.append(
            f"Call START marker count mismatch: summary={total_calls}, markers={len(start_ids)}."
        )
    if completed_calls + failed_calls > total_calls:
        errors.append(
            "Invalid call summary: completed_calls + failed_calls exceeds total_calls."
        )
    if total_calls > 0 and not start_ids:
        errors.append("Missing `[CALL ...] START` markers for traced calls.")

    missing_terminal = sorted(start_ids - terminal_ids)
    if missing_terminal:
        errors.append(
            "Some calls are missing terminal markers (`END` or `FAIL`): "
            + ", ".join(missing_terminal)
        )

    orphan_terminal = sorted(terminal_ids - start_ids)
    if orphan_terminal:
        errors.append(
            "Found terminal markers without matching START markers: "
            + ", ".join(orphan_terminal)
        )

    return errors


def _run_example(
    example_path: Path,
    *,
    python_executable: str,
    trace_runner_path: Path,
    project_src_path: Path,
    timeout_seconds: float | None,
    output_dir: Path,
) -> RunResult:
    """Execute one example and write its combined stdout/stderr output file."""
    command = [python_executable, str(trace_runner_path), str(example_path)]
    started = time.monotonic()
    output_path = output_dir / f"{example_path.name}.output.txt"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_src_path}:{existing_pythonpath}" if existing_pythonpath else str(project_src_path)
    )
    try:
        proc = subprocess.run(  # noqa: S603 - command is controlled and local file paths only
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        duration_seconds = time.monotonic() - started
        header = _render_output_header(
            example_path, command, duration_seconds, exit_code=proc.returncode
        )
        body = (
            f"{header}"
            f"{_render_output_notes(proc.stdout, proc.stderr)}"
            f"{_render_stream_section('STDOUT', proc.stdout)}"
            f"{_render_stream_section('STDERR', proc.stderr)}"
        )
        validation_errors = _validate_output_content(proc.stdout)
        body += _render_validation_section(validation_errors)
        output_path.write_text(body, encoding="utf-8")
        effective_exit_code = proc.returncode if proc.returncode != 0 else (1 if validation_errors else 0)
        return RunResult(
            example_path=example_path,
            output_path=output_path,
            exit_code=effective_exit_code,
            duration_seconds=duration_seconds,
            validation_errors=validation_errors,
        )
    except subprocess.TimeoutExpired as exc:
        duration_seconds = time.monotonic() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(
            "utf-8", errors="replace"
        )
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(
            "utf-8", errors="replace"
        )
        header = _render_output_header(example_path, command, duration_seconds, exit_code=124)
        timeout_content = (
            (
                f"{header}"
                f"timeout_seconds: {timeout_seconds}\n\n"
                f"{_render_output_notes(stdout, stderr)}"
                f"{_render_stream_section('PARTIAL STDOUT', stdout)}"
                f"{_render_stream_section('PARTIAL STDERR', stderr)}"
            )
        )
        validation_errors = _validate_output_content(stdout)
        timeout_content += _render_validation_section(validation_errors)
        output_path.write_text(timeout_content, encoding="utf-8")
        return RunResult(
            example_path=example_path,
            output_path=output_path,
            exit_code=124,
            duration_seconds=duration_seconds,
            validation_errors=validation_errors,
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all examples in sequence and write per-example output files to scripts/output."
        )
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Directory containing numbered example scripts (default: examples).",
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=None,
        help=(
            "Run a single example file instead of scanning --examples-dir "
            "(e.g. examples/13_self_discovery_agent.py)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/output"),
        help="Directory where output files are written (default: scripts/output).",
    )
    parser.add_argument(
        "--python",
        default=_default_python_executable(),
        help=(
            "Python executable to use for running examples "
            "(default: .venv/bin/python when available, else current interpreter)."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional per-example timeout in seconds.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing example.",
    )
    return parser.parse_args()


def main() -> int:
    """Run discovered examples and print a concise summary."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_src_path = Path(__file__).resolve().parents[1] / "src"
    trace_runner_path = Path(__file__).with_name("_example_trace_runner.py").resolve()

    if args.example is not None:
        example_path = args.example.resolve()
        if not example_path.exists():
            print(f"Example file not found: {example_path}")
            return 1
        examples = [example_path]
    else:
        examples = _discover_examples(args.examples_dir)
    if not examples:
        print(f"No matching examples found in: {args.examples_dir}")
        return 1

    results: list[RunResult] = []
    failures = 0
    for index, example_path in enumerate(examples, start=1):
        print(f"[{index:02d}/{len(examples):02d}] running {example_path}")
        result = _run_example(
            example_path,
            python_executable=args.python,
            trace_runner_path=trace_runner_path,
            project_src_path=project_src_path,
            timeout_seconds=args.timeout_seconds,
            output_dir=args.output_dir,
        )

        results.append(result)
        status = "ok" if result.exit_code == 0 else "failed"
        print(f"  -> {status} ({result.duration_seconds:.2f}s) | {result.output_path}")
        if result.validation_errors:
            print(f"     validation_errors: {len(result.validation_errors)}")
        if result.exit_code != 0:
            failures += 1
            if args.fail_fast:
                break

    print("\nSummary")
    print(f"  total: {len(results)}")
    print(f"  failed: {failures}")
    print(f"  output_dir: {args.output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
