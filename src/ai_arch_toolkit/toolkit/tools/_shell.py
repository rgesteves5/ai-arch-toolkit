"""Shell tools — command execution via subprocess."""

from __future__ import annotations

import subprocess

from ai_arch_toolkit.core import tool

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 8000
_MAX_TIMEOUT = 600
_MAX_OUTPUT = 100_000


@tool(
    capability="shell",
    risk_level="critical",
    requires_approval=True,
    approval_reason="Shell command execution can read, write, or destroy local system data.",
)
def run_command(
    command: str,
    timeout: int = _DEFAULT_TIMEOUT,
    max_output: int = _DEFAULT_MAX_OUTPUT,
) -> str:
    """Run a shell command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait (1-600). Defaults to 30.
        max_output: Maximum characters of output to return (1-100000). Defaults to 8000.
    """
    timeout = max(1, min(timeout, _MAX_TIMEOUT))
    max_output = max(1, min(max_output, _MAX_OUTPUT))
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s: {command}"
    except (OSError, ValueError) as e:  # ValueError: a null byte in the command
        return f"Failed to execute: {e}"

    output_parts: list[str] = []
    if result.stdout:
        output_parts.append(result.stdout)
    if result.stderr:
        output_parts.append(f"[stderr]\n{result.stderr}")
    if result.returncode != 0:
        output_parts.append(f"[exit code: {result.returncode}]")

    output = "\n".join(output_parts) if output_parts else "[no output]"

    if len(output) > max_output:
        return output[:max_output] + f"\n\n[Truncated — {len(output)} total chars]"
    return output
