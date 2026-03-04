"""Shell tools — command execution via subprocess."""

from __future__ import annotations

import subprocess

from ai_arch_toolkit.core import tool

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 8000


@tool
def run_command(
    command: str,
    timeout: int = _DEFAULT_TIMEOUT,
    max_output: int = _DEFAULT_MAX_OUTPUT,
) -> str:
    """Run a shell command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait. Defaults to 30.
        max_output: Maximum characters of output to return. Defaults to 8000.
    """
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
    except OSError as e:
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
