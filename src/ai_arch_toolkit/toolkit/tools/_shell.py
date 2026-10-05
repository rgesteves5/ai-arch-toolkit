"""Shell tools — command execution via subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure

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
    cwd: str | None = None,
) -> str:
    """Run a shell command and return its output.

    What the command did is the answer, also when it fails: its output, ``[stderr]``, a non-zero
    ``[exit code: N]``, or that it timed out.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait (1-600). Defaults to 30.
        max_output: Maximum characters of output to return (1-100000). Defaults to 8000.
        cwd: The folder to run the command in. Defaults to the current one.

    Raises:
        ToolFailure: validation_error when ``cwd`` is not a folder or the command has a null
            byte; upstream when the system could not start the shell.
    """
    timeout = max(1, min(timeout, _MAX_TIMEOUT))
    max_output = max(1, min(max_output, _MAX_OUTPUT))
    # Only the command's own process changes folder: the caller's stays where it was.
    folder = None if cwd is None else _folder(cwd)
    if cwd is not None and folder is None:
        raise ToolFailure(
            "validation_error", f"cwd {cwd!r} is not a directory; give a folder that exists."
        )
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=folder,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s: {command}"
    except ValueError as e:  # a null byte in the command
        raise ToolFailure(
            "validation_error", f"the command cannot run: {e}; remove the null byte."
        ) from e
    except OSError as e:
        raise ToolFailure("upstream", f"could not start the shell: {e}") from e

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


def _folder(path: str) -> Path | None:
    """``path`` as a folder that exists, or ``None``."""
    try:
        folder = Path(path).expanduser()
    except RuntimeError:  # "~name" for a user this system does not have
        return None
    return folder if folder.is_dir() else None
