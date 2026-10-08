"""Shell tools — command execution via subprocess (T09).

A command's output exists only for the call that ran it: reading on would run the command again.
So an output that does not fit shows its start, and the footer says its size and how to narrow
the command (``_output``). Each stream is drained as it comes, so memory stays within the limit
however much the command prints.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Annotated

from ai_arch_toolkit.core import Range, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._output import Head, fit

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 8000
_MAX_TIMEOUT = 600
_MAX_OUTPUT = 100_000
_CHUNK = 1 << 16
# How long the readers may take to drain what a command printed before it ended at its deadline.
_DRAIN_S = 0.5
_NARROW = (
    "the rest is not kept: run the command again narrowed, e.g. | grep PATTERN, | head -n N, "
    "| tail -n N or | sed -n 'A,Bp'"
)


@tool(
    capability="shell",
    risk_level="critical",
    requires_approval=True,
    approval_reason="Shell command execution can read, write, or destroy local system data.",
)
def run_command(
    command: str,
    timeout: Annotated[int, Range(1, _MAX_TIMEOUT)] = _DEFAULT_TIMEOUT,
    max_output: Annotated[int, Range(1, _MAX_OUTPUT)] = _DEFAULT_MAX_OUTPUT,
    cwd: str | None = None,
) -> str:
    """Run a shell command and return its output.

    What the command did is the answer, also when it fails: its output, ``[stderr]``, a non-zero
    ``[exit code: N]``, or that it timed out. An output longer than ``max_output`` shows its start
    and its size; it is not kept, so narrow the command to see another part (``| grep``,
    ``| tail -n``, ``| sed -n 'A,Bp'``).

    Args:
        command: The shell command to execute.
        timeout: How many seconds to wait for it.
        max_output: How many characters of output to return, stdout and stderr together.
        cwd: The folder to run the command in. Defaults to the current one.

    Raises:
        ToolFailure: validation_error when ``cwd`` is not a folder or the command has a null
            byte; upstream when the system could not start the shell.
    """
    # Only the command's own process changes folder: the caller's stays where it was.
    folder = None if cwd is None else _folder(cwd)
    if cwd is not None and folder is None:
        raise ToolFailure(
            "validation_error", f"cwd {cwd!r} is not a directory; give a folder that exists."
        )
    ran = _run(command, timeout, folder, keep=max_output + 1)
    if ran is None:
        return f"Command timed out after {timeout}s: {command}"
    code, stdout, stderr = ran
    out, err = fit((stdout, stderr), limit=max_output, rest=_NARROW)
    parts = [out] if stdout.total else []
    if stderr.total:
        parts.append(f"[stderr]\n{err}")
    if code != 0:
        parts.append(f"[exit code: {code}]")
    return "\n".join(parts) if parts else "[no output]"


def _run(
    command: str, timeout: int, folder: Path | None, *, keep: int
) -> tuple[int, Head, Head] | None:
    """Run ``command``: its exit code and the start of its stdout and stderr, or ``None`` when it
    outlived ``timeout`` (or left a process holding its output open that long).

    Each stream is read as text (UTF-8, an undecodable byte replaced) by a thread of its own,
    which keeps its first ``keep`` characters and counts the rest.

    Raises:
        ToolFailure: validation_error for a null byte in the command; upstream when the shell
            cannot start.
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            cwd=folder,
        )
    except ValueError as e:  # a null byte in the command
        raise ToolFailure(
            "validation_error", f"the command cannot run: {e}; remove the null byte."
        ) from e
    except OSError as e:
        raise ToolFailure("upstream", f"could not start the shell: {e}") from e
    deadline = time.monotonic() + timeout
    heads = (Head(keep), Head(keep))
    readers = [
        threading.Thread(target=_drain, args=(stream, head), daemon=True)
        for stream, head in zip((process.stdout, process.stderr), heads, strict=True)
    ]
    for reader in readers:
        reader.start()
    try:
        code = process.wait(timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return None
    for reader in readers:
        reader.join(max(deadline - time.monotonic(), _DRAIN_S))
    if any(reader.is_alive() for reader in readers):
        return None
    return code, *heads


def _drain(stream: IO[str] | None, head: Head) -> None:
    """Read ``stream`` to its end into ``head``."""
    if stream is None:
        return
    with stream:
        while chunk := stream.read(_CHUNK):
            head.add(chunk)


def _folder(path: str) -> Path | None:
    """``path`` as a folder that exists, or ``None``."""
    try:
        folder = Path(path).expanduser()
    except RuntimeError:  # "~name" for a user this system does not have
        return None
    return folder if folder.is_dir() else None
