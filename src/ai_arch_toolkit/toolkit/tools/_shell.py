"""Shell tools — command execution via subprocess (T09).

A command's output exists only for the call that ran it: reading on would run the command again.
So an output that does not fit shows its start, and the footer says its size and how to narrow
the command (``_output``).

A command runs in a process group of its own, and the calling thread alone reads it: both pipes
through one selector, each through an incremental UTF-8 decoder into a bounded head, so memory
stays within the limit however much the command prints. When the call returns, the whole group
has been killed and the pipes are closed: a process the command left in the background
(``yes &``, ``tail -f &``, a dev server) neither outlives the call nor keeps a reader busy; a
group still running when the program exits is killed at exit. Only a process that leaves the
group on its own (``setsid``, ``set -m``) is beyond reach. Process groups are POSIX, so on
Windows the tool refuses to run.
"""

from __future__ import annotations

import atexit
import codecs
import contextlib
import io
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from ai_arch_toolkit.core import Range, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._output import Head, fit

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 8000
_MAX_TIMEOUT = 600
_MAX_OUTPUT = 100_000
_CHUNK = 1 << 16
# How long processes the command left in the background may hold its output open once the shell
# has ended, before they are stopped.
_GRACE_S = 0.5
# The longest the reader waits between two looks at whether the shell has ended (the first wait
# is a millisecond, and each one doubles).
_POLL_S = 0.05
_POSIX = os.name == "posix"
# The process groups of the commands running now. A command the program outlives is stopped by
# its call; one still running when the program exits (its call in a daemon thread, which dies
# unfinished) is stopped here, at exit.
_RUNNING: set[int] = set()
_NARROW = (
    "the rest is not kept: run the command again narrowed, e.g. | grep PATTERN, | head -n N, "
    "| tail -n N or | sed -n 'A,Bp'"
)
_LINGERED = (
    "[the command ended and left processes in the background holding its output: they were "
    "stopped; end the command with wait to let them finish]"
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
    ``| tail -n``, ``| sed -n 'A,Bp'``). The command reads no input, and every process it starts
    is stopped when the call returns: one left in the background (``&``) half a second after the
    command ends, so end the command with ``wait`` to let it finish.

    Args:
        command: The shell command to execute.
        timeout: How many seconds to wait for it.
        max_output: How many characters of the command's output to show, stdout and stderr
            together; the notes that say what was cut come on top.
        cwd: The folder to run the command in. Defaults to the current one.

    Raises:
        ToolFailure: validation_error when ``cwd`` is not a folder or the command has a null
            byte; upstream when the system could not start the shell, or is not POSIX.
    """
    # Only the command's own process changes folder: the caller's stays where it was.
    folder = None if cwd is None else _folder(cwd)
    if cwd is not None and folder is None:
        raise ToolFailure(
            "validation_error", f"cwd {cwd!r} is not a directory; give a folder that exists."
        )
    ran = _run(command, timeout, folder, keep=max_output + 1)
    out, err = fit((ran.stdout, ran.stderr), limit=max_output, rest=_NARROW)
    parts = [out] if ran.stdout.total else []
    if ran.stderr.total:
        parts.append(f"[stderr]\n{err}")
    if ran.code is None:
        parts.append(
            f"[timed out after {timeout}s: the command and every process it started were "
            f"stopped; give it a longer timeout (at most {_MAX_TIMEOUT}) or make it do less]"
        )
    elif ran.code != 0:
        parts.append(f"[exit code: {ran.code}]")
    if ran.lingered:
        parts.append(_LINGERED)
    return "\n".join(parts) if parts else "[no output]"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Ran:
    """What a command did: its exit code (``None`` when it outlived its timeout), the start of
    its stdout and stderr, and whether processes it left in the background still held its
    output when it was stopped."""

    code: int | None
    stdout: Head
    stderr: Head
    lingered: bool


class _Stream:
    """A pipe's bytes as text, kept by a :class:`Head`: decoded as UTF-8 as they come (an
    undecodable byte replaced, a character split between two reads joined), with universal
    newlines, as ``subprocess`` reads text."""

    __slots__ = ("_decoder", "head")

    def __init__(self, keep: int) -> None:
        self.head = Head(keep)
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._decoder = io.IncrementalNewlineDecoder(utf8, translate=True)

    def add(self, data: bytes) -> None:
        self.head.add(self._decoder.decode(data))

    def end(self) -> None:
        """Flush what the decoder holds: a character the stream cut short is replaced."""
        self.head.add(self._decoder.decode(b"", final=True))


def _run(command: str, timeout: int, folder: Path | None, *, keep: int) -> _Ran:
    """Run ``command`` in a process group of its own, read its output until it ends or the
    deadline, then kill the group and close the pipes.

    Each stream keeps its first ``keep`` characters and counts the rest.

    Raises:
        ToolFailure: validation_error for a null byte in the command; upstream when the shell
            cannot start, or the system is not POSIX.
    """
    if not _POSIX:
        raise ToolFailure(
            "upstream",
            "run_command needs a POSIX system (Linux, macOS): it stops every process a command "
            "starts through the command's process group, which Windows does not have.",
        )
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=folder,
            start_new_session=True,
        )
    except ValueError as e:  # a null byte in the command
        raise ToolFailure(
            "validation_error", f"the command cannot run: {e}; remove the null byte."
        ) from e
    except OSError as e:
        raise ToolFailure("upstream", f"could not start the shell: {e}") from e
    _RUNNING.add(process.pid)
    streams = (_Stream(keep), _Stream(keep))
    try:
        ended, lingered = _read(process, streams, time.monotonic() + timeout)
    finally:
        _stop(process)
    for stream in streams:
        stream.end()
    return _Ran(
        code=process.returncode if ended else None,
        stdout=streams[0].head,
        stderr=streams[1].head,
        lingered=lingered,
    )


def _read(
    process: subprocess.Popen[bytes], streams: tuple[_Stream, _Stream], deadline: float
) -> tuple[bool, bool]:
    """Read both pipes, in this thread, until the shell has ended and they are closed, or until
    the deadline; once the shell has ended, the processes it left get ``_GRACE_S`` to close them.

    Returns:
        Whether the shell ended before the deadline, and whether its pipes were still open when
        the reading stopped.
    """
    pipes = {
        pipe.fileno(): stream
        for pipe, stream in zip((process.stdout, process.stderr), streams, strict=True)
        if pipe is not None
    }
    ended_at: float | None = None
    poll = 0.001
    with selectors.DefaultSelector() as selector:
        for fd in pipes:
            selector.register(fd, selectors.EVENT_READ)
        while True:
            now = time.monotonic()
            if ended_at is None and _ended(process):
                ended_at = now
            if ended_at is not None and not selector.get_map():
                return True, False
            until = deadline if ended_at is None else min(deadline, ended_at + _GRACE_S)
            if now >= until:
                return ended_at is not None, ended_at is not None
            wait, poll = min(until - now, poll), min(poll * 2, _POLL_S)
            if not selector.get_map():  # the pipes are closed, the shell still runs
                time.sleep(wait)
                continue
            for key, _ in selector.select(wait):
                if data := os.read(key.fd, _CHUNK):
                    pipes[key.fd].add(data)
                else:
                    selector.unregister(key.fd)


def _ended(process: subprocess.Popen[bytes]) -> bool:
    """Whether the shell has ended. It is not reaped yet (``WNOWAIT``): while its pid is taken,
    no other process group can have the id that ``_stop`` kills."""
    try:
        state = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:  # reaped elsewhere (SIGCHLD ignored)
        return True
    return state is not None


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Kill the command's process group, reap the shell and close the pipes."""
    _kill_group(process.pid)
    _RUNNING.discard(process.pid)
    process.wait()
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()


def _kill_group(pgid: int) -> None:
    # No member left (ProcessLookupError), or, on macOS, none but the ended shell (EPERM).
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


@atexit.register
def _stop_running() -> None:
    """Kill the groups of the commands still running as the program exits."""
    for pgid in list(_RUNNING):
        _kill_group(pgid)


def _folder(path: str) -> Path | None:
    """``path`` as a folder that exists, or ``None``."""
    try:
        folder = Path(path).expanduser()
    except RuntimeError:  # "~name" for a user this system does not have
        return None
    return folder if folder.is_dir() else None
