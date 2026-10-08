"""Tests for toolkit/tools/_shell.py."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core import ApprovalDecision, ToolCall, ToolGroup
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _shell
from ai_arch_toolkit.toolkit.tools._shell import run_command

_SEQ_10000 = sum(len(f"{n}\n") for n in range(1, 10_001))


class TestRunCommand:
    def test_simple_command(self):
        result = run_command("echo hello")
        assert "hello" in result

    def test_exit_code(self):
        result = run_command("false")
        assert "exit code" in result

    def test_stderr(self):
        result = run_command("echo err >&2")
        assert "[stderr]" in result
        assert "err" in result

    def test_timeout(self):
        result = run_command("sleep 10", timeout=1)
        assert "timed out" in result.lower()

    def test_no_output(self):
        result = run_command("true")
        assert "[no output]" in result


class TestLongOutput:
    """A command's output cannot be read again: the footer says its size and how to narrow it."""

    def test_the_start_shows_ending_on_a_line_with_the_size_and_how_to_narrow(self):
        result = run_command("seq 10000", max_output=100)

        shown, footer = result.rsplit("\n[", 1)
        numbers = shown.splitlines()
        assert numbers == [str(n) for n in range(1, len(numbers) + 1)] and len(shown) < 100
        assert footer.startswith(
            f"chars 0-{len(shown) + 1} of {_SEQ_10000} | the rest is not kept"
        )
        assert "| grep" in footer and "| tail -n" in footer and "sed -n" in footer

    def test_stderr_and_the_exit_code_survive_a_long_stdout(self):
        result = run_command("seq 10000; echo boom >&2; exit 3", max_output=200)

        assert "\n[stderr]\nboom\n" in result
        assert result.endswith("\n[exit code: 3]")
        assert f"of {_SEQ_10000} |" in result

    def test_the_size_counts_everything_the_command_printed(self):
        result = run_command("yes | head -c 30000000", max_output=50)

        assert "of 30000000 |" in result
        assert len(result) < 400

    def test_output_that_is_not_utf8_is_read_with_replacement_characters(self):
        assert run_command("printf 'a\\377b'") == "a�b"


class TestArguments:
    def test_a_command_with_a_null_byte_is_a_validation_error(self):
        with pytest.raises(ToolFailure) as caught:
            run_command("echo a\x00b")

        assert caught.value.error.type == "validation_error"
        assert "null byte" in caught.value.error.message

    def test_a_shell_that_cannot_start_is_an_upstream_failure(self):
        with (
            patch.object(subprocess, "Popen", side_effect=OSError("no /bin/sh")),
            pytest.raises(ToolFailure) as caught,
        ):
            run_command("true")

        assert caught.value.error.type == "upstream"
        assert "no /bin/sh" in caught.value.error.message

    @pytest.mark.parametrize(
        "arguments",
        [{"timeout": 0}, {"timeout": 601}, {"max_output": 0}, {"max_output": 100_001}],
    )
    def test_the_executor_refuses_limits_past_the_schema(self, arguments):
        group = ToolGroup(run_command, approval_handler=lambda _r: ApprovalDecision.approve())

        result = group.execute(
            ToolCall(id="c", name="run_command", input={"command": "true", **arguments})
        )

        assert result.error is not None and result.error.type == "validation_error"


class TestWorkingDirectory:
    """``cwd`` runs the command in a folder without moving the process (G-26)."""

    def test_the_command_runs_in_cwd(self, tmp_path):
        (tmp_path / "marker.txt").write_text("x")

        assert run_command("ls", cwd=str(tmp_path)) == "marker.txt\n"

    def test_the_process_stays_where_it_was(self, tmp_path):
        before = os.getcwd()

        run_command("true", cwd=str(tmp_path))

        assert os.getcwd() == before

    def test_cwd_expands_the_home_folder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "marker.txt").write_text("x")

        assert run_command("ls", cwd="~") == "marker.txt\n"

    def test_a_cwd_that_is_not_a_folder_is_a_validation_error(self, tmp_path):
        missing = tmp_path / "missing"
        a_file = tmp_path / "file.txt"
        a_file.write_text("x")

        for cwd in (str(missing), str(a_file), "~no_such_user_g26"):
            with pytest.raises(ToolFailure) as caught:
                run_command("true", cwd=cwd)
            assert caught.value.error.type == "validation_error"
            assert f"cwd {cwd!r} is not a directory" in caught.value.error.message


def _alive(marker: str) -> bool:
    """Whether a process whose command line holds ``marker`` is running."""
    found = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True, check=False)
    return bool(found.stdout.split())


def _gone(marker: str, within: float = 3.0) -> bool:
    """Whether every process with ``marker`` is gone within ``within`` seconds (a killed one is
    reaped by the system, not at once)."""
    end = time.monotonic() + within
    while _alive(marker):
        if time.monotonic() > end:
            return False
        time.sleep(0.05)
    return True


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


class TestNothingIsLeftBehind:
    """When the call returns, every process the command started is stopped and its pipes are
    closed, and no thread reads on: a background writer, a sleeper and a follower holding the
    output, a follower in the foreground, and a sleeper that let go of it."""

    @pytest.fixture
    def marked(self, tmp_path: Path) -> Iterator[dict[str, str]]:
        """Words that mark this test's processes, so they can be found, and killed if a check
        fails."""
        digits = f"{random.randrange(10**8, 10**9)}"
        marks = {
            "word": f"t09a{digits}",
            "seconds": f"30.{digits}",
            "log": str(tmp_path / f"t09a{digits}.log"),
        }
        Path(marks["log"]).write_text("one line\n")
        yield marks
        for mark in (marks["word"], f"sleep {marks['seconds']}"):
            subprocess.run(["pkill", "-9", "-f", mark], check=False)

    @pytest.mark.parametrize(
        ("command", "mark"),
        [
            ("yes {word} &", "word"),
            ("sleep {seconds} &", "seconds"),
            ("tail -f {log} &", "word"),
            ("tail -f {log}", "word"),
            ("sleep {seconds} >/dev/null 2>&1 &", "seconds"),
        ],
    )
    def test_no_process_fd_or_thread_outlives_the_call(
        self, command: str, mark: str, marked: dict[str, str]
    ) -> None:
        marker = marked["word"] if mark == "word" else f"sleep {marked['seconds']}"
        fds, threads = _open_fds(), threading.active_count()
        started = time.monotonic()

        run_command(command.format(**marked), timeout=1)

        took = time.monotonic() - started
        assert (_open_fds(), threading.active_count()) == (fds, threads)
        assert _gone(marker), f"{marker!r} still runs"
        assert took < 3

    def test_a_command_still_running_when_the_program_exits_is_stopped(
        self, marked: dict[str, str]
    ) -> None:
        # The executor runs a tool in a daemon thread, which dies with the program unfinished.
        code = (
            "import threading, time\n"
            "from ai_arch_toolkit.toolkit.tools._shell import run_command\n"
            "threading.Thread(target=run_command, args=('sleep {seconds}',),"
            " kwargs={{'timeout': 60}}, daemon=True).start()\n"
            "time.sleep(0.5)\n"
        ).format(**marked)

        subprocess.run([sys.executable, "-c", code], timeout=30, check=True)

        assert _gone(f"sleep {marked['seconds']}")

    def test_the_timeout_keeps_what_was_printed_and_says_what_to_do(self) -> None:
        result = run_command("echo before; echo oops >&2; sleep 10", timeout=1)

        assert result.startswith("before\n\n[stderr]\noops\n\n")
        assert result.endswith(
            "[timed out after 1s: the command and every process it started were stopped; "
            "give it a longer timeout (at most 600) or make it do less]"
        )

    def test_a_command_that_leaves_a_process_in_the_background_ends_with_its_exit_code(
        self, marked: dict[str, str]
    ) -> None:
        started = time.monotonic()

        result = run_command(f"sleep {marked['seconds']} & echo started; exit 4", timeout=20)

        assert time.monotonic() - started < 3  # the grace, not the timeout
        assert result == (
            "started\n\n[exit code: 4]\n"
            "[the command ended and left processes in the background holding its output: they "
            "were stopped; end the command with wait to let them finish]"
        )

    def test_wait_lets_the_background_finish(self) -> None:
        assert run_command("(sleep 0.2; echo late) & echo now; wait") == "now\nlate\n"

    def test_the_command_reads_no_input(self) -> None:
        started = time.monotonic()

        assert run_command("cat", timeout=20) == "[no output]"
        assert time.monotonic() - started < 3

    def test_windows_is_refused_as_it_has_no_process_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_shell, "_POSIX", False)

        with pytest.raises(ToolFailure) as caught:
            run_command("echo hello")

        assert caught.value.error.type == "upstream"
        assert "POSIX" in caught.value.error.message
