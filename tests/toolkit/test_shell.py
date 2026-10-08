"""Tests for toolkit/tools/_shell.py."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core import ApprovalDecision, ToolCall, ToolGroup
from ai_arch_toolkit.core._tools._result import ToolFailure
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

    def test_a_process_left_holding_the_output_ends_at_the_timeout(self):
        assert "timed out" in run_command("sleep 3 & echo started", timeout=1)


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
