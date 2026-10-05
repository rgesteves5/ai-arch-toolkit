"""Tests for toolkit/tools/_shell.py."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._shell import run_command


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

    def test_output_truncation(self):
        result = run_command("seq 10000", max_output=100)
        assert "Truncated" in result

    def test_no_output(self):
        result = run_command("true")
        assert "[no output]" in result


class TestArguments:
    def test_a_command_with_a_null_byte_is_a_validation_error(self):
        with pytest.raises(ToolFailure) as caught:
            run_command("echo a\x00b")

        assert caught.value.error.type == "validation_error"
        assert "null byte" in caught.value.error.message

    def test_a_shell_that_cannot_start_is_an_upstream_failure(self):
        with (
            patch.object(subprocess, "run", side_effect=OSError("no /bin/sh")),
            pytest.raises(ToolFailure) as caught,
        ):
            run_command("true")

        assert caught.value.error.type == "upstream"
        assert "no /bin/sh" in caught.value.error.message

    def test_timeout_and_max_output_are_clamped(self):
        assert run_command("echo hello", timeout=-1) == "hello\n"
        assert run_command("echo hello", max_output=-1).startswith("h\n\n[Truncated")


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
