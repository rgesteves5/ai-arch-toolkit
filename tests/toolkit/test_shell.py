"""Tests for toolkit/tools/_shell.py."""

from __future__ import annotations

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
    def test_a_command_with_a_null_byte_is_an_error_string(self):
        assert run_command("echo a\x00b").startswith("Failed to execute")

    def test_timeout_and_max_output_are_clamped(self):
        assert run_command("echo hello", timeout=-1) == "hello\n"
        assert run_command("echo hello", max_output=-1).startswith("h\n\n[Truncated")
