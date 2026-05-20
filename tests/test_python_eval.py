"""Tests for the safe Python REPL with output capture."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._python import python_repl


class TestPythonReplOutputCapture:
    """Test REPL-style output: last expression + captured print."""

    def test_single_expression(self):
        assert python_repl("2 + 2") == "4"

    def test_sorted_list(self):
        assert python_repl("sorted(['c','a','b'])") == "['a', 'b', 'c']"

    def test_assignment_then_expression(self):
        assert python_repl("x = 2 + 2\nx") == "4"

    def test_assignment_then_sum(self):
        assert python_repl("x = [1,2,3]\nsum(x)") == "6"

    def test_print_captures_output(self):
        assert python_repl("x = 2 + 2\nprint(x)") == "4"

    def test_print_in_loop(self):
        assert python_repl("for x in [1,2,3]: print(x)") == "1\n2\n3"

    def test_prints_plus_last_expr(self):
        assert python_repl('print("a")\nprint("b")\n2+2') == "a\nb\n\n4"

    def test_assignment_only_returns_none(self):
        assert python_repl("x = 5") == "None"

    def test_print_with_sep(self):
        assert python_repl('print(1, 2, 3, sep=", ")') == "1, 2, 3"

    def test_print_with_end(self):
        assert python_repl('print("hello", end="!")\nprint(" world")') == "hello! world"

    def test_multiple_prints(self):
        assert python_repl('print("line1")\nprint("line2")') == "line1\nline2"


class TestPythonReplErrorCapture:
    """Test partial output capture on errors."""

    def test_error_with_partial_output(self):
        result = python_repl('print("start")\n1/0')
        assert result == "start\n\nError: division by zero"

    def test_error_without_output(self):
        result = python_repl("1/0")
        assert result == "Error: division by zero"

    def test_syntax_error(self):
        result = python_repl("def foo():")
        assert result.startswith("Error:")

    def test_blocked_function(self):
        result = python_repl("eval('1')")
        assert "Error:" in result
