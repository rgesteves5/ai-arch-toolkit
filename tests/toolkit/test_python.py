"""python_repl's output: a program's output cannot be read again, so a long one shows its start,
its size and how to print less (T09)."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _python
from ai_arch_toolkit.toolkit.tools._python import python_repl

_PRINTED = sum(len(f"{n}\n") for n in range(6000))


def test_a_long_output_shows_its_start_its_size_and_how_to_print_less() -> None:
    result = python_repl("for n in range(6000):\n    print(n)")

    shown, footer = result.rsplit("\n[", 1)
    numbers = shown.splitlines()
    assert numbers == [str(n) for n in range(len(numbers))]
    assert len(shown) <= _python._MAX_OUTPUT
    assert footer.startswith(f"chars 0-{len(shown) + 1} of {_PRINTED} | the rest is not kept")
    assert "printing less" in footer


def test_the_last_value_and_an_error_come_after_a_long_output() -> None:
    value = python_repl("for n in range(6000):\n    print(n)\n'the answer'")
    error = python_repl("for n in range(6000):\n    print(n)\n1/0")

    assert value.endswith("]\n\nthe answer")
    assert error.endswith("]\n\nError: division by zero")


def test_what_a_program_prints_is_kept_only_up_to_the_limit() -> None:
    evaluator = _python._SafeEvaluator()

    for _ in range(50):
        evaluator.scope["print"]("y" * 10_000)

    assert evaluator.printed.total == 50 * 10_001
    assert len(evaluator.printed.text) == _python._MAX_OUTPUT + 1


def test_a_short_output_is_unchanged() -> None:
    assert python_repl('print("a")\nprint("b")\n2+2') == "a\nb\n\n4"


def test_a_refusal_keeps_what_was_printed_within_the_limit() -> None:
    with pytest.raises(ToolFailure) as caught:
        python_repl("for n in range(6000):\n    print(n)\nimport os")

    output = caught.value.error.details["output"]
    assert output.startswith("0\n1\n2\n") and "| the rest is not kept" in output


@pytest.mark.parametrize(
    ("code", "shown"),
    [
        ('"a".strip("a")', ""),  # an empty string is a value, not None
        ('print("x")\n""', "x\n\n"),
        ("None", "None"),
        ("x = 1", "None"),
        ('print("x")', "x"),
    ],
)
def test_a_value_shows_when_there_is_one_even_empty(code: str, shown: str) -> None:
    assert python_repl(code) == shown
