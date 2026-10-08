"""Tests for toolkit/tools/_math.py."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._math import math_eval, unit_convert


class TestMathEval:
    def test_basic_arithmetic(self):
        assert math_eval("2 + 3") == "5"
        assert math_eval("10 - 4") == "6"
        assert math_eval("6 * 7") == "42"
        assert math_eval("15 / 3") == "5"

    def test_power(self):
        assert math_eval("2 ** 10") == "1024"
        assert math_eval("2^10") == "1024"  # caret alias

    def test_modulo_and_floor_div(self):
        assert math_eval("17 % 5") == "2"
        assert math_eval("17 // 5") == "3"

    def test_constants(self):
        result = float(math_eval("pi"))
        assert abs(result - 3.14159) < 0.001

    def test_functions(self):
        assert math_eval("sqrt(144)") == "12"
        assert math_eval("abs(-5)") == "5"
        assert math_eval("factorial(5)") == "120"
        assert math_eval("max(3, 7)") == "7"

    def test_trig(self):
        result = float(math_eval("sin(0)"))
        assert abs(result) < 0.0001

    def test_nested(self):
        assert math_eval("sqrt(abs(-16))") == "4"

    def test_division_by_zero(self):
        message = _refused(math_eval, "1 / 0")
        assert "division by zero" in message

    def test_syntax_error(self):
        _refused(math_eval, "2 +* 3")

    def test_unknown_function(self):
        message = _refused(math_eval, "evil(42)")
        assert "unknown function 'evil'" in message

    def test_no_builtins_access(self):
        _refused(math_eval, "__import__('os')")


class TestUnitConvert:
    def test_length(self):
        result = unit_convert(1, "km", "miles")
        assert "0.621371" in result

    def test_mass(self):
        result = unit_convert(1, "kg", "lbs")
        assert "2.20462" in result

    def test_temperature_c_to_f(self):
        result = unit_convert(100, "celsius", "fahrenheit")
        assert "212" in result

    def test_temperature_f_to_c(self):
        result = unit_convert(32, "f", "c")
        assert "0" in result

    def test_volume(self):
        result = unit_convert(1, "gal", "liters")
        assert "3.78541" in result

    def test_speed(self):
        result = unit_convert(100, "km/h", "mph")
        assert "62" in result

    def test_time(self):
        result = unit_convert(1, "hour", "minutes")
        assert "60" in result

    def test_area(self):
        result = unit_convert(1, "km2", "hectares")
        assert "100" in result

    def test_case_insensitive(self):
        result = unit_convert(100, "KM", "Miles")
        assert "62" in result

    def test_incompatible_units(self):
        message = _refused(unit_convert, 1, "km", "kg")
        assert "cannot convert from 'km' to 'kg'" in message

    def test_unknown_unit(self):
        _refused(unit_convert, 1, "furlong", "m")

    def test_aliases(self):
        r1 = unit_convert(1, "kilometer", "mile")
        r2 = unit_convert(1, "km", "mi")
        # Both should give same conversion
        assert "0.621371" in r1
        assert "0.621371" in r2

    def test_the_answer_states_the_precision_it_rounds_to(self):
        assert unit_convert(1, "km", "miles") == (
            "1 km = 0.621371 miles (rounded to 6 significant digits)"
        )

    def test_temperatures_round_to_the_same_precision(self):
        assert unit_convert(36.6, "celsius", "fahrenheit") == (
            "36.6 celsius = 97.88 fahrenheit (rounded to 6 significant digits)"
        )
        assert unit_convert(1, "c", "k").startswith("1 c = 274.15 k")

    def test_numbers_are_never_in_scientific_notation(self):
        assert unit_convert(1e9, "km", "mm").startswith("1000000000.0 km = 1000000000000000 mm")
        assert unit_convert(1e-7, "m", "km").startswith("0.0000001 m = 0.0000000001 km")

    def test_the_factors_are_the_exact_definitions(self):
        # A tablespoon is three teaspoons; the old six-digit factors made it 3.00001.
        assert unit_convert(1, "tbsp", "tsp").startswith("1 tbsp = 3 tsp")
        assert unit_convert(1_000_000, "lb", "g").startswith("1000000 lb = 453592000 g")
        assert unit_convert(1, "gal", "fl_oz").startswith("1 gal = 128 fl_oz")

    def test_a_result_beyond_a_float_is_a_validation_error(self):
        assert "beyond" in _refused(unit_convert, 1e308, "km", "mm")
        assert "finite" in _refused(unit_convert, float("nan"), "km", "mi")

    def test_an_integer_beyond_a_float_is_a_validation_error(self):
        # The validator keeps an int as it is; 10**400 has no float.
        assert "beyond a float's range" in _refused(unit_convert, 10**400, "km", "m")
        assert unit_convert(10**20, "km", "m").startswith("100000000000000000000 km = ")


class TestMathGuards:
    def test_a_long_expression_is_refused(self):
        message = _refused(math_eval, "1+" * 600 + "1")
        assert "expression longer than 1000 characters" in message

    @pytest.mark.parametrize("expression", ["factorial(3000)", "10**4000 * 10**4000", "7**9000"])
    def test_a_result_too_large_to_print_is_refused_before_it_is_computed(self, expression):
        assert "result too large" in _refused(math_eval, expression)

    def test_deep_nesting_is_an_error(self):
        _refused(math_eval, "-" * 999 + "1")

    def test_large_but_printable_results_still_work(self):
        assert math_eval("2**1000") == str(2**1000)
        assert math_eval("factorial(100)").startswith("93326215443944")
        assert math_eval("pow(3, 10**50, 7)") == str(pow(3, 10**50, 7))


def _refused(tool, *args) -> str:
    """The message of the validation_error the tool raises for ``args``."""
    with pytest.raises(ToolFailure) as caught:
        tool(*args)
    assert caught.value.error.type == "validation_error"
    return caught.value.error.message
