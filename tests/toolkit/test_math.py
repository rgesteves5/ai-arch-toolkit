"""Tests for toolkit/tools/_math.py."""

from __future__ import annotations

import pytest

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
        result = math_eval("1 / 0")
        assert "Error" in result

    def test_syntax_error(self):
        result = math_eval("2 +* 3")
        assert "Error" in result

    def test_unknown_function(self):
        result = math_eval("evil(42)")
        assert "Error" in result

    def test_no_builtins_access(self):
        result = math_eval("__import__('os')")
        assert "Error" in result


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
        result = unit_convert(1, "km", "kg")
        assert "Cannot convert" in result

    def test_aliases(self):
        r1 = unit_convert(1, "kilometer", "mile")
        r2 = unit_convert(1, "km", "mi")
        # Both should give same conversion
        assert "0.621371" in r1
        assert "0.621371" in r2


class TestMathGuards:
    def test_a_long_expression_is_refused(self):
        assert math_eval("1+" * 600 + "1") == "Error: expression longer than 1000 characters"

    @pytest.mark.parametrize("expression", ["factorial(3000)", "10**4000 * 10**4000", "7**9000"])
    def test_a_result_too_large_to_print_is_refused_before_it_is_computed(self, expression):
        assert math_eval(expression).startswith("Error: result too large")

    def test_deep_nesting_is_an_error(self):
        assert math_eval("-" * 999 + "1").startswith("Error")

    def test_large_but_printable_results_still_work(self):
        assert math_eval("2**1000") == str(2**1000)
        assert math_eval("factorial(100)").startswith("93326215443944")
        assert math_eval("pow(3, 10**50, 7)") == str(pow(3, 10**50, 7))
