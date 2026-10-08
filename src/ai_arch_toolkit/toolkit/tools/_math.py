"""Math tools — safe expression evaluation and unit conversion."""

from __future__ import annotations

import ast
import math
import operator
from decimal import Decimal
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure

# ---------------------------------------------------------------------------
# Safe math evaluator
# ---------------------------------------------------------------------------

# Big-integer arithmetic runs in C holding the GIL: no timeout can stop it, not even the
# executor's. So each operation that can grow a number estimates its result first, and refuses
# one too large to print (str() of an int already refuses more than 4300 digits, ~14 300 bits).
_MAX_EXPRESSION_CHARS = 1000
_MAX_RESULT_BITS = 15_000
_MAX_ROUND_DIGITS = 1000


def _refuse_over(bits: float) -> None:
    if bits > _MAX_RESULT_BITS:
        msg = f"result too large (over {_MAX_RESULT_BITS} bits)"
        raise ValueError(msg)


def _power(base: Any, exponent: Any, modulus: Any = None) -> Any:
    whole = isinstance(base, int) and isinstance(exponent, int)
    if modulus is None and whole and exponent > 0 and abs(base) > 1:
        _refuse_over(exponent * math.log2(abs(base)))
    return pow(base, exponent) if modulus is None else pow(base, exponent, modulus)


def _multiply(left: Any, right: Any) -> Any:
    if isinstance(left, int) and isinstance(right, int):
        _refuse_over(abs(left).bit_length() + abs(right).bit_length())
    return left * right


def _factorial(n: Any) -> Any:
    if isinstance(n, int) and n > 1:
        _refuse_over(math.lgamma(n + 1) / math.log(2))
    return math.factorial(n)


def _round(number: Any, ndigits: Any = None) -> Any:
    if isinstance(ndigits, int) and abs(ndigits) > _MAX_ROUND_DIGITS:
        msg = f"round() ndigits must be within ±{_MAX_ROUND_DIGITS}"
        raise ValueError(msg)
    return round(number, ndigits)


_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: _multiply,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _power,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": _round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": _factorial,
    "gcd": math.gcd,
    "min": min,
    "max": max,
    "pow": _power,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            args = [_safe_eval(a) for a in node.args]
            return _FUNCTIONS[node.func.id](*args)
        raise ValueError(f"unknown function {ast.unparse(node.func)!r}")
    raise ValueError(f"unsupported expression {ast.unparse(node)[:80]!r}")


@tool(capability="compute")
def math_eval(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports: +, -, *, /, //, %, ** (power), parentheses.
    Constants: pi, e, tau, inf.
    Functions: sqrt, abs, round, sin, cos, tan, asin, acos, atan,
               log, log2, log10, exp, ceil, floor, factorial, gcd, min, max, pow.
    An expression longer than 1000 characters, or one whose result would be too large to print,
    is refused before it is computed.

    Args:
        expression: A math expression, e.g. "sqrt(144) + 3 * pi".

    Raises:
        ToolFailure: validation_error when the expression is too long, does not parse, uses
            something outside the list above, or cannot be computed (a division by zero, a
            result too large).
    """
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ToolFailure(
            "validation_error",
            f"expression longer than {_MAX_EXPRESSION_CHARS} characters; split it into parts",
        )
    # Allow ^ as power operator
    expression = expression.replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        # Format nicely — integers stay clean
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            return str(int(result))
        return str(result)
    except (ArithmeticError, RecursionError, MemoryError) as e:  # a valid expression, no value
        reason = str(e) or type(e).__name__
        raise ToolFailure("validation_error", f"the expression has no value: {reason}") from e
    except (ValueError, TypeError, SyntaxError) as e:
        raise ToolFailure(
            "validation_error",
            f"cannot evaluate the expression: {str(e) or type(e).__name__}; "
            "use only the operators, constants and functions math_eval lists",
        ) from e


# ---------------------------------------------------------------------------
# Unit converter
# ---------------------------------------------------------------------------

# The exact definitions (NIST Handbook 44, appendix C; NIST SP 811, appendix B), so the six
# significant digits a conversion shows are right: the international pound and foot, the US
# gallon (231 cubic inches) and its parts, the international acre.
_POUND = 453.59237
_FOOT = 0.3048
_MILE = 1609.344
_GALLON = 3.785411784
_ACRE = 4046.8564224

# All conversions go through a base unit per category.
# Format: {(category, unit_name): factor_to_base}
# Base units: meter, gram, second, kelvin, liter, m², m/s

_LENGTH: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "nm": 1852.0,
    "nautical_mile": 1852.0,
    "nautical_miles": 1852.0,
}

_MASS: dict[str, float] = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "mg": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
    "lb": _POUND,
    "lbs": _POUND,
    "pound": _POUND,
    "pounds": _POUND,
    "oz": _POUND / 16,
    "ounce": _POUND / 16,
    "ounces": _POUND / 16,
    "ton": 1_000_000.0,
    "tonne": 1_000_000.0,
    "tonnes": 1_000_000.0,
    "st": _POUND * 14,
    "stone": _POUND * 14,
}

_VOLUME: dict[str, float] = {
    "l": 1.0,
    "liter": 1.0,
    "liters": 1.0,
    "litre": 1.0,
    "litres": 1.0,
    "ml": 0.001,
    "milliliter": 0.001,
    "milliliters": 0.001,
    "gal": _GALLON,
    "gallon": _GALLON,
    "gallons": _GALLON,
    "qt": _GALLON / 4,
    "quart": _GALLON / 4,
    "quarts": _GALLON / 4,
    "pt": _GALLON / 8,
    "pint": _GALLON / 8,
    "pints": _GALLON / 8,
    "cup": _GALLON / 16,
    "cups": _GALLON / 16,
    "fl_oz": _GALLON / 128,
    "fluid_ounce": _GALLON / 128,
    "tbsp": _GALLON / 256,
    "tablespoon": _GALLON / 256,
    "tsp": _GALLON / 768,
    "teaspoon": _GALLON / 768,
}

_SPEED: dict[str, float] = {
    "m/s": 1.0,
    "mps": 1.0,
    "km/h": 1000 / 3600,
    "kmh": 1000 / 3600,
    "kph": 1000 / 3600,
    "mph": _MILE / 3600,
    "knot": 1852 / 3600,
    "knots": 1852 / 3600,
    "kn": 1852 / 3600,
    "ft/s": 0.3048,
    "fps": 0.3048,
}

_AREA: dict[str, float] = {
    "m2": 1.0,
    "sq_m": 1.0,
    "square_meter": 1.0,
    "km2": 1_000_000.0,
    "sq_km": 1_000_000.0,
    "ha": 10_000.0,
    "hectare": 10_000.0,
    "hectares": 10_000.0,
    "acre": _ACRE,
    "acres": _ACRE,
    "ft2": _FOOT**2,
    "sq_ft": _FOOT**2,
    "square_foot": _FOOT**2,
    "mi2": _MILE**2,
    "sq_mi": _MILE**2,
}

_TIME: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "wk": 604800.0,
    "week": 604800.0,
    "weeks": 604800.0,
}

_CATEGORIES: list[dict[str, float]] = [_LENGTH, _MASS, _VOLUME, _SPEED, _AREA, _TIME]


def _convert_temperature(value: float, from_u: str, to_u: str) -> float | None:
    """Temperature needs special handling — not a simple ratio."""
    temp_aliases = {
        "c": "c",
        "celsius": "c",
        "°c": "c",
        "f": "f",
        "fahrenheit": "f",
        "°f": "f",
        "k": "k",
        "kelvin": "k",
    }
    f = temp_aliases.get(from_u)
    t = temp_aliases.get(to_u)
    if f is None or t is None:
        return None
    # Convert to Celsius first
    if f == "c":
        c = value
    elif f == "f":
        c = (value - 32) * 5 / 9
    else:
        c = value - 273.15
    # Convert from Celsius to target
    if t == "c":
        return c
    if t == "f":
        return c * 9 / 5 + 32
    return c + 273.15


# A conversion shows this many significant digits, and says so: the factors are exact, but a
# float is not (T09).
_DIGITS = 6


@tool(capability="compute")
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units.

    Supports length, mass, volume, speed, area, time, and temperature. The answer is rounded to
    6 significant digits, and says so; numbers are never in scientific notation.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit, e.g. "km", "lbs", "celsius", "gallons".
        to_unit: Target unit, e.g. "miles", "kg", "fahrenheit", "liters".

    Raises:
        ToolFailure: validation_error when a unit is unknown, the two are not in the same
            category, or the value or the result is not a finite number.
    """
    try:
        finite = math.isfinite(value)
    except OverflowError as e:  # an int the validator kept, which no float holds
        raise ToolFailure(
            "validation_error", "the value is beyond a float's range; convert a smaller value"
        ) from e
    if not finite:
        raise ToolFailure("validation_error", f"value {value!r} is not a finite number; give one")
    result = _converted(value, from_unit.lower().strip(), to_unit.lower().strip())
    if result is None:
        raise ToolFailure(
            "validation_error",
            f"cannot convert from {from_unit!r} to {to_unit!r}; both units must be known and in "
            "the same category (length, mass, volume, speed, area, time or temperature)",
        )
    if not math.isfinite(result):
        raise ToolFailure(
            "validation_error",
            f"{_plain(value)} {from_unit} in {to_unit} is beyond a float's range; convert a "
            "smaller value",
        )
    return (
        f"{_plain(value)} {from_unit} = {_plain(result, _DIGITS)} {to_unit} "
        f"(rounded to {_DIGITS} significant digits)"
    )


def _converted(value: float, from_u: str, to_u: str) -> float | None:
    """``value`` in ``to_u``, or ``None`` when the two units are not in one category."""
    temperature = _convert_temperature(value, from_u, to_u)
    if temperature is not None:
        return temperature
    for category in _CATEGORIES:
        if from_u in category and to_u in category:
            return value * category[from_u] / category[to_u]
    return None


def _plain(number: float, digits: int | None = None) -> str:
    """``number`` in positional notation, never scientific: as given, or rounded to ``digits``
    significant digits."""
    written = repr(number) if digits is None else f"{number:.{digits}g}"
    return format(Decimal(written), "f")
