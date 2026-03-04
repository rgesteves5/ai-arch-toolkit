"""Math tools — safe expression evaluation and unit conversion."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ai_arch_toolkit.core import tool

# ---------------------------------------------------------------------------
# Safe math evaluator
# ---------------------------------------------------------------------------

_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
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
    "round": round,
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
    "factorial": math.factorial,
    "gcd": math.gcd,
    "min": min,
    "max": max,
    "pow": pow,
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
        raise ValueError(f"Unknown function: {ast.dump(node.func)}")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def math_eval(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports: +, -, *, /, //, %, ** (power), parentheses.
    Constants: pi, e, tau, inf.
    Functions: sqrt, abs, round, sin, cos, tan, asin, acos, atan,
               log, log2, log10, exp, ceil, floor, factorial, gcd, min, max, pow.

    Args:
        expression: A math expression, e.g. "sqrt(144) + 3 * pi".
    """
    # Allow ^ as power operator
    expression = expression.replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        # Format nicely — integers stay clean
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            return str(int(result))
        return str(result)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Unit converter
# ---------------------------------------------------------------------------

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
    "lb": 453.592,
    "lbs": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "ton": 1_000_000.0,
    "tonne": 1_000_000.0,
    "tonnes": 1_000_000.0,
    "st": 6350.29,
    "stone": 6350.29,
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
    "gal": 3.78541,
    "gallon": 3.78541,
    "gallons": 3.78541,
    "qt": 0.946353,
    "quart": 0.946353,
    "quarts": 0.946353,
    "pt": 0.473176,
    "pint": 0.473176,
    "pints": 0.473176,
    "cup": 0.236588,
    "cups": 0.236588,
    "fl_oz": 0.0295735,
    "fluid_ounce": 0.0295735,
    "tbsp": 0.0147868,
    "tablespoon": 0.0147868,
    "tsp": 0.00492892,
    "teaspoon": 0.00492892,
}

_SPEED: dict[str, float] = {
    "m/s": 1.0,
    "mps": 1.0,
    "km/h": 0.277778,
    "kmh": 0.277778,
    "kph": 0.277778,
    "mph": 0.44704,
    "knot": 0.514444,
    "knots": 0.514444,
    "kn": 0.514444,
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
    "acre": 4046.86,
    "acres": 4046.86,
    "ft2": 0.092903,
    "sq_ft": 0.092903,
    "square_foot": 0.092903,
    "mi2": 2_589_988.0,
    "sq_mi": 2_589_988.0,
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


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units.

    Supports length, mass, volume, speed, area, time, and temperature.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit, e.g. "km", "lbs", "celsius", "gallons".
        to_unit: Target unit, e.g. "miles", "kg", "fahrenheit", "liters".
    """
    from_u = from_unit.lower().strip()
    to_u = to_unit.lower().strip()

    # Temperature (special case)
    temp = _convert_temperature(value, from_u, to_u)
    if temp is not None:
        return f"{value} {from_unit} = {temp:.4g} {to_unit}"

    # Ratio-based categories
    for cat in _CATEGORIES:
        if from_u in cat and to_u in cat:
            result = value * cat[from_u] / cat[to_u]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"

    return f"Cannot convert from {from_unit!r} to {to_unit!r}. Units must be in the same category."
