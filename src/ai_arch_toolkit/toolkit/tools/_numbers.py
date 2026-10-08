"""Numbers as the tools write them: every digit the source sent, never in scientific notation
(the tools contract, output point 5)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


def plain_number(value: float | str) -> str:
    """``value`` in plain decimal digits, as precise as the source sent it.

    A float is written from its shortest exact text (``repr``): 2.91849e+13 reads
    29184900000000, 1e-05 reads 0.00001, and a whole float loses its ``.0``. A decimal string
    (``"+1.750"``, ``"1.0E7"``) loses its ``+`` and its exponent, and keeps the digits it gave.
    Anything that is no finite number comes back as it was.
    """
    if isinstance(value, int):  # bool included: JSON true is no measure, and reads as itself
        return str(value)
    try:
        number = Decimal(repr(value) if isinstance(value, float) else value.strip())
    except InvalidOperation:
        return str(value)
    if not number.is_finite():
        return str(value)
    text = format(number, "f")
    return text.removesuffix(".0") if isinstance(value, float) else text
