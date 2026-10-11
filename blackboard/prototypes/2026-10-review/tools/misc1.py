import math
from ai_arch_toolkit.toolkit.tools._filesystem import _human_size
for n in (1536, 1992294, 1_900_000_000, 1023):
    print(n, "->", _human_size(n))

from ai_arch_toolkit.toolkit.tools._math import math_eval, unit_convert
for expr in ("2**14500", "inf - inf"):
    try:
        print(expr, "->", math_eval(expr)[:60])
    except Exception as e:
        print(expr, "->", type(e).__name__, getattr(e, "error", e))
print(unit_convert(500, "nm", "m"))

from ai_arch_toolkit.toolkit.tools._python import python_repl
for code in ("re.search('(', 'x')", "'x'.encode('nope')"):
    try:
        print(code, "->", python_repl(code))
    except Exception as e:
        print(code, "->", type(e).__name__, str(e)[:80])

from ai_arch_toolkit.toolkit.tools._geo import distance_between
bad = 0
import itertools
for lat in [x / 7 for x in range(-630, 631)]:
    for lon in [0.0, 13.7, 45.3, 77.1, 123.4]:
        lon2 = lon - 180 if lon > 0 else lon + 180
        try:
            distance_between(lat, lon, -lat, lon2)
        except ValueError as e:
            bad += 1
            if bad == 1:
                print("antipodal", (lat, lon, -lat, lon2), "->", type(e).__name__, e)
print("antipodal failures:", bad)

from ai_arch_toolkit.toolkit.tools._json import json_extract
print(json_extract('{"a": null, "b": true, "c": 1e-05}', "a"), json_extract('{"b": true}', "b"), json_extract('{"c": 1e-05}', "c"))

from ai_arch_toolkit.toolkit.tools._datetime import date_format
try:
    print(repr(date_format("2024-01-01", "a\x00b")))
except Exception as e:
    print("date_format NUL ->", type(e).__name__, e)
