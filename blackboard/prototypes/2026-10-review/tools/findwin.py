import time, random
from ai_arch_toolkit.toolkit.tools._window import find_window
random.seed(1)
words = "the quick brown fox jumps over a lazy dog while seven eager experts never rest".split()
def text(n):
    out, size = [], 0
    while size < n:
        line = " ".join(random.choice(words) for _ in range(12))
        out.append(line); size += len(line) + 1
    return "\n".join(out)[:n]
for n, limit, needle in [(1_000_000, 6_000, "e"), (1_000_000, 20_000, "e"), (1_000_000, 100_000, "e"), (2_000_000, 100_000, "e")]:
    t = text(n)
    s = time.perf_counter()
    w = find_window(t, needle, limit=limit)
    print(f"N={n:,} limit={limit:,} needle={needle!r} matches={w.total:,} -> {time.perf_counter()-s:.2f}s")
