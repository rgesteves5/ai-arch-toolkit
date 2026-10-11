import time, tracemalloc
from ai_arch_toolkit.toolkit.tools._wiki_html import wiki_text

def nested(levels, rowspan, text):
    inner = f"<table><tr><td rowspan={rowspan}>{text}</td></tr>" + "<tr></tr>" * (rowspan - 1) + "</table>"
    html = inner
    for _ in range(levels - 1):
        html = f"<table><tr><td rowspan={rowspan}>{html}</td></tr>" + "<tr></tr>" * (rowspan - 1) + "</table>"
    return html

for levels, rowspan, size in [(1, 500, 1000), (2, 100, 100), (2, 500, 100), (3, 50, 100)]:
    html = nested(levels, rowspan, "x" * size)
    tracemalloc.start()
    t = time.perf_counter()
    out = wiki_text(html)
    dt = time.perf_counter() - t
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"levels={levels} rowspan={rowspan} cell={size}B input={len(html):,}B -> text={len(out.text):,} chars "
          f"(x{len(out.text)//len(html)}) peak={peak/1e6:.0f}MB in {dt:.2f}s")
