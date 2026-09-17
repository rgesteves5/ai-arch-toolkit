"""Dynamic output-bound probe for local/pure tools + negative-cap bypass (no network)."""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock


def _blocked(*a, **k):
    raise OSError("network blocked")


socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.socket.connect = _blocked

from ai_arch_toolkit.toolkit.tools import (  # noqa: E402
    base64_encode, csv_read, json_extract, math_eval, regex_search, wikipedia_article,
)
from ai_arch_toolkit.toolkit.tools.dangerous import (  # noqa: E402
    http_get, list_directory, read_file, scrape_text, search_files,
)

box = Path(sys.argv[1]) / "bounds"
(box / "many").mkdir(parents=True, exist_ok=True)
(box / "long_line.txt").write_text("needle " + "x" * 5_000_000)  # ONE line, 5 MB
(box / "wide.csv").write_text("a,b\n" + "y" * 2_000_000 + ",1\n")
(box / "many_lines.txt").write_text("\n".join(f"line {i}" for i in range(100_000)))
for i in range(3000):
    (box / "many" / f"f{i:05d}.txt").touch()
os.chdir(box)


def fake(body: bytes):
    resp = MagicMock()
    resp.read.return_value = body
    resp.headers.get_content_charset.return_value = "utf-8"
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def show(label: str, fn, *a, **k) -> None:
    try:
        out = fn(*a, **k)
        print(f"{label:58s} -> {len(out):>10,d} chars")
    except Exception as e:  # noqa: BLE001
        print(f"{label:58s} -> RAISED {type(e).__name__}: {str(e)[:60]}")


show("read_file(long_line.txt)  [default max_lines=200]", read_file, "long_line.txt")
show("read_file(many_lines.txt, max_lines=-1)", read_file, "many_lines.txt", -1)
show("read_file(many_lines.txt, max_lines=10**9)", read_file, "many_lines.txt", 10**9)
show("search_files('.', 'needle')  [default max_results=50]", search_files, ".", "needle")
show("search_files('.', 'line', max_results=10**9)", search_files, ".", "line", 10**9)
show("csv_read(wide.csv)  [default max_rows=100]", csv_read, "wide.csv")
show("csv_read(many_lines.txt, max_rows=10**9)", csv_read, "many_lines.txt", 10**9)
show("list_directory('many')  [3000 entries, no cap param]", list_directory, "many")
show("list_directory('.', '**/*')  [recursive glob]", list_directory, ".", "**/*")
show("regex_search('a'*100_000, '')  [100 KB in]", regex_search, "a" * 100_000, "")
show("json_extract(200 KB json, 'a')", json_extract, json.dumps({"a": ["z" * 100] * 2000}), "a")
show("base64_encode('a'*1_000_000)", base64_encode, "a" * 1_000_000)
show("math_eval('10**4000')", math_eval, "10**4000")
show("math_eval('10**5000')", math_eval, "10**5000")

big = ("<p>" + "w" * 3_000_000 + "</p>").encode()
urllib.request.urlopen = lambda *a, **k: fake(big)
show("http_get(url)  [3 MB body, default max_chars=8000]", http_get, "https://example.com/")
show("http_get(url, max_chars=-1)", http_get, "https://example.com/", -1)
show("scrape_text(url, max_chars=10**9)", scrape_text, "https://example.com/", 10**9)
page = json.dumps({"query": {"pages": {"1": {"title": "T", "extract": "e" * 2_000_000}}}}).encode()
urllib.request.urlopen = lambda *a, **k: fake(page)
show("wikipedia_article('T')  [2 MB extract, default 4000]", wikipedia_article, "T")
show("wikipedia_article('T', max_chars=-1)", wikipedia_article, "T", -1)
show("wikipedia_article('T', max_chars=10**9)", wikipedia_article, "T", 10**9)
