"""Child process: exercise ONE tool with hostile-but-valid-typed args. No network, read-only.

Usage: harness_child.py <tool_name> <sandbox_cwd>
Prints one JSON object per call (flushed), so a hard kill still leaves partial results.

Modes
  offline   urllib.request.urlopen mocked -> raises URLError("offline")            (the spec)
  realopen  REAL urlopen, socket layer blocked -> exercises urllib/http.client pre-I/O checks
  badbody   urlopen mocked -> returns hostile bodies / HTTPError (benign args only)
"""

from __future__ import annotations

import email.message
import importlib
import io
import json
import os
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

TOOL = sys.argv[1]
SANDBOX = sys.argv[2]
BUDGET_S = 8.0
T0 = time.monotonic()

# ---- hard network block (safety net for every mode, incl. third-party HTTP stacks) ----------


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise OSError("network blocked by harness")


socket.create_connection = _blocked  # type: ignore[assignment]
socket.getaddrinfo = _blocked  # type: ignore[assignment]
socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.socket.connect_ex = _blocked  # type: ignore[method-assign]

REAL_URLOPEN = urllib.request.urlopen
REQUESTS: list[dict[str, Any]] = []
SLEEPS: list[float] = []


def _fake_sleep(seconds: float) -> None:  # tools throttle with time.sleep; record, never wait
    SLEEPS.append(round(float(seconds), 2))


time.sleep = _fake_sleep  # type: ignore[assignment]


def _record(req: Any, timeout: Any) -> str:
    if isinstance(req, urllib.request.Request):
        url, headers, method = req.full_url, dict(req.header_items()), req.get_method()
    else:
        url, headers, method = str(req), {}, "GET"
    REQUESTS.append({"url": url[:400], "headers": sorted(headers), "method": method, "timeout": timeout})
    return url


def urlopen_offline(req: Any, *a: Any, timeout: Any = None, **k: Any) -> Any:
    _record(req, timeout)
    raise urllib.error.URLError("offline")


def urlopen_real(req: Any, *a: Any, timeout: Any = None, **k: Any) -> Any:
    _record(req, timeout)
    return REAL_URLOPEN(req, *a, timeout=timeout, **k)


class FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.status = 200
        self.headers = email.message.Message()
        self.headers["Content-Type"] = "application/json; charset=utf-8"

    def read(self, n: int = -1) -> bytes:
        return self._body if n is None or n < 0 else self._body[:n]

    def getheader(self, name: str, default: Any = None) -> Any:
        return self.headers.get(name, default)

    def info(self) -> Any:
        return self.headers

    def geturl(self) -> str:
        return "https://example.invalid/"

    def __enter__(self) -> FakeResp:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def make_body_urlopen(body: bytes | int) -> Any:
    def _open(req: Any, *a: Any, timeout: Any = None, **k: Any) -> Any:
        url = _record(req, timeout)
        if isinstance(body, int):
            raise urllib.error.HTTPError(
                url, body, "Hostile", email.message.Message(), io.BytesIO(b"<html>" + b"x" * 5000)
            )
        return FakeResp(body)

    return _open


# ---- argument generation ---------------------------------------------------------------------

LONG = "A" * 100_000
STR_HOSTILE = {
    "empty": "",
    "long100k": LONG,
    "dotdot": "../..",
    "nul_ctrl": "a\x00b\n\r\t c",
    "url": "https://evil.example/api.php?x=1#",
    "hostinj": "evil.example/#",
    "space_q": "a b?c=d&e#f/../g",
}
INT_HOSTILE = {"zero": 0, "neg1": -1, "neg1e9": -(10**9), "pos1e9": 10**9, "pos1e30": 10**30}
NUM_HOSTILE = {"zero": 0.0, "neg1e9": -1e9, "pos1e9": 1e9, "pos1e308": 1e308, "neg0": -0.0}

BENIGN_BY_NAME: dict[str, Any] = {
    "city": "Lisbon", "name": "Portugal", "title": "Python", "term": "test", "word": "test",
    "date_str": "2024-01-15", "from_date": "2024-01-01", "to_date": "2024-01-31",
    "start_date": "2024-01-01", "end_date": "2024-01-31", "pub_start_date": "2024-01-01",
    "pub_end_date": "2024-01-31", "start_time": "2024-01-01", "end_time": "2024-01-31",
    "time_str": "12:30", "tz": "Europe/Lisbon", "from_tz": "Europe/Lisbon", "to_tz": "Asia/Tokyo",
    "expression": "1+1", "text": "abc abc", "encoded": "YWJj", "json_string": '{"a": [1, 2]}',
    "format_out": "%d/%m/%Y", "unit": "km", "from_unit": "km", "to_unit": "mi",
    "doi": "10.1000/xyz123", "pmid": "12345678", "arxiv_id": "2301.00001",
    "cve_id": "CVE-2021-44228", "isbn": "9780140328721", "video_url_or_id": "dQw4w9WgXcQ",
    "accession": "P69905", "pdb_id": "4HHB", "chembl_id": "CHEMBL25", "nct_id": "NCT04280705",
    "rxcui": "161", "qid": "Q42", "indicator": "NY.GDP.MKTP.CD", "indicator_code": "WHOSIS_000001",
    "country": "PT", "countries": "PT;ES", "language": "en", "ip": "8.8.8.8",
    "dataset_id": "nama_10_gdp", "geo_codes": "PT,ES", "barcode": "3017620422003",
    "barcodes": "3017620422003,5449000000996", "work_id": "OL45883W", "ror_id": "https://ror.org/05a28rw58",
    "setid": "1efe378e-fee1-4ae9-a4a2-9b8d25ad1d35", "taxon_key": "2435099", "event_id": "us7000abcd",
    "identifier": "12345678", "source": "MED", "paper_id": "10.1000/xyz123", "term_id": "FOODON_00001002",
    "component_id": "ATP", "recall_number": "F-0283-2017", "tag_key": "amenity", "tag_value": "cafe",
    "bbox": "38.70,-9.20,38.75,-9.10", "languages": "en", "year": "2020", "start_year": "2010",
    "end_year": "2020", "from_year": "2010", "to_year": "2020", "api_url": "https://en.wiktionary.org/w/api.php",
    "molecule_chembl_id": "CHEMBL25", "target_chembl_id": "CHEMBL204", "ndc": "0002-3227-30",
    "drug_name": "aspirin", "url": "https://example.com/", "command": "true", "code": "1+1",
    "lat": 38.7, "lon": -9.1, "lat1": 38.7, "lon1": -9.1, "lat2": 41.1, "lon2": -8.6,
    "latitude": 38.7, "longitude": -9.1, "value": 1.0,
}
BENIGN_BY_TOOL: dict[tuple[str, str], Any] = {
    ("csv_read", "path"): "data.csv", ("read_file", "path"): "notes.txt",
    ("list_directory", "path"): ".", ("list_directory", "pattern"): "*",
    ("search_files", "directory"): ".", ("search_files", "pattern"): "needle",
    ("json_extract", "path"): "a[0]", ("regex_search", "pattern"): "a",
    ("date_diff", "start"): "2024-01-01", ("date_diff", "end"): "2024-01-31",
    ("date_diff", "unit"): "days", ("rxnorm_drug_search", "name"): "aspirin",
    ("gbif_species_match", "name"): "Puma concolor", ("overpass_query", "query"): "[out:json];node(1);out;",
    ("wikidata_sparql", "query"): "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
    ("weather_units", "unit"): "celsius", ("distance_between", "unit"): "km",
}
TARGETED: dict[str, list[str]] = {
    "pattern": ["(", "[", "", "/etc/*", "**", "***", "../*", "**/../**", "(?P<x>"],
    "expression": ["1/0", "(" * 300, "-" * 3000 + "1", "10**400*1.5", "1e308*10", "2**-1e9", "a" * 10],
    "format_out": ["%", "%Q", "%" * 1000, "%9999999999d"],
    "json_string": ["[" * 100_000, '{"a":' * 3000 + "1" + "}" * 3000, "NaN", "1e999"],
    "path": ["a" + "[0]" * 2000, "[" * 50, "[-1]", "[999999999999999999999]", "sub", "/dev/null", "."],
    "tz": ["/etc/passwd", "../../etc/passwd", "A" * 300, "Europe/../Europe/Lisbon"],
    "from_tz": ["/etc/passwd", "../../etc/passwd", "A" * 300],
    "time_str": ["24:61", "9999-99-99 99:99", "0001-01-01 00:00", "9999-12-31 23:59"],
    "date_str": ["0001-01-01", "9999-12-31", "9999-12-31 23:59"],
}
LAST_RESORT_SLOW: dict[str, list[dict[str, Any]]] = {  # run last: may hang until the hard kill
    "regex_search": [{"text": "a" * 40 + "!", "pattern": "(a+)+$"}],
    "math_eval": [{"expression": "9**9**9"}],
}


def benign(tool: str, param: str, spec: dict[str, Any]) -> Any:
    if (tool, param) in BENIGN_BY_TOOL:
        return BENIGN_BY_TOOL[(tool, param)]
    kind = spec.get("type")
    if param in BENIGN_BY_NAME:
        value = BENIGN_BY_NAME[param]
        if (kind == "string") == isinstance(value, str):
            return value
    if "default" in spec:
        return spec["default"]
    return {"string": "test", "integer": 1, "number": 1.0, "boolean": False}.get(kind, "test")


def hostile_values(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec.get("type")
    if kind == "string":
        return STR_HOSTILE
    if kind == "integer":
        return INT_HOSTILE
    if kind == "number":
        return NUM_HOSTILE
    if kind == "boolean":
        return {"true": True, "false": False}
    return {}


def abbreviate(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 60:
            out[k] = f"{v[:24]!s}...<len {len(v)}>"
        else:
            out[k] = v
    return out


def call(fn: Any, mode: str, label: str, args: dict[str, Any]) -> None:
    REQUESTS.clear()
    SLEEPS.clear()
    start = time.monotonic()
    rec: dict[str, Any] = {"tool": TOOL, "mode": mode, "label": label, "args": abbreviate(args)}
    print(json.dumps({**rec, "event": "start"}), flush=True)
    try:
        value = fn(**args)
    except BaseException as exc:  # noqa: BLE001 - we want everything, incl. RecursionError
        tb = traceback.extract_tb(exc.__traceback__)
        frame = next((f for f in reversed(tb) if "toolkit/tools" in f.filename), tb[-1])
        rec.update(
            outcome="RAISED",
            exc_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            exc_msg=str(exc)[:160],
            where=f"{os.path.basename(frame.filename)}:{frame.lineno}",
        )
    else:
        if isinstance(value, str):
            rec.update(outcome="str", out_len=len(value), head=value[:100])
        else:
            rec.update(outcome="NON_STR", out_type=type(value).__name__)
    rec["dt"] = round(time.monotonic() - start, 3)
    rec["requests"] = list(REQUESTS)
    rec["sleeps"] = list(SLEEPS)
    rec["event"] = "end"
    print(json.dumps(rec), flush=True)


def main() -> None:
    os.chdir(SANDBOX)
    fn = None
    for ns in ("ai_arch_toolkit.toolkit.tools", "ai_arch_toolkit.toolkit.tools.dangerous"):
        mod = importlib.import_module(ns)
        if TOOL in mod.__all__:
            fn = getattr(mod, TOOL)
    assert fn is not None, TOOL
    schema = fn.__tool_definition__.schema.input_schema
    props: dict[str, dict[str, Any]] = schema.get("properties", {})
    base = {p: benign(TOOL, p, s) for p, s in props.items()}

    plans: list[tuple[str, dict[str, Any]]] = [("benign", dict(base))]
    # uniform profiles: every param of a type gets the same hostile value
    for label in ("empty", "long100k", "dotdot", "nul_ctrl", "url"):
        args = {}
        for p, s in props.items():
            kind = s.get("type")
            if kind == "string":
                args[p] = STR_HOSTILE[label]
            elif kind == "integer":
                args[p] = {"empty": 0, "long100k": 10**9, "dotdot": -1, "nul_ctrl": 10**30, "url": -(10**9)}[label]
            elif kind == "number":
                args[p] = {"empty": 0.0, "long100k": 1e9, "dotdot": -1.0, "nul_ctrl": 1e308, "url": -1e9}[label]
            else:
                args[p] = label in {"long100k", "nul_ctrl"}
        plans.append((f"uniform:{label}", args))
    # one hostile param at a time, everything else benign
    for p, s in props.items():
        for label, value in hostile_values(s).items():
            plans.append((f"one:{p}={label}", {**base, p: value}))
        for i, value in enumerate(TARGETED.get(p, [])):
            plans.append((f"targeted:{p}#{i}", {**base, p: value}))

    for mode, opener in (("offline", urlopen_offline), ("realopen", urlopen_real)):
        urllib.request.urlopen = opener  # tools call urllib.request.urlopen via the module attr
        for label, args in plans:
            if time.monotonic() - T0 > BUDGET_S:
                print(json.dumps({"tool": TOOL, "event": "budget_exhausted", "mode": mode}), flush=True)
                break
            call(fn, mode, label, args)

    bodies: list[bytes | int] = [
        b"", b"not json", b"null", b"[]", b"{}", b'"str"', b"123", b"<html><body>x</body></html>",
        b'{"results": null, "items": null, "data": null, "query": null, "response": null}',
        b"[null]", b"[[]]", b'[{"a": null}]', b"\xff\xfe\x00", 500, 404, 429,
    ]
    for body in bodies:
        if time.monotonic() - T0 > BUDGET_S:
            break
        urllib.request.urlopen = make_body_urlopen(body)
        call(fn, "badbody", f"body={body!r}"[:40], dict(base))

    urllib.request.urlopen = urlopen_offline
    for args in LAST_RESORT_SLOW.get(TOOL, []):
        call(fn, "offline", "slow", {**base, **args})
    print(json.dumps({"tool": TOOL, "event": "done", "elapsed": round(time.monotonic() - T0, 2)}), flush=True)


if __name__ == "__main__":
    main()
