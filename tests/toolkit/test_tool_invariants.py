"""Executable invariants every toolkit tool keeps.

The tools are discovered from the modules of ``toolkit.tools`` (``pkgutil`` and
``__tool_definition__``), so a new tool is held to these rules without being listed anywhere:

1. it is exported by exactly one namespace, ``toolkit.tools`` or ``toolkit.tools.dangerous``, under
   its schema name;
2. its declared ``capability`` is what its code reaches (AST call graph), and nothing that reaches
   files, a shell or a Python evaluator lives outside ``dangerous``;
3. hostile arguments never move a request off its module's hosts and base paths;
4. it never raises, whatever the arguments or the response body;
5. through the governed executor, its output stays within ``max_output_chars``.

Sockets are blocked and the throttle never sleeps (``conftest.py``).
"""

from __future__ import annotations

import ast
import email.message
import importlib
import inspect
import io
import json
import pkgutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

import pytest

from ai_arch_toolkit.core import ApprovalDecision, ToolCall, ToolGroup
from ai_arch_toolkit.toolkit.tools import _http
from ai_arch_toolkit.toolkit.tools._http import Api
from tests.toolkit.http_fakes import respond

PACKAGE = "ai_arch_toolkit.toolkit.tools"
SAFE = importlib.import_module(PACKAGE)
DANGEROUS = importlib.import_module(f"{PACKAGE}.dangerous")

# run_command runs its argument as a shell command: hostile arguments would run on this machine.
NOT_CALLED = frozenset({"run_command"})


def _tools() -> dict[str, Callable[..., Any]]:
    found: dict[str, Callable[..., Any]] = {}
    for info in pkgutil.iter_modules(SAFE.__path__):
        if not info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{PACKAGE}.{info.name}")
        for value in vars(module).values():
            if (
                inspect.isfunction(value)
                and value.__module__ == module.__name__
                and "__tool_definition__" in vars(value)
            ):
                found[value.__name__] = value
    return found


TOOLS = _tools()


def _capability(name: str) -> str | None:
    return TOOLS[name].__tool_definition__.policy.capability


NETWORK = sorted(name for name in TOOLS if _capability(name) == "network")
CALLED = sorted(set(TOOLS) - NOT_CALLED)


# --- 1. Exports --------------------------------------------------------------------------------


def test_every_tool_is_exported_by_exactly_one_namespace_under_its_schema_name() -> None:
    safe, dangerous = set(SAFE.__all__), set(DANGEROUS.__all__)

    assert not safe & dangerous
    assert set(TOOLS) == safe | dangerous
    for name, fn in TOOLS.items():
        assert vars(SAFE if name in safe else DANGEROUS)[name] is fn
        assert fn.__tool_definition__.schema.name == name


# --- 2. Capabilities ---------------------------------------------------------------------------

_NETWORK_NAMES = frozenset({f"{PACKAGE}._http.Api", f"{PACKAGE}._http.fetch_page"})
_NETWORK_LIBRARIES = frozenset({"youtube_transcript_api", "requests"})
_FILESYSTEM_ROOTS = frozenset({"pathlib", "os", "shutil", "glob", "tempfile"})
_LOCAL_ONLY = frozenset({"filesystem", "shell", "python"})


def _imports(nodes: list[ast.stmt]) -> dict[str, str]:
    """Local name -> dotted origin, for the import statements among ``nodes``."""
    names: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition(".")[0]
                names[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _reached(module: ModuleType, root: str) -> set[str]:
    """Every import origin the function ``root`` reaches through the module's own definitions."""
    tree = ast.parse(inspect.getsource(module))
    imports = _imports(tree.body)
    definitions: dict[str, ast.AST] = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            definitions.update(
                (target.id, node.value) for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            definitions[node.target.id] = node.value
    origins: set[str] = set()
    seen: set[str] = set()
    todo = [definitions[root]]
    while todo:
        for node in ast.walk(todo.pop()):
            if isinstance(node, ast.Import | ast.ImportFrom):
                origins.update(_imports([node]).values())
            if not isinstance(node, ast.Name) or node.id in seen:
                continue
            seen.add(node.id)
            if node.id in imports:
                origins.add(imports[node.id])
            elif node.id in definitions:
                todo.append(definitions[node.id])
            elif node.id == "open":
                origins.add("builtins.open")
    return origins


def _reached_capability(origins: set[str]) -> str:
    roots = {origin.partition(".")[0] for origin in origins}
    if "subprocess" in roots:
        return "shell"
    if roots & _FILESYSTEM_ROOTS or "builtins.open" in origins:
        return "filesystem"
    if origins & _NETWORK_NAMES or roots & _NETWORK_LIBRARIES:
        return "network"
    return "compute"


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_the_declared_capability_is_what_the_code_reaches(name: str) -> None:
    fn = TOOLS[name]
    reached = _reached_capability(_reached(sys.modules[fn.__module__], name))
    declared = _capability(name)

    # python_repl evaluates code in a sandbox: it reaches nothing, and declares what it runs.
    assert declared == reached or (declared, reached) == ("python", "compute")
    if declared in _LOCAL_ONLY:
        assert name in DANGEROUS.__all__


def test_the_capability_detector_sees_through_helpers_and_constants() -> None:
    assert _reached_capability(_reached(sys.modules[f"{PACKAGE}._json"], "csv_read")) == (
        "filesystem"
    )
    assert _reached_capability(_reached(sys.modules[f"{PACKAGE}._shell"], "run_command")) == (
        "shell"
    )
    assert _reached_capability({f"{PACKAGE}._http.Api", "json"}) == "network"
    assert _reached_capability({"math", "re"}) == "compute"


# --- Hostile arguments --------------------------------------------------------------------------

_HOSTILE: dict[str, dict[str, Any]] = {
    "string": {
        "empty": "",
        "long": "A" * 100_000,
        "dotdot": "../..",
        "control": "a\x00b\n\r\t c",
        "url": "https://evil.example/api.php?x=1#",
        "host": "evil.example/#",
        "query": "a b?c=d&e#f/../g",
    },
    "integer": {"zero": 0, "minus": -1, "minus_big": -(10**9), "big": 10**9, "huge": 10**30},
    "number": {"zero": 0.0, "minus_big": -1e9, "big": 1e9, "max": 1e308, "minus_zero": -0.0},
    "boolean": {"true": True, "false": False},
}
_UNIFORM = ("empty", "long", "dotdot", "control", "url")

# Valid values, so a tool gets past its own checks and reaches the network or the file system.
_BENIGN_BY_NAME: dict[str, Any] = {
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
    "rxcui": "161", "qid": "Q42", "indicator": "NY.GDP.MKTP.CD",
    "indicator_code": "WHOSIS_000001", "country": "PT", "countries": "PT;ES", "language": "en",
    "ip": "8.8.8.8", "dataset_id": "nama_10_gdp", "geo_codes": "PT,ES",
    "barcode": "3017620422003", "barcodes": "3017620422003,5449000000996",
    "work_id": "OL45883W", "ror_id": "https://ror.org/05a28rw58",
    "setid": "1efe378e-fee1-4ae9-a4a2-9b8d25ad1d35", "taxon_key": "2435099",
    "event_id": "us7000abcd", "identifier": "12345678", "source": "MED",
    "paper_id": "10.1000/xyz123", "term_id": "FOODON_00001002", "component_id": "ATP",
    "recall_number": "F-0283-2017", "tag_key": "amenity", "tag_value": "cafe",
    "bbox": "38.70,-9.20,38.75,-9.10", "languages": "en", "year": "2020",
    "start_year": "2010", "end_year": "2020", "from_year": "2010", "to_year": "2020",
    "api_url": "https://en.wiktionary.org/w/api.php", "molecule_chembl_id": "CHEMBL25",
    "target_chembl_id": "CHEMBL204", "ndc": "0002-3227-30", "drug_name": "aspirin",
    "url": "https://example.com/", "code": "1+1", "lat": 38.7, "lon": -9.1, "lat1": 38.7,
    "lon1": -9.1, "lat2": 41.1, "lon2": -8.6, "latitude": 38.7, "longitude": -9.1, "value": 1.0,
}  # fmt: skip
_BENIGN_BY_TOOL: dict[tuple[str, str], Any] = {
    ("csv_read", "path"): "data.csv",
    ("read_file", "path"): "notes.txt",
    ("list_directory", "path"): ".",
    ("list_directory", "pattern"): "*",
    ("search_files", "directory"): ".",
    ("search_files", "pattern"): "needle",
    ("json_extract", "path"): "a[0]",
    ("regex_search", "pattern"): "a",
    ("date_diff", "start"): "2024-01-01",
    ("date_diff", "end"): "2024-01-31",
    ("date_diff", "unit"): "days",
    ("rxnorm_drug_search", "name"): "aspirin",
    ("gbif_species_match", "name"): "Puma concolor",
    ("overpass_query", "query"): "[out:json];node(1);out;",
    ("wikidata_sparql", "query"): "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1",
    ("weather_units", "unit"): "celsius",
    ("distance_between", "unit"): "km",
}
_TARGETED: dict[str, list[Any]] = {
    "pattern": ["(", "[", "", "/etc/*", "**", "***", "../*", "**/../**", "(?P<x>", "(a+)+$"],
    "expression": ["1/0", "(" * 300, "-" * 3000 + "1", "10**400*1.5", "1e308*10", "2**-1e9"],
    "format_out": ["%", "%Q", "%" * 1000, "%9999999999d"],
    "json_string": ["[" * 100_000, '{"a":' * 3000 + "1" + "}" * 3000, "NaN", "1e999"],
    "path": ["a" + "[0]" * 2000, "[" * 50, "[-1]", "[999999999999999999999]", "/dev/null", "."],
    "tz": ["/etc/passwd", "../../etc/passwd", "A" * 300, "Europe/../Europe/Lisbon"],
    "from_tz": ["/etc/passwd", "../../etc/passwd", "A" * 300],
    "time_str": ["24:61", "9999-99-99 99:99", "0001-01-01 00:00", "9999-12-31 23:59"],
    "date_str": ["0001-01-01", "9999-12-31", "9999-12-31 23:59"],
}


def _properties(name: str) -> dict[str, dict[str, Any]]:
    return TOOLS[name].__tool_definition__.schema.input_schema.get("properties", {})


def _benign(name: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for param, spec in _properties(name).items():
        kind = spec.get("type")
        value = _BENIGN_BY_NAME.get(param)
        if (name, param) in _BENIGN_BY_TOOL:
            args[param] = _BENIGN_BY_TOOL[(name, param)]
        elif value is not None and (kind == "string") == isinstance(value, str):
            args[param] = value
        elif "default" in spec:
            args[param] = spec["default"]
        else:
            args[param] = {"integer": 1, "number": 1.0, "boolean": False}.get(kind, "test")
    return args


def _uniform(name: str, label: str) -> dict[str, Any]:
    index = _UNIFORM.index(label)
    args: dict[str, Any] = {}
    for param, spec in _properties(name).items():
        values = list(_HOSTILE.get(spec.get("type", ""), {}).values())
        args[param] = values[index % len(values)] if values else _benign(name)[param]
    return args


def _plans(name: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """The benign call, every argument hostile at once, then one hostile argument at a time."""
    base = _benign(name)
    yield "benign", base
    for label in _UNIFORM:
        yield f"uniform:{label}", _uniform(name, label)
    for param, spec in _properties(name).items():
        for label, value in _HOSTILE.get(spec.get("type", ""), {}).items():
            yield f"{param}={label}", {**base, param: value}
        for index, value in enumerate(_TARGETED.get(param, [])):
            yield f"{param}#{index}", {**base, param: value}


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small working directory, three levels deep, so "../.." stays inside ``tmp_path``."""
    cwd = tmp_path / "a" / "b" / "c"
    cwd.mkdir(parents=True)
    (cwd / "notes.txt").write_text("needle in a file\n")
    (cwd / "data.csv").write_text("a,b\n1,2\n")
    monkeypatch.chdir(cwd)
    return cwd


def _offline(sent: list[str]) -> Callable[[urllib.request.Request, float], Any]:
    """An ``_http._open`` that records each request and fails like a dead network."""

    def open_(request: urllib.request.Request, timeout: float) -> Any:
        sent.append(request.full_url)
        raise urllib.error.URLError("offline")

    return open_


# --- 3. Fixed hosts ----------------------------------------------------------------------------


def _allowed(module: ModuleType, url: str) -> bool:
    parts = urlsplit(url)
    apis = [value for value in vars(module).values() if isinstance(value, Api)]
    domains: frozenset[str] = vars(module).get("_DOMAINS", frozenset())
    on_an_api = any(
        parts.hostname == api.host and parts.path.startswith(urlsplit(api.base).path)
        for api in apis
    )
    host = parts.hostname or ""
    in_domains = any(host == domain or host.endswith(f".{domain}") for domain in domains)
    return parts.scheme == "https" and (on_an_api or in_domains)


@pytest.mark.parametrize("name", sorted(set(NETWORK) & set(SAFE.__all__)))
def test_hostile_arguments_never_move_a_request_off_the_modules_apis(
    name: str, monkeypatch: pytest.MonkeyPatch, sandbox: Path
) -> None:
    sent: list[str] = []
    monkeypatch.setattr(_http, "_open", _offline(sent))
    module = sys.modules[TOOLS[name].__module__]

    for _label, args in _plans(name):
        TOOLS[name](**args)

    for url in sent:
        segments = urlsplit(url).path.split("/")
        assert _allowed(module, url), url
        assert "." not in segments and ".." not in segments, url


# --- 4. Never raises ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", CALLED)
def test_hostile_arguments_never_raise(
    name: str, monkeypatch: pytest.MonkeyPatch, sandbox: Path
) -> None:
    # First with the network failing inside _http, then with the real opener and blocked sockets.
    for opener in (_offline([]), _http._open):
        monkeypatch.setattr(_http, "_open", opener)
        for label, args in _plans(name):
            result = TOOLS[name](**args)
            assert isinstance(result, str), (label, result)


_BODIES: list[bytes | int] = [
    b"",
    b"not json",
    b"null",
    b"[]",
    b"{}",
    b'"str"',
    b"123",
    b"<html><body>x</body></html>",
    b'{"results": null, "items": null, "data": null, "query": null, "response": null}',
    b"[null]",
    b"[[]]",
    b'[{"a": null}]',
    b"\xff\xfe\x00",
    500,
    404,
    429,
]


def _answering(body: bytes | int) -> Callable[[urllib.request.Request, float], Any]:
    def open_(request: urllib.request.Request, timeout: float) -> Any:
        if isinstance(body, int):
            error_body = io.BytesIO(b"<html>" + b"x" * 5000)
            raise urllib.error.HTTPError(
                request.full_url, body, "Hostile", email.message.Message(), error_body
            )
        return respond(body, content_type="application/json; charset=utf-8")

    return open_


@pytest.mark.parametrize("name", NETWORK)
def test_hostile_response_bodies_never_raise(
    name: str, monkeypatch: pytest.MonkeyPatch, sandbox: Path
) -> None:
    for body in _BODIES:
        monkeypatch.setattr(_http, "_open", _answering(body))
        result = TOOLS[name](**_benign(name))
        assert isinstance(result, str), (body, result)


def _keys_read(module: ModuleType) -> list[str]:
    """The string keys a module reads from responses: ``x.get("key")`` and ``x["key"]``."""
    keys: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return sorted(keys)


def _shaped_bodies(keys: list[str]) -> list[bytes]:
    """Every key the module reads, holding the wrong kind of value at once, one level deep too."""
    wrong: list[object] = [None, "x", 1, [None], [["x"]], {"x": None}]
    bodies: list[object] = [{key: value for key in keys} for value in wrong]
    bodies += [{key: {inner: value for inner in keys} for key in keys} for value in wrong]
    bodies += [{key: [{inner: value for inner in keys}] for key in keys} for value in wrong]
    bodies += [[{key: value for key in keys}] for value in wrong]
    return [json.dumps(body).encode() for body in bodies]


@pytest.mark.parametrize("name", NETWORK)
def test_bodies_built_from_the_keys_a_module_reads_never_raise(
    name: str, monkeypatch: pytest.MonkeyPatch, sandbox: Path
) -> None:
    # A tool that reads a response outside its _http parse boundary crashes on one of these.
    for body in _shaped_bodies(_keys_read(sys.modules[TOOLS[name].__module__])):
        monkeypatch.setattr(_http, "_open", _answering(body))
        result = TOOLS[name](**_benign(name))
        assert isinstance(result, str), (body[:200], result)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("math_eval", {"expression": "9**9**9"}),
        ("math_eval", {"expression": "factorial(10**7)"}),
        ("math_eval", {"expression": "round(5, -10**9)"}),
        ("math_eval", {"expression": "pow(10, 10**9)"}),
        ("regex_search", {"text": "a" * 40 + "!", "pattern": "(a+)+$"}),
        ("regex_search", {"text": "a" * 40 + "!", "pattern": "(a|aa)+$"}),
    ],
)
def test_computations_that_would_hold_the_gil_are_refused_at_once(
    name: str, args: dict[str, str]
) -> None:
    # A long C call holds the GIL, so neither the executor's timeout nor pytest-timeout can stop
    # it in this process: the guard must refuse it before it starts. A child process with a hard
    # deadline keeps a missing guard from hanging the suite.
    code = f"from {PACKAGE} import {name}; print({name}(**{args!r}))"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(("Error", "Invalid regex", "Pattern refused")), (
        completed.stdout
    )


# --- 5. Bounded output -------------------------------------------------------------------------

_LIMIT = 1000
_MARKER_ROOM = 200


def _approve_all(request: object) -> ApprovalDecision:
    return ApprovalDecision.approve()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("name", CALLED)
def test_output_through_the_executor_stays_within_the_limit(
    name: str, monkeypatch: pytest.MonkeyPatch, sandbox: Path
) -> None:
    group = ToolGroup(TOOLS[name], approval_handler=_approve_all, max_output_chars=_LIMIT)
    big = b'{"x": "' + b"y" * 5_000_000 + b'"}'
    runs = [("benign, 5 MB body", _answering(big), _benign(name))]
    runs += [(label, _offline([]), args) for label, args in _plans(name)]

    for label, opener, args in runs:
        monkeypatch.setattr(_http, "_open", opener)
        result = group.execute(ToolCall(id="call", name=name, input=args))
        assert len(result.to_model_text()) <= _LIMIT + _MARKER_ROOM, label
