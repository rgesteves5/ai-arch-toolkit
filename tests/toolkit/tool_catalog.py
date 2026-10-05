"""The toolkit's tools, as the suite-wide tests see them: discovered, called with valid arguments.

The tools are found in the modules of ``toolkit.tools`` (``pkgutil`` and ``__tool_definition__``),
so a new tool is held to the invariants (``test_tool_invariants.py``) and the contract
(``test_tool_contract.py``) without being listed anywhere.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from typing import Any, get_args

from ai_arch_toolkit.core import ToolFailure, ToolFailureType, ToolResult

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


def capability(name: str) -> str | None:
    return TOOLS[name].__tool_definition__.policy.capability


NETWORK = sorted(name for name in TOOLS if capability(name) == "network")
CALLED = sorted(set(TOOLS) - NOT_CALLED)


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
    ("weather_units", "unit"): "c",
    ("who_series", "country"): "PRT",
    ("wiktionary_entry", "language"): "English",
    ("distance_between", "unit"): "km",
    ("clinical_trials_search", "query"): "asthma",
    ("earthquake_search", "max_radius_km"): 100.0,
}


def schema_properties(fn: Callable[..., Any]) -> dict[str, dict[str, Any]]:
    """The parameters of a tool's input schema, by name."""
    return fn.__tool_definition__.schema.input_schema.get("properties", {})


def properties(name: str) -> dict[str, dict[str, Any]]:
    return schema_properties(TOOLS[name])


def benign(name: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for param, spec in properties(name).items():
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


FAILURE_TYPES = frozenset(get_args(ToolFailureType.__value__))


def text_of(result: object) -> str | None:
    """What a tool answered as text: its ``str``, or the text of the successful ``ToolResult`` a
    windowed answer is (``Window.result()``); ``None`` for anything else."""
    if isinstance(result, ToolResult):
        return result.value if result.ok and isinstance(result.value, str) else None
    return result if isinstance(result, str) else None


def answer(name: str, args: dict[str, Any]) -> str | ToolFailure:
    """The tool's text, or the typed failure it raised; any other exception fails the test."""
    try:
        result = TOOLS[name](**args)
    except ToolFailure as failure:
        assert failure.error.type in FAILURE_TYPES, failure.error
        assert failure.error.message.strip(), failure.error
        return failure
    text = text_of(result)
    assert text is not None, result
    return text
