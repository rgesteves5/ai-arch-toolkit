"""Executable ownership boundaries, with synthetic violations as canaries."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/ai_arch_toolkit/core"


def imported_modules(source: str, package: str) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            modules.append(resolve_name(name, package) if node.level else name)
    return modules


def forbidden_toolkit_imports(source: str, package: str) -> list[str]:
    return [
        name
        for name in imported_modules(source, package)
        if name == "ai_arch_toolkit.toolkit" or name.startswith("ai_arch_toolkit.toolkit.")
    ]


def test_core_never_imports_toolkit() -> None:
    violations = {}
    for path in CORE.rglob("*.py"):
        package = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts[:-1])
        found = forbidden_toolkit_imports(path.read_text(), package)
        if found:
            violations[str(path.relative_to(ROOT))] = found
    assert not violations


def test_import_boundary_detector_rejects_absolute_and_relative_imports() -> None:
    assert forbidden_toolkit_imports(
        "from ai_arch_toolkit.toolkit.budget import BudgetPolicy", "ai_arch_toolkit.core"
    )
    assert forbidden_toolkit_imports("from ..toolkit import budget", "ai_arch_toolkit.core")
    assert not forbidden_toolkit_imports("from ._response import Usage", "ai_arch_toolkit.core")


def capability_discovery(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("getattr", "hasattr")
    ]


def test_call_pipeline_has_explicit_capabilities() -> None:
    for name in ("_llm.py", "_attempts.py", "_response.py", "_sync.py"):
        path = CORE / name
        assert path.exists(), name
        assert not capability_discovery(path.read_text()), name


def test_capability_detector_rejects_magic_finalizer_hooks() -> None:
    assert capability_discovery('getattr(finalizer, "_stream_abandon", None)')
    assert capability_discovery('hasattr(controller, "failure_bound")')
    assert not capability_discovery("lifecycle.abandon()")


def test_call_facade_is_small_and_functions_are_bounded() -> None:
    facade = CORE / "_llm.py"
    assert len(facade.read_text().splitlines()) < 800
    for path in (facade, CORE / "_attempts.py"):
        tree = ast.parse(path.read_text())
        lengths = {
            f"{node.name}:{node.lineno}": node.end_lineno - node.lineno + 1
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert max(lengths.values()) <= 60, lengths


def test_provider_dispatch_and_attempt_start_have_one_home() -> None:
    attempts = ast.parse((CORE / "_attempts.py").read_text())
    facade = ast.parse((CORE / "_llm.py").read_text())
    dispatch = next(
        node
        for node in attempts.body
        if isinstance(node, ast.FunctionDef) and node.name == "dispatch"
    )

    def provider_calls(tree: ast.AST, names: tuple[str, ...]) -> set[int]:
        return {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in names
            and "provider" in ast.unparse(node.func.value)
        }

    io = ("complete", "stream", "stream_events")
    assert len(provider_calls(dispatch, io)) == 2  # complete, and one stream for both views
    assert provider_calls(attempts, io) == provider_calls(dispatch, io)
    assert not provider_calls(facade, io)
    # Preparation happens before admission, in the pipeline only.
    assert provider_calls(attempts, ("prepare",))
    assert not provider_calls(facade, ("prepare",))
    starts = [
        node
        for node in ast.walk(attempts)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mark_started"
    ]
    assert len(starts) == 1
    old_names = {
        "_try_with_tracking",
        "_StreamRun",
        "_FallbackStreamRun",
        "_single_stream",
        "_stream_with_fallbacks",
        "_meter_request",
    }
    assert (
        not {
            node.name
            for node in ast.walk(facade)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        & old_names
    )


def model_prefix_checks(source: str) -> list[int]:
    """Lines comparing a model id's prefix or suffix outside the id grammar."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("startswith", "endswith")
        and "model" in ast.unparse(node.func.value).lower()
    ]


def _model_prefix_offenders() -> set[str]:
    return {
        str(path.relative_to(CORE))
        for path in CORE.rglob("*.py")
        if path.name != "_model_id.py" and model_prefix_checks(path.read_text())
    }


def test_model_ids_are_matched_only_by_the_grammar() -> None:
    assert _model_prefix_offenders() == set()


def test_model_prefix_detector_rejects_prefix_and_suffix_checks() -> None:
    assert model_prefix_checks('self._model.startswith("gpt-5")')
    assert model_prefix_checks("model_id.endswith(SUFFIXES)")
    assert not model_prefix_checks('path.startswith("/")')


_SDK_MODULES = ("openai", "anthropic", "httpx", "httpx2", "grpc", "genai_errors", "aiohttp")


def sdk_errors_outside_the_mapper(source: str) -> list[int]:
    """Lines that catch or test an SDK exception anywhere but a ``map_error`` method."""

    def mentions_sdk(node: ast.AST | None) -> bool:
        return node is not None and any(
            isinstance(name, ast.Name) and name.id in _SDK_MODULES for name in ast.walk(node)
        )

    lines: list[int] = []

    def visit(node: ast.AST, in_mapper: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inside = in_mapper or (
                isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                and child.name == "map_error"
            )
            if isinstance(child, ast.ExceptHandler) and mentions_sdk(child.type):
                lines.append(child.lineno)
            if (
                not inside
                and isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "isinstance"
                and len(child.args) == 2
                and mentions_sdk(child.args[1])
            ):
                lines.append(child.lineno)
            visit(child, inside)

    visit(ast.parse(source), False)
    return lines


def _sdk_error_offenders() -> set[str]:
    return {
        str(path.relative_to(CORE))
        for path in (CORE / "_providers").glob("_*.py")
        if sdk_errors_outside_the_mapper(path.read_text())
    }


def test_sdk_exceptions_are_known_only_by_each_adapters_mapper() -> None:
    assert _sdk_error_offenders() == set()


def test_sdk_exception_detector_flags_catches_and_checks_outside_the_mapper() -> None:
    assert sdk_errors_outside_the_mapper("try:\n    x()\nexcept openai.APIError:\n    pass\n")
    assert sdk_errors_outside_the_mapper(
        "def send(e):\n    return isinstance(e, httpx2.ReadError)\n"
    )
    mapper = "def map_error(self, exc, *, sent):\n    return isinstance(exc, openai.APIError)\n"
    assert not sdk_errors_outside_the_mapper(mapper)


_NETWORK_MODULES = ("urllib.request", "urllib.error", "http.client", "socket", "ssl")
_HTTP_DOOR = ROOT / "src/ai_arch_toolkit/toolkit/tools/_http.py"


def network_access(source: str) -> list[int]:
    """Lines that import or name a stdlib module that reaches the network.

    ``urllib.parse`` only parses, so it is not one of them.
    """

    def reaches(name: str) -> bool:
        return any(name == module or name.startswith(f"{module}.") for module in _NETWORK_MODULES)

    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names = [f"{node.value.id}.{node.attr}"]
        if any(reaches(name) for name in names):
            lines.append(node.lineno)
    return lines


def _network_offenders() -> set[str]:
    return {
        str(path.relative_to(ROOT))
        for path in (ROOT / "src/ai_arch_toolkit").rglob("*.py")
        if "nanope" not in path.parts and path != _HTTP_DOOR and network_access(path.read_text())
    }


def test_only_the_tools_http_module_reaches_the_network() -> None:
    assert _network_offenders() == set()


def test_network_detector_flags_imports_and_attribute_use() -> None:
    assert network_access("import urllib.request\n")
    assert network_access("from urllib import request\n")
    assert network_access("import urllib.parse\nurllib.request.urlopen('x')\n")
    assert network_access("from http.client import HTTPSConnection\n")
    assert network_access("import socket\n")
    assert not network_access("from urllib.parse import urlsplit\n")
    assert not network_access("from http import HTTPStatus\n")


def duplicated_windows(first: str, second: str, size: int = 8) -> int:
    """How many runs of ``size`` lines (whitespace-normalised, at least 6 non-blank) repeat."""

    def windows(source: str) -> set[str]:
        lines = [line.strip() for line in source.splitlines()]
        return {
            "\n".join(lines[i : i + size])
            for i in range(len(lines) - size + 1)
            if sum(1 for line in lines[i : i + size] if line) >= 6
        }

    return len(windows(first) & windows(second))


def test_the_memory_graph_store_does_not_reimplement_the_graph_facade() -> None:
    core = (CORE / "graph/_store.py").read_text()
    memory = (ROOT / "src/ai_arch_toolkit/toolkit/memory/graph/_store.py").read_text()

    assert duplicated_windows(core, memory) == 0


def test_the_duplicate_detector_sees_a_copied_block() -> None:
    block = "\n".join(f"x{i} = {i}" for i in range(10))
    assert duplicated_windows(block, "\n".join(["pass", block, "pass"])) == 3
    assert duplicated_windows(block, block.replace("x5", "y5")) == 0


_FLOW_OPTIONS = frozenset({"timeout", "trace_capture", "policy", "budget_policy"})
_FACTORIES = ROOT / "src/ai_arch_toolkit/toolkit/agents/flows"
_BUILDERS = ROOT / "src/ai_arch_toolkit/toolkit/agents/_builders.py"


def flow_options_by_name(source: str) -> list[str]:
    """Where a module takes one of the four ``Flow`` options as a parameter or passes it by name.

    The options travel together, as ``**options`` (``FlowOptions``); spelling one out repeats the
    declaration this rule keeps in one place.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            found += [f"{node.name}({a.arg})" for a in arguments if a.arg in _FLOW_OPTIONS]
        elif isinstance(node, ast.keyword) and node.arg in _FLOW_OPTIONS:
            found.append(f"{node.arg}= (line {node.value.lineno})")
    return found


def spec_options_read_outside(source: str, reader: str) -> list[str]:
    """Functions other than ``reader`` that read one of a spec's ``Flow`` options."""
    return [
        f"{func.name}: .{node.attr}"
        for func in ast.parse(source).body
        if isinstance(func, ast.FunctionDef) and func.name != reader
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute) and node.attr in _FLOW_OPTIONS
    ]


def test_the_flow_factories_take_the_flow_options_as_one_set() -> None:
    offenders = {
        path.name: found
        for path in sorted(_FACTORIES.glob("_*.py"))
        if (found := flow_options_by_name(path.read_text()))
    }
    assert offenders == {}
    factories = [
        node
        for path in sorted(_FACTORIES.glob("_*.py"))
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_flow")
    ]
    assert len(factories) == 9
    unpacked = {
        node.name: ast.unparse(node.args.kwarg.annotation)
        for node in factories
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None
    }
    assert unpacked == {node.name: "Unpack[FlowOptions]" for node in factories}


def test_the_builders_read_the_specs_flow_options_in_one_place() -> None:
    source = _BUILDERS.read_text()

    assert flow_options_by_name(source) == []
    assert spec_options_read_outside(source, "_flow_options") == []


def test_the_flow_option_detectors_see_a_spelled_out_option() -> None:
    source = "def f(llm, *, timeout=None):\n    return Flow(policy=p, name='x', **options)\n"
    assert flow_options_by_name(source) == ["f(timeout)", "policy= (line 2)"]
    assert flow_options_by_name("def f(llm, **options):\n    return Flow(**options)\n") == []
    reads = "def _flow_options(s):\n    return s.policy\n\ndef b(s):\n    return s.timeout\n"
    assert spec_options_read_outside(reads, "_flow_options") == ["b: .timeout"]


_STATE_KEYS = frozenset({"task", "messages", "answer", "response"})
_AGENTS = ROOT / "src/ai_arch_toolkit/toolkit/agents"
_KEYS_MODULE = _AGENTS / "flows/_keys.py"


def state_key_literals(source: str) -> list[int]:
    """Lines that spell one of the state keys the strategies and the runner share."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and node.value in _STATE_KEYS
    ]


def test_the_shared_state_keys_are_spelled_in_one_module() -> None:
    offenders = {
        str(path.relative_to(ROOT)): lines
        for path in sorted(_AGENTS.rglob("*.py"))
        if path != _KEYS_MODULE and (lines := state_key_literals(path.read_text()))
    }
    assert offenders == {}


def test_the_state_key_detector_sees_a_key_spelled_out() -> None:
    assert state_key_literals('answer = snap.require("task")\n') == [1]
    assert state_key_literals('x = {ANSWER: text}\nprint(f"Task: {task}", "Answer:")\n') == []


def test_the_python_evaluator_vets_callables_without_capability_discovery() -> None:
    evaluator = ROOT / "src/ai_arch_toolkit/toolkit/tools/_python.py"

    assert capability_discovery(evaluator.read_text()) == []


def react_loops_built(source: str) -> list[int]:
    """Lines that build or seed a ReAct flow by hand, which only ``_react`` may do."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"react_flow", "react_initial_state"}
    ]


def test_the_strategies_run_an_inner_react_through_one_primitive() -> None:
    builders = {
        path.name: lines
        for path in sorted(_FACTORIES.glob("_*.py"))
        if path.name != "_react.py" and (lines := react_loops_built(path.read_text()))
    }
    assert builders == {}


def test_the_react_loop_detector_sees_a_hand_built_loop() -> None:
    assert react_loops_built("inner = react_flow(llm, tools)\n") == [1]
    assert react_loops_built("state = State(operational=react_initial_state(task))\n") == [1]
    assert react_loops_built("answer = await run_react(llm, tools, task)\n") == []
