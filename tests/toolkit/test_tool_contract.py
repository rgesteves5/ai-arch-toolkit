"""The contract every toolkit tool keeps (T00, D37 to D42), and the debt of those that do not yet.

Per tool, five points, each proven by a case (``contract_cases.py``, ``error_bodies.py``) or
read from the tool itself:

- ``window``: a tool that cuts ends its text with the window's footer (T03), and the call the
  footer names returns the next part, numbered from where the first one ended; a tool whose output
  exists only for its call (``ONCE``: a command's, a program's) names no call, and says the size
  and how to narrow the output instead;
- ``errors``: each error answer its source documents gives a typed ``ToolFailure`` that carries
  the source's words;
- ``not_found``: a lookup of a resource that does not exist is ``not_found``;
- ``zero``: a search that finds nothing is a success that says so, with the query;
- ``limits``: every integer parameter declares its limits (``Range``, T04a), the docstring does not
  contradict them, and neither the tool nor a helper of its module clamps them on its own.

A point that fails is debt, listed in ``contract_debt.py``: the test fails when a tool owes a point
the list does not name (the list only shrinks) and when it keeps one the list still names (delete
the line). The network is the canned answers' (``_http._open``), behind a fresh throttle for each
check, and a library a tool reaches without the door is the case's stand-in (``patches``); sockets
are blocked.
"""

from __future__ import annotations

import ast
import email.message
import inspect
import io
import json
import re
import sys
import types
import typing
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import pytest

from ai_arch_toolkit.core import Range, ToolFailure, tool
from ai_arch_toolkit.toolkit.tools import _http
from ai_arch_toolkit.toolkit.tools._http import Api
from ai_arch_toolkit.toolkit.tools._window import Window, find_window, page_window, text_window
from tests.toolkit.contract_cases import (
    KINDS,
    NOT_FOUND_CASES,
    ONCE,
    SOURCELESS,
    UNBOUNDED,
    WHOLE,
    WINDOW_CASES,
    ZERO_CASES,
    Answer,
    Case,
    ZeroCase,
)
from tests.toolkit.contract_debt import CUT_HELPERS, DEBT
from tests.toolkit.error_bodies import ERROR_BODIES, ErrorBody
from tests.toolkit.http_fakes import FakeResponse
from tests.toolkit.tool_catalog import (
    NETWORK,
    PACKAGE,
    TOOLS,
    benign,
    schema_properties,
    text_of,
)

POINTS = ("window", "errors", "not_found", "zero", "limits")
_TOOLS_DIR = Path(sys.modules[PACKAGE].__path__[0])

type Tool = Callable[..., Any]


# --- Calling a tool against canned answers ------------------------------------------------------


def _response(answer: Answer) -> FakeResponse:
    body = answer.body
    if isinstance(body, dict | list):
        data, kind = json.dumps(body).encode(), "application/json"
    elif isinstance(body, str):
        data, kind = body.encode(), "text/plain; charset=utf-8"
    else:
        data, kind = body, "application/json"
    response = FakeResponse(data, kind, answer.status)
    for name, value in answer.headers.items():
        response.headers[name] = value
    return response


def _opener(answers: tuple[Answer, ...]) -> Callable[[urllib.request.Request, float], Any]:
    """An ``_http._open`` that gives ``answers`` in turn, the last one again once they run out."""
    served = iter(answers or (Answer(status=599),))
    last: list[Answer] = []

    def open_(request: urllib.request.Request, timeout: float) -> Any:
        answer = next(served, None) or last[0]
        last[:] = [answer]
        if answer.status < 400:
            return _response(answer)
        headers = email.message.Message()
        for name, value in answer.headers.items():
            headers[name] = value
        raw = _response(answer).read1(-1)
        raise urllib.error.HTTPError(
            request.full_url, answer.status, "Error", headers, io.BytesIO(raw)
        )

    return open_


def _serve(monkeypatch: pytest.MonkeyPatch, case: Case) -> None:
    """The source answers ``case``'s answers, behind a throttle of its own (a rest one answer
    asks for does not reach the next check), ``case``'s patches are in place and its files are
    on disk."""
    monkeypatch.setattr(_http, "_THROTTLE", _http._Throttle(sleep=lambda _seconds: None))
    monkeypatch.setattr(_http, "_open", _opener(case.answers))
    for target, value in case.patches.items():
        monkeypatch.setattr(target, value)
    for relative, content in case.files.items():
        path = Path.cwd() / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _outcome(fn: Tool, args: dict[str, Any]) -> str | ToolFailure | None:
    """The tool's text or typed failure; ``None`` for anything else (it broke the contract)."""
    try:
        result = fn(**args)
    except ToolFailure as failure:
        return failure
    except Exception:  # any other exception is a broken contract, not a crash of the test
        return None
    return text_of(result)


def _args(fn: Tool, case: Case) -> dict[str, Any]:
    base = benign(fn.__name__) if fn.__name__ in TOOLS else {}
    return {**base, **case.args}


def _run(fn: Tool, case: Case, monkeypatch: pytest.MonkeyPatch) -> str | ToolFailure | None:
    _serve(monkeypatch, case)
    return _outcome(fn, _args(fn, case))


# --- Point 1: the window ---------------------------------------------------------------------

_FOOTER = re.compile(
    r"\[(?P<unit>chars|results|matches) (?P<first>\d+)-(?P<last>\d+)(?: of (?P<total>\d+))?"
    r'(?: for ".*?")? \| (?P<onward>.*)\]\s*\Z'
)


def _next_call(onward: str) -> dict[str, Any] | None:
    """The arguments of ``next: a=1, b="x"``; ``None`` when the footer names no next call."""
    if not onward.startswith("next: "):
        return None
    text, args, decoder = onward.removeprefix("next: "), {}, json.JSONDecoder()
    try:
        while text:
            name, _, rest = text.partition("=")
            value, end = decoder.raw_decode(rest)
            args[name.strip()] = value
            text = rest[end:].removeprefix(",").strip()
    except ValueError:  # not the window's footer
        return None
    return args or None


def _window(text: str | ToolFailure | None) -> tuple[re.Match[str], str] | None:
    """The footer of a windowed text and the body above it."""
    if not isinstance(text, str) or (footer := _FOOTER.search(text)) is None:
        return None
    return footer, text[: footer.start()]


def window_kept(fn: Tool, case: Case | None, monkeypatch: pytest.MonkeyPatch) -> bool:
    """The footer names a next call, whose answer is another part, numbered on from the first."""
    if case is None:
        return False
    _serve(monkeypatch, case)
    args = _args(fn, case)
    first = _window(_outcome(fn, args))
    onward = _next_call(first[0]["onward"]) if first else None
    if first is None or onward is None:
        return False
    second = _window(_outcome(fn, {**args, **onward}))
    if second is None or second[0]["unit"] != first[0]["unit"] or second[1] == first[1]:
        return False
    step = 0 if first[0]["unit"] == "chars" else 1
    return int(second[0]["first"]) == int(first[0]["last"]) + step


def once_kept(
    fn: Tool, case: Case | None, monkeypatch: pytest.MonkeyPatch, narrow: str = ""
) -> bool:
    """The output that does not fit ends with the window's footer: what was shown, the size, and
    how to narrow the output, in the words the tool declares (``narrow``, else its ``ONCE``
    entry's), where other tools name the call that reads on. No next call, not even a malformed
    one, and no dead end in other words."""
    if case is None:
        return False
    found = _window(_run(fn, case, monkeypatch))
    if found is None:
        return False
    footer = found[0]
    onward = footer["onward"]
    words = narrow or ONCE[fn.__name__].narrow
    return (
        footer["unit"] == "chars"
        and footer["total"] is not None
        and int(footer["last"]) < int(footer["total"])
        and not onward.startswith("next")
        and words in onward
    )


# --- Point 2: the source's errors --------------------------------------------------------------


def _module(fn: Tool) -> ModuleType:
    return sys.modules[fn.__module__]


def _has_source(name: str) -> bool:
    """A network tool whose source documents its errors (not one of ``SOURCELESS``)."""
    return name in NETWORK and TOOLS[name].__module__.rsplit(".", 1)[-1] not in SOURCELESS


def _error_bodies(name: str) -> list[ErrorBody]:
    module = TOOLS[name].__module__.rsplit(".", 1)[-1]
    return [
        body
        for body in ERROR_BODIES.get(module, ())
        if name in body.tools or (not body.tools and name in NETWORK)
    ]


def errors_kept(fn: Tool, bodies: list[ErrorBody], monkeypatch: pytest.MonkeyPatch) -> bool:
    """Each error answer gives a failure of its type that carries the source's words."""
    for body in bodies:
        answer = Answer(body=body.body, status=body.status, headers=body.headers)
        failure = _run(fn, Case(args={}, answers=(answer,)), monkeypatch)
        if not (
            isinstance(failure, ToolFailure)
            and failure.error.type == body.type
            and body.says in failure.error.message
        ):
            return False
    return bool(bodies)


# --- Point 3: a missing resource, zero results --------------------------------------------------

_NOTHING = re.compile(r"\b(no|not|none|nothing|zero|0)\b", re.IGNORECASE)


def not_found_kept(fn: Tool, case: Case | None, monkeypatch: pytest.MonkeyPatch) -> bool:
    if case is None:
        return False
    failure = _run(fn, case, monkeypatch)
    return isinstance(failure, ToolFailure) and failure.error.type == "not_found"


def zero_kept(fn: Tool, case: ZeroCase | None, monkeypatch: pytest.MonkeyPatch) -> bool:
    """The text says nothing was found, and names the query."""
    if case is None:
        return False
    text = _run(fn, case, monkeypatch)
    return isinstance(text, str) and case.says in text and _NOTHING.search(text) is not None


# --- Point 4: limits in the signature ----------------------------------------------------------

_DOC_RANGE = re.compile(r"\((-?\d+)\s*(?:-|to)\s*(-?\d+)\)")


def _integer_branches(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The schema's integer branches of a parameter: itself, or its ``anyOf`` members."""
    return [branch for branch in spec.get("anyOf", [spec]) if branch.get("type") == "integer"]


def _callee(node: ast.AST) -> str:
    """The name a call calls: ``f(…)`` and ``x.f(…)`` give ``f``; anything else, ``""``."""
    if not isinstance(node, ast.Call):
        return ""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return node.func.attr if isinstance(node.func, ast.Attribute) else ""


def _reached(fn: Tool, param: str) -> Iterator[tuple[ast.FunctionDef, frozenset[str]]]:
    """The tool's definition and each function of its module it hands ``param`` to, directly or
    not, with the names ``param`` goes by there."""
    defs = {
        node.name: node
        for node in ast.parse(inspect.getsource(_module(fn))).body
        if isinstance(node, ast.FunctionDef)
    }
    seen: set[tuple[str, frozenset[str]]] = set()
    pending = [(fn.__name__, frozenset({param}))]
    while pending:
        name, names = pending.pop()
        if (name, names) in seen or name not in defs:
            continue
        seen.add((name, names))
        yield defs[name], names
        for call in ast.walk(defs[name]):
            if not isinstance(call, ast.Call) or (callee := _callee(call)) not in defs:
                continue
            params = [arg.arg for arg in defs[callee].args.args]
            handed = {
                params[index]
                for index, arg in enumerate(call.args)
                if index < len(params) and _mentions(arg, names)
            }
            handed |= {kw.arg for kw in call.keywords if kw.arg and _mentions(kw.value, names)}
            if handed:
                pending.append((callee, frozenset(handed)))


def _mentions(node: ast.AST, names: frozenset[str]) -> bool:
    return any(isinstance(inner, ast.Name) and inner.id in names for inner in ast.walk(node))


def _clamps(fn: Tool, param: str) -> bool:
    """The tool or a helper it hands ``param`` to bounds it itself: a ``*clamp*(param, …)`` call,
    or a ``min``/``max`` of ``param`` (or of a ``min``/``max`` of it) and a constant."""

    def constant(node: ast.AST) -> bool:
        if isinstance(node, ast.UnaryOp):
            node = node.operand
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int | float) and not isinstance(node.value, bool)
        return isinstance(node, ast.Name) and node.id.lstrip("_").isupper()

    for definition, names in _reached(fn, param):

        def named(node: ast.AST, names: frozenset[str] = names) -> bool:
            return isinstance(node, ast.Name) and node.id in names

        def holds(node: ast.AST, named: Callable[[ast.AST], bool] = named) -> bool:
            if isinstance(node, ast.Call) and _callee(node) in ("min", "max"):
                return any(named(arg) for arg in node.args)
            return named(node)

        for node in ast.walk(definition):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee(node)
            if "clamp" in callee.lower() and any(named(arg) for arg in node.args):
                return True
            minmax = callee in ("min", "max") and any(holds(arg) for arg in node.args)
            if minmax and any(constant(arg) for arg in node.args):
                return True
    return False


def _declared_int(fn: Tool, param: str) -> bool:
    """The signature makes ``param`` an ``int`` (through aliases, ``Annotated`` and unions)."""
    try:
        pending = [typing.get_type_hints(fn, include_extras=True).get(param)]
    except (NameError, TypeError):
        return False
    while pending:
        hint = pending.pop()
        while isinstance(hint, typing.TypeAliasType):
            hint = hint.__value__
        origin = typing.get_origin(hint)
        if origin is typing.Annotated:
            pending.append(typing.get_args(hint)[0])
        elif origin in (types.UnionType, typing.Union):
            pending.extend(typing.get_args(hint))
        elif hint is int:
            return True
    return False


def limits_kept(fn: Tool, unbounded: dict[tuple[str, str], str]) -> bool:
    """Every integer parameter is an integer in the schema, bounded there (or listed as
    unbounded, with why), its docstring's range is the schema's, and no code of its module
    clamps it."""
    for param, spec in schema_properties(fn).items():
        if _declared_int(fn, param) and not _integer_branches(spec):
            return False  # the schema lost the type, and with it any bound
        for branch in _integer_branches(spec):
            bounds = (branch.get("minimum"), branch.get("maximum"))
            if bounds == (None, None) and (fn.__name__, param) not in unbounded:
                return False
            written = _DOC_RANGE.search(spec.get("description", ""))
            if written is not None and (int(written[1]), int(written[2])) != bounds:
                return False
            if _clamps(fn, param):
                return False
    return True


# --- The debt ----------------------------------------------------------------------------------


def unmet(name: str, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The points ``name`` does not keep."""
    fn = TOOLS[name]
    for api in vars(_module(fn)).values():
        if isinstance(api, Api) and api.key_env:  # a key the tool needs is there
            monkeypatch.setenv(api.key_env, "test-key")
    owed: set[str] = set()
    window = once_kept if name in ONCE else window_kept
    if name not in WHOLE and not window(fn, WINDOW_CASES.get(name), monkeypatch):
        owed.add("window")
    if _has_source(name) and not errors_kept(fn, _error_bodies(name), monkeypatch):
        owed.add("errors")
    lookup = KINDS.get(name) == "lookup"
    if lookup and not not_found_kept(fn, NOT_FOUND_CASES.get(name), monkeypatch):
        owed.add("not_found")
    if KINDS.get(name) == "search" and not zero_kept(fn, ZERO_CASES.get(name), monkeypatch):
        owed.add("zero")
    if not limits_kept(fn, UNBOUNDED):
        owed.add("limits")
    return owed


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_each_tool_keeps_the_contract_or_owes_what_the_debt_lists(
    name: str, monkeypatch: pytest.MonkeyPatch, sandbox: Path
) -> None:
    owed = unmet(name, monkeypatch)
    listed = DEBT.get(name, frozenset())

    assert owed <= listed, (
        f"{name} owes {sorted(owed - listed)}, which tests/toolkit/contract_debt.py does not "
        "list: keep the contract (the debt list only shrinks)"
    )
    assert listed <= owed, (
        f"{name} now keeps {sorted(listed - owed)}: delete it from tests/toolkit/contract_debt.py"
    )


def test_the_debt_names_only_tools_and_points() -> None:
    assert set(DEBT) <= set(TOOLS)
    assert all(points and set(points) <= set(POINTS) for points in DEBT.values())


def test_every_tool_says_what_it_does_and_every_case_can_run() -> None:
    assert set(KINDS) == set(TOOLS)
    assert set(TOOLS) >= WHOLE
    assert set(ONCE) <= set(TOOLS) - WHOLE
    assert {name for name, _param in UNBOUNDED} <= set(TOOLS)
    assert set(WINDOW_CASES) <= set(TOOLS) - WHOLE
    assert {name for name in NOT_FOUND_CASES if KINDS.get(name) != "lookup"} == set()
    assert {name for name in ZERO_CASES if KINDS.get(name) != "search"} == set()
    modules = {path.stem for path in _TOOLS_DIR.glob("_*.py")}
    assert set(ERROR_BODIES) <= modules
    assert set(SOURCELESS) <= modules
    misplaced = {
        name
        for module, bodies in ERROR_BODIES.items()
        for body in bodies
        for name in body.tools
        if name not in TOOLS or TOOLS[name].__module__.rsplit(".", 1)[-1] != module
    }
    assert misplaced == set()


def _cut_helpers() -> int:
    """Definitions of the copied cut helpers, ``_truncate`` and ``_trim``, in ``toolkit/tools``."""
    return sum(
        isinstance(node, ast.FunctionDef) and node.name in ("_truncate", "_trim")
        for path in _TOOLS_DIR.glob("_*.py")
        for node in ast.walk(ast.parse(path.read_text()))
    )


def test_the_copied_cut_helpers_only_go() -> None:
    # Every cut goes through the window (T03, D39): whoever deletes a copy lowers the count.
    assert _cut_helpers() == CUT_HELPERS


# --- The checks themselves ---------------------------------------------------------------------

_LINES = "".join(f"line {number}\n" for number in range(200))
_PAGE = 20


@tool(capability="compute")
def windowed(offset: Annotated[int, Range(0)] = 0) -> str:
    """Read a long text.

    Args:
        offset: Where to start.
    """
    return text_window(_LINES, offset=offset, limit=300).text()


@tool(capability="compute")
def paged(offset: Annotated[int, Range(0)] = 0) -> str:
    """List many items.

    Args:
        offset: The first item.
    """
    return page_window(_LINES.splitlines(), offset=offset, limit=20).text()


@tool(capability="compute")
def found(find: str = "line 1", offset: Annotated[int, Range(0)] = 0) -> str:
    """Find a term in a long text.

    Args:
        find: The term.
        offset: Where to search on from.
    """
    return find_window(_LINES, find, offset=offset, limit=200, context=10).text()


@tool(capability="compute")
def stuck(offset: Annotated[int, Range(0)] = 0) -> str:
    """Claim to read on, but always show the start.

    Args:
        offset: Ignored.
    """
    return text_window(_LINES, offset=0, limit=300).text()


def _start_of_output(rest: str) -> Window:
    return Window(body=_LINES[:300], unit="chars", first=0, last=300, total=len(_LINES), rest=rest)


@tool(capability="compute")
def narrowed(code: str = "") -> str:
    """Run something whose output cannot be read again.

    Args:
        code: Ignored.
    """
    return _start_of_output("the rest is not kept: run it again printing less").text()


@tool(capability="compute")
def dead_end(code: str = "") -> str:
    """Run something, and leave the rest of its output out of reach.

    Args:
        code: Ignored.
    """
    return _start_of_output("").text()


@tool(capability="compute")
def misnamed(code: str = "") -> str:
    """Run something, and name a next call that is not one.

    Args:
        code: Ignored.
    """
    return _start_of_output("next: run it again printing less").text()


@tool(capability="compute")
def reworded(code: str = "") -> str:
    """Run something, and say in other words that the rest is out of reach.

    Args:
        code: Ignored.
    """
    return _start_of_output("the rest cannot be read").text()


@tool(capability="compute")
def silent(max_results: int = 10) -> str:
    """List some items.

    Args:
        max_results: How many (1-50).
    """
    return "\n".join(_LINES.splitlines()[: max(1, min(max_results, 50))])


@tool(capability="compute")
def bounded(max_results: Annotated[int, Range(1, 50)] = 10) -> str:
    """List some items.

    Args:
        max_results: How many (1-50).
    """
    return "\n".join(_LINES.splitlines()[:max_results])


def _first(items: list[str], count: int) -> list[str]:
    return items[: min(count, _PAGE)]


@tool(capability="compute")
def helped(max_results: Annotated[int, Range(1, 50)] = 10) -> str:
    """List some items.

    Args:
        max_results: How many (1-50).
    """
    return "\n".join(_first(_LINES.splitlines(), max_results))


@tool(capability="compute", schema={"max_results": {"type": "string"}})
def mistyped(max_results: Annotated[int, Range(1, 50)] = 10) -> str:
    """List some items.

    Args:
        max_results: How many.
    """
    return "\n".join(_LINES.splitlines()[:max_results])


@tool(capability="compute")
def contradicted(max_results: Annotated[int, Range(1, 50)] = 10) -> str:
    """List some items.

    Args:
        max_results: How many (1 to 20).
    """
    return "\n".join(_LINES.splitlines()[:max_results])


@tool(capability="compute")
def search(query: str = "zzqq") -> str:
    """Search.

    Args:
        query: The query.
    """
    return f"No items found for {query!r}."


@tool(capability="compute")
def listing(query: str = "zzqq") -> str:
    """Search, and print a heading over nothing.

    Args:
        query: The query.
    """
    return f"Results for {query!r}:\n"


def _library() -> str:
    return "the real library"


@tool(capability="compute")
def through_a_library() -> str:
    """Answer what a library outside the HTTP door says."""
    return _library()


@tool(capability="compute")
def lookup(key: str = "x") -> str:
    """Look up one item.

    Args:
        key: The item.
    """
    raise ToolFailure("not_found", f"no item {key!r}; search with search")


@tool(capability="compute")
def refusing(key: str = "x") -> str:
    """Refuse every key.

    Args:
        key: The item.
    """
    raise ToolFailure("validation_error", f"invalid key {key!r}")


class TestTheChecks:
    """Each check passes a tool that keeps its point and fails one that does not."""

    @pytest.mark.parametrize("fn", [windowed, paged, found])
    def test_a_window_whose_next_call_reads_on_is_kept(
        self, fn: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert window_kept(fn, Case(args={}), monkeypatch)

    def test_a_silent_cut_a_window_that_does_not_move_or_no_case_owe_the_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert not window_kept(silent, Case(args={}), monkeypatch)
        assert not window_kept(stuck, Case(args={}), monkeypatch)
        assert not window_kept(windowed, None, monkeypatch)

    def test_an_output_that_says_its_size_and_how_to_narrow_it_is_kept_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        words = "printing less"
        assert once_kept(narrowed, Case(args={}), monkeypatch, words)
        assert not once_kept(dead_end, Case(args={}), monkeypatch, words)  # no way to the rest
        assert not once_kept(reworded, Case(args={}), monkeypatch, words)  # nor in other words
        assert not once_kept(windowed, Case(args={}), monkeypatch, words)  # it names a next call
        assert not once_kept(misnamed, Case(args={}), monkeypatch, words)  # a malformed one too
        assert not once_kept(narrowed, Case(args={}), monkeypatch, "| grep")  # not its words
        assert not once_kept(silent, Case(args={}), monkeypatch, words)
        assert not once_kept(narrowed, None, monkeypatch, words)

    def test_limits_in_the_signature_are_kept(self) -> None:
        assert limits_kept(bounded, {})
        assert limits_kept(windowed, {})

    def test_a_clamp_a_helpers_clamp_a_contradicting_docstring_or_no_bound_owe_the_limits(
        self,
    ) -> None:
        assert not limits_kept(silent, {})
        assert not limits_kept(helped, {})
        assert not limits_kept(contradicted, {})
        assert not limits_kept(mistyped, {})  # an int the schema no longer calls an integer
        assert not limits_kept(silent, {("silent", "max_results"): "test"})  # it still clamps

    def test_zero_results_said_with_the_query_are_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert zero_kept(search, ZeroCase(args={}, says="zzqq"), monkeypatch)
        assert not zero_kept(listing, ZeroCase(args={}, says="zzqq"), monkeypatch)
        assert not zero_kept(search, ZeroCase(args={}, says="other"), monkeypatch)
        assert not zero_kept(search, None, monkeypatch)

    def test_only_not_found_keeps_the_missing_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert not_found_kept(lookup, Case(args={}), monkeypatch)
        assert not not_found_kept(refusing, Case(args={}), monkeypatch)
        assert not not_found_kept(lookup, None, monkeypatch)

    def test_an_error_answer_must_give_its_type_and_the_sources_words(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        said = ErrorBody(source="test", status=404, type="not_found", says="no item")
        other = ErrorBody(source="test", status=404, type="not_found", says="gone")
        assert errors_kept(lookup, [said], monkeypatch)
        assert not errors_kept(lookup, [said, other], monkeypatch)
        assert not errors_kept(refusing, [said], monkeypatch)
        assert not errors_kept(lookup, [], monkeypatch)

    def test_a_cases_patches_stand_in_for_a_source_reached_without_the_door(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case = Case(args={}, patches={f"{__name__}._library": lambda: "the canned answer"})

        assert _run(through_a_library, case, monkeypatch) == "the canned answer"

    def test_the_next_call_reads_every_argument(self) -> None:
        assert _next_call('next: find="a, b=c", offset=16500') == {
            "find": "a, b=c",
            "offset": 16500,
        }
        assert _next_call("end") is None
        assert _next_call("next: page two") is None
