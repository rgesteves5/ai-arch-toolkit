# Code Style

This project uses modern, Ruff-formatted Python. "Google-style" in this repo
means the docstring format, not the full Google Python codebase style. The
code is intentionally closer to typed, async-first library Python: small public
facades, internal helpers, frozen data contracts, and explicit factories.

## Linting Contract

- Target Python is 3.13+. Every Python file starts with
  `from __future__ import annotations`.
- Ruff is the formatter and linter. The repo uses `target-version = "py313"`,
  line length 99, and rules `E F W I UP B SIM RUF`.
- Run `uv run ruff format src tests examples` after edits. Run
  `uv run ruff check src tests examples` before handing off a code change.
- `src/ai_arch_toolkit/nanope` is excluded from the strict repo-wide Ruff and
  Pyright checks because it contains work-in-progress app code with its own
  idioms.
- Do not hand-format imports or wrapping. Let Ruff choose the final layout.

The result should look compact and regular: sorted imports, one clear blank-line
rhythm, no decorative alignment, no manual wrapping games, and no type comments
where annotations can express the same thing.

## Code Shape

Prefer the existing local style:

- Public API is re-exported through package `__init__.py` files with explicit
  `__all__`.
- Internal modules are `_`-prefixed.
- Shared data shapes use PEP 695 type aliases, for example
  `type Content = str | list[ContentPart]`.
- Dataclasses are `frozen=True, slots=True`; add `kw_only=True` once they have
  three or more fields.
- Core code is async-first. Sync wrappers are convenience surfaces over async
  implementations, not separate logic paths.
- Toolkit tools return useful error strings instead of raising, so agents can
  continue from the tool result.

The repo should read as a typed toolkit, not as an object hierarchy for its own
sake. Keep public contracts explicit and keep implementation helpers small.

## Classes, Dataclasses, And Functions

Use a class when the thing has state, lifecycle, identity, or a stable public
contract. Good examples are `LLM`, `Flow`, `ToolGroup`, `Graph`, `GraphStore`,
`KnowledgeRegistry`, providers, backends, and middleware-like objects.

Use a dataclass when the thing is a data contract that moves through the
system. Good examples are responses, usage, tool calls, policies, results,
steps, nodes, edges, and flow events.

Use a function when the thing is a transformation, factory, helper, recipe, or
tool. Good examples are `create_provider()`, `infer_schema()`, `run_tools()`,
`react_flow()`, `react_initial_state()`, and toolkit functions decorated with
`@tool`.

Avoid adding a class just to group helper functions. Prefer a small internal
function until there is real state, lifecycle, polymorphism, or a public
contract worth naming.

## Docstrings

Docstrings are Google-style and should describe behavior, not repeat type
annotations. Types live in the signature.

Use docstrings for:

- Modules with a short purpose statement.
- Public classes, methods, functions, and factories.
- Internal helpers whose contract is not obvious from the signature.
- Every `@tool` function, because the docstring helps generate the model-facing
  tool schema.

Skip docstrings for tiny private helpers when the name and signature already
explain the contract.

Good shape:

```python
def search_documents(query: str, limit: int = 5) -> list[DocumentHit]:
    """Search indexed documents by query text.

    Args:
        query: Search phrase to match.
        limit: Maximum number of hits to return.
    """
```

For `@tool`, treat the docstring as runtime API text for the model. The first
sentence should say what the tool does. `Args:` should explain what each value
means in user terms. Avoid implementation details the model cannot act on.

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city using Open-Meteo.

    Args:
        city: City name, for example "Tokyo" or "London".
    """
```

Use `Returns:` only when the return value has semantics that are not obvious
from the function name and annotation. Do not add it mechanically.

## Comments

Comments should explain why a branch exists, which invariant is being
protected, or which provider/runtime quirk is being handled. They should not
describe the next line in plainer English.

Good comments in this repo usually cover:

- Provider-specific API differences.
- Budget, metering, and terminal error behavior.
- Stream lifecycle and concurrency edge cases.
- Non-obvious compatibility choices.
- Section breaks in large modules, using a light form such as
  `# --- Registration ---`.

Avoid comments like `# increment counter` or `# return result`. If the code is
hard to read without that kind of comment, rename variables or extract a helper.

## Self-Check Questions

Before adding or changing code, ask:

- Does this belong in `core`, `toolkit`, or `nanope`?
- Is this a public contract, a data contract, or just a transformation?
- Should this be a class, a frozen dataclass, or a function?
- Is the public API exported through the nearest `__init__.py` and `__all__`?
- Can Ruff format this cleanly without manual alignment?
- Does the docstring describe behavior and arguments without repeating types?
- If this is a `@tool`, would an agent understand when and how to call it?
- If this is a toolkit tool, does it return an error string instead of raising?
- Are comments explaining invariants and edge cases rather than restating code?
- Are tests placed near the matching subsystem and using the established
  fixtures/mocking style?
