# API Documentation

The package includes full module and class docstrings for public APIs in:

- `ai_arch_toolkit`
- `ai_arch_toolkit.llm`
- `ai_arch_toolkit.agents`
- `ai_arch_toolkit.tools`

## Auto-generate API docs with pdoc

Generate HTML API docs:

```bash
uv sync --group docs
uv run pdoc ai_arch_toolkit -o site/api
```

Serve generated docs:

```bash
uv run pdoc ai_arch_toolkit --http :8080
```

