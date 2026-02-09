# ai-arch-toolkit

Lightweight unified LLM client for OpenAI, Anthropic, xAI, Gemini, Mistral, and Groq — plus a tool registry, 8 agent architectures, and batch APIs.

## Quick Start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> && cd ai-arch-toolkit
uv sync --dev

# Run tests
uv run pytest

# Run an example
uv run python examples/01_hello_world.py
```

See [docs/uv-guide.md](docs/uv-guide.md) for detailed development workflow.
