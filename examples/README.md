# ai-arch-toolkit Examples

Runnable examples showcasing every major feature of the toolkit.

## Prerequisites

```bash
# Install the package in editable mode
uv pip install -e .
```

## Environment Variables

Set API keys for the providers you want to use:

| Variable            | Provider  | Required by                          |
|---------------------|-----------|--------------------------------------|
| `OPENAI_API_KEY`    | OpenAI    | 01, 04, 10, 14                       |
| `ANTHROPIC_API_KEY` | Anthropic | 02, 05, 07, 13                       |
| `XAI_API_KEY`       | xAI       | 06, 09, 12                           |
| `GEMINI_API_KEY`    | Gemini    | 03, 08, 11                           |

## Example Index

| #  | File                            | Provider  | Feature                              |
|----|---------------------------------|-----------|--------------------------------------|
| 01 | `01_hello_world.py`             | OpenAI    | Basic Client chat                    |
| 02 | `02_multi_turn_conversation.py` | Anthropic | Message history, system prompt       |
| 03 | `03_streaming.py`               | Gemini    | stream() + stream_events()           |
| 04 | `04_structured_output.py`       | OpenAI    | JSON schema enforcement              |
| 05 | `05_tool_calling.py`            | Anthropic | Manual Tool / ToolCall / ToolResult  |
| 06 | `06_tool_registry.py`           | xAI       | @tool decorator + ToolRegistry       |
| 07 | `07_thinking.py`                | Anthropic | ThinkingConfig (effort + budget)     |
| 08 | `08_multimodal.py`              | Gemini    | ImagePart with URL                   |
| 09 | `09_server_tools.py`            | xAI       | xai-responses provider, ServerTool   |
| 10 | `10_react_agent.py`             | OpenAI    | ReActAgent + events + tools          |
| 11 | `11_plan_execute_agent.py`      | Gemini    | PlanExecuteAgent                     |
| 12 | `12_tree_of_thoughts_agent.py`  | xAI       | TreeOfThoughts (reasoning-only)      |
| 13 | `13_self_discovery_agent.py`    | Anthropic | SelfDiscovery 4-phase reasoning      |
| 14 | `14_async_client.py`            | OpenAI    | AsyncClient + parallel requests      |

## Running

```bash
# Run any example directly
python examples/01_hello_world.py

# Or from the examples directory
cd examples && python 01_hello_world.py
```

> **Cost note:** These examples make real API calls and will incur charges
> on your provider accounts. All examples use cost-effective models to
> minimise spend.
