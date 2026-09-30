# Content & Messages

Helpers for building messages and multimodal content. A message's content is `Content` — a plain string or a list of typed parts — so the same call site handles text, images, PDFs, and prompt caching. `LLM` calls take a string or a list of messages; agent flows take `Content` as their task (see the end of this page).

`Content` is the provider-input contract. It is distinct from a
[Resource](resources.md), which is a loaded application asset, and from a
[Prompt](prompts.md), which organizes resolved instruction sections. Render a prompt to text
before passing it as the `system` argument. Files that should be sent natively to a provider
remain `DocumentPart`/`ImagePart`; files used to construct instructions are Resources.

## Message constructors

```python
from ai_arch_toolkit import user, assistant, system, tool_result

messages = [
    system("You are a helpful assistant."),
    user("What's the weather?"),
    assistant("Let me check that for you."),
    tool_result("22°C and sunny", tool_use_id="call_123", name="get_weather"),
]
```

## Multimodal content

```python
from pathlib import Path

from ai_arch_toolkit import user, image, document, cache

# Image (URL, base64, or raw bytes)
messages = [user(["Describe this image:", image("https://example.com/photo.jpg")])]
messages = [user(["Describe this:", image(raw_bytes, media_type="image/png")])]

# PDF document (base64 or raw bytes; a file path is not read)
pdf = Path("report.pdf").read_bytes()
messages = [user(["Summarize this:", document(pdf, name="report.pdf")])]

# Anthropic prompt caching
messages = [user([cache(long_context), "Now answer my question."])]

# A cached system prompt (Anthropic)
messages = [{"role": "system", "content": [cache(long_instructions)]}, user("Hi")]
```

A `cache()` part only changes the request on Anthropic. The other providers receive its text as
ordinary text, so the same messages work everywhere.

Helper signatures:

```python
image(source: str | bytes, media_type="image/png") -> ImagePart
document(source: str | bytes, media_type="application/pdf", name=None) -> DocumentPart
cache(content: str) -> CachePart
```

## Content type

```python
type ContentPart = str | ImagePart | DocumentPart | CachePart
type Content = str | list[ContentPart]
```

Every agent strategy takes `Content` as its task input, but only `react` and `completion` send it
to the model as content parts, so images and documents reach the model through those two, and
vision plus tools through `react`. The other strategies turn a non-string task into text
(`str(task)`). See [Flow Architecture](flow-architecture.md).
