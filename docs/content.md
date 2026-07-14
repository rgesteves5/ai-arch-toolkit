# Content & Messages

Helpers for building messages and multimodal content. Every `LLM` and agent flow accepts `Content` — a plain string or a list of typed parts — so the same call site handles text, images, PDFs, and prompt caching.

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
from ai_arch_toolkit import user, image, document, cache

# Image (URL, base64, or raw bytes)
messages = [user(["Describe this image:", image("https://example.com/photo.jpg")])]
messages = [user(["Describe this:", image(raw_bytes, media_type="image/png")])]

# PDF document
messages = [user(["Summarize this:", document("report.pdf", media_type="application/pdf")])]

# Anthropic prompt caching
messages = [user([cache(long_context), "Now answer my question."])]
```

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

Because all agents accept `Content` as their task input, you can pass images and documents to **any** agent flow — enabling vision + tools use cases. See [Flow Architecture](flow-architecture.md).
