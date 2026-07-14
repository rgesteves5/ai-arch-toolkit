# Prompt Messages & Content

`toolkit.prompts` can keep a prompt as an ordered conversation instead of flattening
everything into one system string. `PromptMessage` accepts literal text, a resolved `Prompt`,
an unresolved `PromptTemplate`, or the core `Content` type (text plus multimodal parts).

```python
from ai_arch_toolkit import Prompt, PromptConversation, PromptMessage

conversation = PromptConversation(
    messages=(
        PromptMessage(role="system", content=Prompt.from_text("You are a writer.")),
        PromptMessage(role="user", content="Write a short opening."),
    )
)

rendered = conversation.render()
messages, system = rendered.to_llm_request()
response = llm.complete_sync(messages, system=system)
```

The conversation is rendered deterministically. `RenderedPromptConversation` contains one
`RenderedPromptMessage` per input message and a SHA-256 fingerprint over role and content.
Prompt-backed messages retain the rendered prompt and its section provenance.

## Multimodal messages

Core content parts remain provider-agnostic:

```python
from ai_arch_toolkit import PromptConversation, PromptMessage, document, image

conversation = PromptConversation(
    messages=(
        PromptMessage(
            role="user",
            content=[
                "Compare these files.",
                image(image_bytes, "image/png"),
                document(pdf_bytes, name="brief.pdf"),
            ],
        ),
    )
)
messages, system = conversation.render().to_llm_request()
```

At most one textual `system` message is extracted by `to_llm_request()`. User and assistant
messages stay in order and keep their multimodal parts. A non-text system message is rejected
because the core LLM facade currently models `system` as text.

`PromptConversation` is a composition utility; it does not call an LLM, choose a provider, or
activate prompt caching. Pass its plain `(messages, system)` result to `LLM` and configure
provider-specific behavior there.
