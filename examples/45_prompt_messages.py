"""45 — Compose ordered prompt messages, including multimodal Content. No API key required."""

from __future__ import annotations

from ai_arch_toolkit import Prompt, PromptConversation, PromptMessage, document, image

conversation = PromptConversation(
    messages=(
        PromptMessage(role="system", content=Prompt.from_text("You are a careful reviewer.")),
        PromptMessage(
            role="user",
            content=[
                "Review this brief and its diagram.",
                image(b"fake-image", "image/png"),
                document(b"fake-pdf", name="brief.pdf"),
            ],
        ),
    )
)

rendered = conversation.render()
messages, system = rendered.to_llm_request()
print(f"system: {system}")
print(f"messages: {len(messages)}")
print(f"fingerprint: {rendered.fingerprint}")
