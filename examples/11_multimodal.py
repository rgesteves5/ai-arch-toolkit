"""11 — Multimodal (Image + Text).

Send an image alongside text using the image() helper.
Works with vision-capable models (GPT-4o, Claude, Gemini).
"""

from ai_arch_toolkit.core import LLM, image, user

llm = LLM("gpt-4.1-nano")

# --- Image from URL ---
print("=== Image from URL ===")
messages = [
    user(
        [
            "What do you see in this image? Reply in one sentence.",
            image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
            ),
        ]
    ),
]

response = llm.complete_sync(messages)
print(f"Answer: {response.text}")
print(f"Tokens — in: {response.usage.input_tokens}, out: {response.usage.output_tokens}")
