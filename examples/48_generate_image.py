"""48 — Image Generation (generate, edit, and draw inside a turn).

An image model draws from a prompt and edits by a prompt and input images, through
``LLM.generate_image()``; the images come back as bytes with their MIME type, and the
call is metered like any other. A chat model on OpenAI's host can also draw in the
middle of its turn with the hosted ``image_generation()`` tool.

Image models: OpenAI's GPT Image (``gpt-image-2.5-flare``…), Gemini's image models (a
paid project), xAI's ``grok-imagine-image-2.0``, Meta's ``muse-image-1.0``.
"""

from pathlib import Path

from ai_arch_toolkit import LLM, image, image_generation

out = Path("generated")
out.mkdir(exist_ok=True)

# 1. Generate: the portable options become each provider's own (here a 1360x768 size).
painter = LLM("gpt-image-2.5-flare")
drawn = painter.generate_image_sync(
    "A small red lighthouse on a rocky shore at sunset, flat illustration",
    aspect_ratio="16:9",
    quality="low",
)
picture = drawn.images[0]
(out / "lighthouse.png").write_bytes(picture.data)
print(f"Drew {picture.media_type}, {len(picture.data)} bytes, ${drawn.cost:.4f}")

# 2. Edit: the prompt says what to change in the input image.
edited = painter.generate_image_sync(
    "Make the sky purple and keep everything else", images=[image(picture.data)], quality="low"
)
(out / "lighthouse-purple.png").write_bytes(edited.images[0].data)
print(f"Edited, ${edited.cost:.4f}")

# 3. Draw inside a turn: the chat model decides to draw, next to its answer.
chat = LLM("gpt-5-nano", max_tokens=8000)
turn = chat.complete_sync(
    "Draw a blue sailboat, then describe it in one sentence.",
    tools=[image_generation(model="gpt-image-2.5-flare", quality="low")],
)
for index, drawn_image in enumerate(turn.images):
    (out / f"sailboat-{index}.png").write_bytes(drawn_image.data)
print("Answer:", turn.text, f"(${turn.cost:.4f}, image included)")
