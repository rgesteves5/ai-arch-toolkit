"""14 — Async Client (OpenAI).

Demonstrates AsyncClient with:
  - Single async chat
  - Async streaming
  - Parallel requests with asyncio.gather()
"""

import asyncio

from ai_arch_toolkit import AsyncClient


async def main():
    client = AsyncClient("openai", model="gpt-5-nano")

    # --- Single async request ---
    print("=== Async Chat ===")
    resp = await client.chat("What is 2 + 2? Reply in one word.")
    print(f"Answer: {resp.text}\n")

    # --- Async streaming ---
    print("=== Async Streaming ===")
    async for chunk in client.stream("Count from 1 to 5, one number per line."):
        print(chunk, end="", flush=True)
    print("\n")

    # --- Parallel requests ---
    print("=== Parallel Requests (3 concurrent) ===")
    questions = [
        "Name one planet in our solar system.",
        "Name one programming language.",
        "Name one chemical element.",
    ]
    responses = await asyncio.gather(*(client.chat(q) for q in questions))

    for question, resp in zip(questions, responses, strict=True):
        print(f"  Q: {question}")
        print(f"  A: {resp.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
