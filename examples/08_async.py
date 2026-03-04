"""08 — Async Usage.

LLM is async-first. Use complete() and stream() directly in async code.
Parallel requests are natural with asyncio.gather().
"""

import asyncio

from ai_arch_toolkit import LLM


async def main():
    async with LLM("gpt-4.1-nano") as llm:
        # --- Single async request ---
        print("=== Async Complete ===")
        resp = await llm.complete("What is 2 + 2? Reply in one word.")
        print(f"Answer: {resp.text}\n")

        # --- Async streaming ---
        print("=== Async Streaming ===")
        stream = llm.stream("Count from 1 to 5, one number per line.")
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print(f"\n[Tokens: {stream.response.usage.output_tokens}]\n")

        # --- Parallel requests ---
        print("=== Parallel Requests (3 concurrent) ===")
        questions = [
            "Name one planet in our solar system.",
            "Name one programming language.",
            "Name one chemical element.",
        ]
        responses = await asyncio.gather(*(llm.complete(q) for q in questions))

        for question, resp in zip(questions, responses, strict=True):
            print(f"  Q: {question}")
            print(f"  A: {resp.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
