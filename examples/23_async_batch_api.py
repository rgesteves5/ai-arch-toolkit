"""23 — Async Batch API (OpenAI).

Submit an async batch, poll status, then fetch parsed results.
"""

import asyncio
import os

from ai_arch_toolkit import AsyncBatchClient, BatchRequest, Message


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY to run this async batch example.")

    client = AsyncBatchClient("openai", model="gpt-5-nano", api_key=api_key)
    requests = [
        BatchRequest(
            custom_id="q1",
            messages=[Message(role="user", content="Give one synonym for fast.")],
        ),
        BatchRequest(
            custom_id="q2",
            messages=[Message(role="user", content="What is 12 * 9?")],
        ),
    ]

    job = await client.submit(requests)
    print(f"Submitted batch: {job.id} (status={job.status})")

    max_polls = 30
    polls = 0
    terminal_states = {"completed", "ended", "failed", "canceled"}
    while polls < max_polls and job.status not in terminal_states:
        await asyncio.sleep(2)
        job = await client.status(job)
        print(f"Batch status: {job.status}")
        polls += 1

    if polls >= max_polls and job.status not in terminal_states:
        print("Batch still running; check status later using this batch ID.")
        return

    if job.status in {"completed", "ended"}:
        results = await client.results(job)
        print("\nResults:")
        for item in results:
            if item.response is not None:
                print(f"  {item.custom_id}: {item.response.text}")
            else:
                print(f"  {item.custom_id}: ERROR -> {item.error}")
    else:
        print("Batch finished without successful completion:", job.status)


if __name__ == "__main__":
    asyncio.run(main())
