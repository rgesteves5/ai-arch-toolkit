"""20 — Batch API (OpenAI).

Submit a small batch, poll status, and fetch parsed results.
"""

import os
import time

from ai_arch_toolkit import BatchClient, BatchRequest, Message

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Set OPENAI_API_KEY to run this batch example.")

client = BatchClient("openai", model="gpt-5-nano", api_key=api_key)

requests = [
    BatchRequest(custom_id="q1", messages=[Message(role="user", content="What is 2 + 2?")]),
    BatchRequest(
        custom_id="q2",
        messages=[Message(role="user", content="Name one moon of Jupiter.")],
    ),
]

job = client.submit(requests)
print(f"Submitted batch: {job.id} (status={job.status})")

max_polls = 30  # ~60s total at 2s interval
polls = 0
while polls < max_polls and job.status not in {"completed", "ended", "failed", "canceled"}:
    time.sleep(2)
    job = client.status(job)
    print(f"Batch status: {job.status}")
    polls += 1

if polls >= max_polls and job.status not in {"completed", "ended", "failed", "canceled"}:
    print("Batch still running; stop here and check status later with the saved batch ID.")
    raise SystemExit(0)

if job.status in {"completed", "ended"}:
    results = client.results(job)
    print("\nResults:")
    for item in results:
        if item.response is not None:
            print(f"  {item.custom_id}: {item.response.text}")
        else:
            print(f"  {item.custom_id}: ERROR -> {item.error}")
else:
    print("Batch finished without successful completion:", job.status)
