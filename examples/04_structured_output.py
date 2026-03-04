"""04 — Structured Output.

Use OutputSchema to enforce a strict JSON response format.
The parsed result is available via response.parsed.
"""

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import OutputSchema

llm = LLM("gpt-4.1-nano")

schema = OutputSchema(
    name="book_recommendations",
    schema={
        "type": "object",
        "properties": {
            "books": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "year": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["title", "author", "year", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["books"],
        "additionalProperties": False,
    },
)

response = llm.complete_sync(
    "Recommend 3 classic science fiction novels.",
    output_schema=schema,
)

if response.parsed:
    for book in response.parsed["books"]:
        print(f"  {book['title']} by {book['author']} ({book['year']})")
        print(f"    → {book['reason']}\n")
else:
    print("Raw:", response.text)
