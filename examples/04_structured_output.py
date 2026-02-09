"""04 — Structured Output (OpenAI).

Use JsonSchema to enforce a strict response format and parse the result.
"""

import json

from ai_arch_toolkit import Client, JsonSchema

client = Client("openai", model="gpt-5-nano")

schema = JsonSchema(
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

response = client.chat(
    "Recommend 3 classic science fiction novels.",
    json_schema=schema,
)

data = json.loads(response.text)
for book in data["books"]:
    print(f"  {book['title']} by {book['author']} ({book['year']})")
    print(f"    → {book['reason']}\n")
