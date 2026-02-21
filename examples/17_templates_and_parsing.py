"""17 — Prompt Templates + Output Parsing (OpenAI).

Use PromptTemplate/ChatTemplate to format prompts and parse
JSON output using parse_json_as.
"""

from dataclasses import dataclass

from ai_arch_toolkit import (
    ChatTemplate,
    Client,
    PromptTemplate,
    parse_json_as,
)


@dataclass(frozen=True, slots=True)
class Book:
    title: str
    author: str
    year: int


client = Client("openai", model="gpt-5-nano")

prompt = PromptTemplate(
    "Recommend one {genre} book and return JSON with keys: title, author, year."
)
response = client.chat(prompt.format(genre="science fiction"))
book = parse_json_as(response.text, Book)
print("From PromptTemplate:", book)

chat_template = ChatTemplate.from_tuples(
    [
        ("system", "You are a strict JSON assistant."),
        (
            "user",
            "Return only JSON with keys title, author, year for one classic {genre} novel.",
        ),
    ]
)
messages = chat_template.format_messages(genre="mystery")
response2 = client.chat(messages)
book2 = parse_json_as(response2.text, Book)
print("From ChatTemplate:", book2)

