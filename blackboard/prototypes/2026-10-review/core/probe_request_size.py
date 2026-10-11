"""The request size the worst case reads counts the repr of each assistant turn's ``_raw``."""

from __future__ import annotations

from google.genai import types

from ai_arch_toolkit.core._attempts import _request_size
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._gemini import _parse_sdk_response

png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4096  # ~1 MB image the model drew
raw = types.GenerateContentResponse(
    candidates=[
        types.Candidate(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Here it is."),
                    types.Part(inline_data=types.Blob(data=png, mime_type="image/png")),
                ],
            )
        )
    ]
)
turn = _parse_sdk_response(raw, "gemini-3.1-flash-image").to_message()
messages = [{"role": "user", "content": "draw a cat"}, turn, {"role": "user", "content": "bluer"}]
chars, media = _request_size(Request(messages=messages, system=None, tools=None, model="m"))
print(f"image bytes {len(png):,}; counted chars {chars:,} (~{chars // 4:,} tokens held); media parts {media}")
without = [{k: v for k, v in m.items() if k != "_raw"} for m in messages]
chars2, _ = _request_size(Request(messages=without, system=None, tools=None, model="m"))
print(f"without _raw: {chars2:,} chars")
