#!/usr/bin/env python3
"""Front I, task I01: the image APIs of OpenAI, Gemini, xAI and Meta, live.

Records how each provider answers what D46 and D47 leave to measurement
(``blackboard/tasks/I01-image-probe.md``):

1. does the Images API report ``usage`` on the newer GPT Image models;
2. where the Responses ``image_generation`` tool bills its image;
3. does a stateless edit work, replaying the ``image_generation_call`` item;
4. Gemini's image models: usage by modality, thought images, signatures on image parts, streaming,
   ``candidate_count``, ``image_config``;
5. OpenAI's sizes, ``n`` and output formats;
6. xAI's ``image.sample``: base64, the reported cost, an edit from a data URL;
7. Meta's ``muse-image-1.0`` on the Responses API.

It drives the SDKs directly, since the adapters are what I03 builds, and it costs money: each
check declares its worst case and is skipped when that no longer fits ``--max-cost``. Every image
is asked at the cheapest quality and size. Run it from the repository root:

    set -a; source .env; set +a
    uv run python scripts/probe_images.py --max-cost 2
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("scripts/output/model-probes")
PROMPT = "A small red lighthouse on a rocky shore at sunset, flat vector illustration."
DRAW = f"Draw this image: {PROMPT}"
EDIT = "Now make the sky purple and keep everything else the same."
META_BASE_URL = "https://api.meta.ai/v1"

OPENAI_IMAGE = "gpt-image-2.5-flare"
OPENAI_MAINLINE = "gpt-5-nano"
GEMINI_FLASH = "gemini-3.1-flash-image"
GEMINI_LITE = "gemini-3.1-flash-lite-image"
XAI_V1 = "grok-imagine-image"
XAI_V2 = "grok-imagine-image-2.0"
META_IMAGE = "muse-image-1.0"


@dataclass(frozen=True, slots=True, kw_only=True)
class Check:
    """One question, the worst it can cost, and the request that answers it."""

    number: str
    provider: str
    question: str
    worst_usd: float
    run: Callable[[Probe], str]


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """What one check found."""

    number: str
    provider: str
    question: str
    outcome: str
    seconds: float = 0.0
    skipped: bool = False


@dataclass(slots=True)
class Budget:
    """Worst cases charged so far against a cap: a check runs only when its worst case fits."""

    cap: float
    spent: float = 0.0

    def admits(self, worst: float) -> bool:
        return self.spent + worst <= self.cap

    def charge(self, worst: float) -> None:
        self.spent += worst


def redact(text: str, limit: int = 400) -> str:
    """An error message on one line, without organization ids or key fragments."""
    text = re.sub(r"\borg-[A-Za-z0-9]+", "org-<redacted>", text)
    text = re.sub(r"\b(sk|xai)-[A-Za-z0-9_*\-]+", r"\1-<redacted>", text)
    text = re.sub(r"\bAIza[0-9A-Za-z_\-]+", "AIza<redacted>", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sniff(data: bytes) -> str:
    """The media type the image bytes declare by their signature, or ``"unknown"``."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "unknown"


def dimensions(data: bytes) -> tuple[int, int] | None:
    """A PNG's or a WebP's width and height from its header, or ``None`` for any other format."""
    kind = sniff(data)
    if kind == "image/png" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if kind == "image/webp" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8X":  # 24-bit canvas width and height, minus one
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if chunk == b"VP8 ":  # lossy: 14-bit sizes after the frame tag and start code
            width, height = struct.unpack("<HH", data[26:30])
            return width & 0x3FFF, height & 0x3FFF
        if chunk == b"VP8L":  # lossless: 14 bits each, minus one, after the signature byte
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def dump(value: Any) -> str:
    """A pydantic model, a proto, or a plain value as compact JSON."""
    if value is None:
        return "None"
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    elif hasattr(value, "ListFields"):
        from google.protobuf.json_format import MessageToDict

        value = MessageToDict(value)
    return json.dumps(value, default=str, separators=(",", ":"))


def describe_image(data: bytes) -> str:
    size = dimensions(data)
    shape = f" {size[0]}x{size[1]}" if size else ""
    return f"{sniff(data)}{shape}, {len(data)} bytes"


@dataclass(slots=True)
class Probe:
    """The clients, the images kept for edits, and where the images are written."""

    out_dir: Path
    images: dict[str, tuple[bytes, str]] = field(default_factory=dict)
    kept: dict[str, Any] = field(default_factory=dict)
    _clients: dict[str, Any] = field(default_factory=dict)

    def save(self, name: str, data: bytes, media_type: str | None = None) -> None:
        media_type = media_type or sniff(data)
        suffix = media_type.rsplit("/", 1)[-1].replace("jpeg", "jpg")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / f"{name}.{suffix}").write_bytes(data)
        self.images.setdefault(name.split("-", 1)[0], (data, media_type))

    @property
    def openai(self) -> Any:
        if "openai" not in self._clients:
            import openai

            self._clients["openai"] = openai.OpenAI(timeout=300, max_retries=0)
        return self._clients["openai"]

    @property
    def meta(self) -> Any:
        if "meta" not in self._clients:
            import openai

            self._clients["meta"] = openai.OpenAI(
                base_url=META_BASE_URL,
                api_key=os.environ["MODEL_API_KEY"],
                timeout=300,
                max_retries=0,
            )
        return self._clients["meta"]

    @property
    def gemini(self) -> Any:
        if "gemini" not in self._clients:
            from google import genai

            key = os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
            self._clients["gemini"] = genai.Client(api_key=key)
        return self._clients["gemini"]

    @property
    def xai(self) -> Any:
        if "xai" not in self._clients:
            import xai_sdk

            self._clients["xai"] = xai_sdk.Client(api_key=os.environ["XAI_API_KEY"])
        return self._clients["xai"]


# ---------------------------------------------------------------------------
# OpenAI: the Images API (questions 1 and 5)
# ---------------------------------------------------------------------------


def _images_outcome(probe: Probe, name: str, response: Any) -> str:
    images = [base64.b64decode(item.b64_json) for item in response.data or []]
    for index, data in enumerate(images):
        probe.save(f"{name}-{index}" if index else name, data)
    shapes = "; ".join(describe_image(data) for data in images)
    revised = any(item.revised_prompt for item in response.data or [])
    return (
        f"{len(images)} image(s): {shapes}. usage={dump(response.usage)}. "
        f"output_format={response.output_format}, size={response.size}, "
        f"quality={response.quality}, revised_prompt={revised}"
    )


def openai_generate(model: str, **options: Any) -> Callable[[Probe], str]:
    def run(probe: Probe) -> str:
        asked = {"size": "1024x1024", "quality": "low", **options}
        response = probe.openai.images.generate(model=model, prompt=PROMPT, **asked)
        return _images_outcome(probe, f"openai-{model}", response)

    return run


def openai_edit(probe: Probe) -> str:
    source, media_type = probe.images["openai"]
    response = probe.openai.images.edit(
        model=OPENAI_IMAGE,
        image=("lighthouse.png", source, media_type),
        prompt=EDIT,
        size="1024x1024",
        quality="low",
    )
    return _images_outcome(probe, "openai-edit", response)


# ---------------------------------------------------------------------------
# OpenAI: the Responses image_generation tool (questions 2 and 3)
# ---------------------------------------------------------------------------

TOOL: dict[str, Any] = {
    "type": "image_generation",
    "model": OPENAI_IMAGE,
    "quality": "low",
    "size": "1024x1024",
}


def _tool_turn(probe: Probe, input_items: Any, **options: Any) -> Any:
    return probe.openai.responses.create(
        model=OPENAI_MAINLINE,
        input=input_items,
        tools=[TOOL],
        tool_choice={"type": "image_generation"},
        reasoning={"effort": "low"},
        store=False,
        **options,
    )


def _turn_outcome(probe: Probe, name: str, response: Any) -> str:
    kinds = [item.type for item in response.output]
    calls = [item for item in response.output if item.type == "image_generation_call"]
    notes = []
    for index, call in enumerate(calls):
        if call.result:
            data = base64.b64decode(call.result)
            probe.save(f"{name}-{index}" if index else name, data)
            notes.append(describe_image(data))
        fields = call.model_dump(exclude_none=True, exclude={"result"})
        notes.append(f"call={json.dumps(fields, default=str)}")
    whole = response.model_dump(mode="json")
    extra = {key: whole[key] for key in ("billing", "tool_usage") if key in whole}
    return (
        f"output={kinds}; {'; '.join(notes)}; usage={dump(response.usage)}; "
        f"billing and tool_usage={json.dumps(extra, default=str)}"
    )


def openai_tool_turn(probe: Probe) -> str:
    response = _tool_turn(probe, DRAW)
    probe.kept["openai-turn"] = response
    return _turn_outcome(probe, "openai-turn", response)


def openai_plain_turn(probe: Probe) -> str:
    response = probe.openai.responses.create(
        model=OPENAI_MAINLINE, input=DRAW, reasoning={"effort": "low"}, store=False
    )
    return f"output={[item.type for item in response.output]}; usage={dump(response.usage)}"


def _replayed(probe: Probe, *, keep_result: bool) -> list[Any]:
    first = probe.kept["openai-turn"]
    items: list[Any] = [{"role": "user", "content": DRAW}]
    for item in first.output:
        wire = item.model_dump(mode="json", exclude_none=True, by_alias=True)
        if item.type == "image_generation_call" and not keep_result:
            wire["result"] = None
        items.append(wire)
    items.append({"role": "user", "content": EDIT})
    return items


def openai_stateless_edit(keep_result: bool) -> Callable[[Probe], str]:
    def run(probe: Probe) -> str:
        response = _tool_turn(probe, _replayed(probe, keep_result=keep_result))
        name = "openai-replay" if keep_result else "openai-replay-null"
        return _turn_outcome(probe, name, response)

    return run


def openai_stateless_edit_as_input(probe: Probe) -> str:
    """The earlier turn replayed without its image call, the image sent back as an input."""
    first = probe.kept["openai-turn"]
    items: list[Any] = [{"role": "user", "content": DRAW}]
    image = ""
    for item in first.output:
        if item.type == "image_generation_call":
            image = item.result or ""
            continue
        items.append(item.model_dump(mode="json", exclude_none=True, by_alias=True))
    items.append(
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
                {"type": "input_text", "text": EDIT},
            ],
        }
    )
    response = _tool_turn(probe, items)
    return _turn_outcome(probe, "openai-replay-input", response)


def openai_images_stream(probe: Probe) -> str:
    counts: dict[str, int] = {}
    shapes: list[str] = []
    stream = probe.openai.images.generate(
        model=OPENAI_IMAGE,
        prompt=PROMPT,
        size="1024x1024",
        quality="low",
        partial_images=2,
        stream=True,
    )
    for event in stream:
        counts[event.type] = counts.get(event.type, 0) + 1
        encoded = getattr(event, "b64_json", None)
        if encoded:
            shapes.append(f"{event.type}: {describe_image(base64.b64decode(encoded))}")
        if event.type == "image_generation.completed":
            shapes.append(f"usage={dump(getattr(event, 'usage', None))}")
    return f"events={json.dumps(counts)}; {'; '.join(shapes)}"


def openai_tool_stream(probe: Probe) -> str:
    counts: dict[str, int] = {}
    partial_shapes: list[str] = []
    final: Any = None
    stream = probe.openai.responses.create(
        model=OPENAI_MAINLINE,
        input=DRAW,
        tools=[{**TOOL, "partial_images": 2}],
        tool_choice={"type": "image_generation"},
        reasoning={"effort": "low"},
        store=False,
        stream=True,
    )
    for event in stream:
        counts[event.type] = counts.get(event.type, 0) + 1
        if event.type == "response.image_generation_call.partial_image":
            partial_shapes.append(describe_image(base64.b64decode(event.partial_image_b64)))
        elif event.type == "response.completed":
            final = event.response
    return (
        f"events={json.dumps(counts)}; partials=[{'; '.join(partial_shapes)}]; "
        f"final usage={dump(final.usage if final else None)}"
    )


# ---------------------------------------------------------------------------
# Gemini (question 4)
# ---------------------------------------------------------------------------


def _parts_outcome(probe: Probe, name: str, response: Any) -> str:
    candidates = response.candidates or []
    notes = []
    for c_index, candidate in enumerate(candidates):
        parts = (candidate.content.parts if candidate.content else None) or []
        for p_index, part in enumerate(parts):
            signed = bool(part.thought_signature)
            if part.inline_data and part.inline_data.data:
                data = part.inline_data.data
                if not part.thought:
                    probe.save(f"{name}-{c_index}-{p_index}", data, part.inline_data.mime_type)
                notes.append(
                    f"[{c_index}.{p_index}] image thought={bool(part.thought)} signed={signed} "
                    f"mime={part.inline_data.mime_type} ({describe_image(data)})"
                )
            elif part.text is not None:
                notes.append(
                    f"[{c_index}.{p_index}] text thought={bool(part.thought)} signed={signed} "
                    f"{len(part.text)} chars"
                )
    finish = [c.finish_reason.value if c.finish_reason else None for c in candidates]
    return (
        f"candidates={len(candidates)} finish={finish}; parts: {' | '.join(notes)}; "
        f"usage={dump(response.usage_metadata)}; model_version={response.model_version}"
    )


def _gemini_config(**options: Any) -> Any:
    from google.genai import types

    image = {key: options.pop(key) for key in list(options) if key.startswith("image_")}
    image_config = types.ImageConfig(**{k.removeprefix("image_"): v for k, v in image.items()})
    return types.GenerateContentConfig(image_config=image_config, **options)


def gemini_generate(model: str, name: str, **options: Any) -> Callable[[Probe], str]:
    def run(probe: Probe) -> str:
        response = probe.gemini.models.generate_content(
            model=model, contents=PROMPT, config=_gemini_config(**options)
        )
        probe.kept[name] = response
        return _parts_outcome(probe, name, response)

    return run


def gemini_replay(strip_signatures: bool) -> Callable[[Probe], str]:
    def run(probe: Probe) -> str:
        from google.genai import types

        first = probe.kept["gemini-flash"]
        content = first.candidates[0].content
        if strip_signatures:
            parts = [
                part.model_copy(update={"thought_signature": None}) if part.inline_data else part
                for part in content.parts
            ]
            content = types.Content(role="model", parts=parts)
        contents = [
            types.Content(role="user", parts=[types.Part(text=PROMPT)]),
            content,
            types.Content(role="user", parts=[types.Part(text=EDIT)]),
        ]
        config = _gemini_config(response_modalities=["TEXT", "IMAGE"], image_image_size="512")
        response = probe.gemini.models.generate_content(
            model=GEMINI_FLASH, contents=contents, config=config
        )
        name = "gemini-replay-unsigned" if strip_signatures else "gemini-replay"
        return _parts_outcome(probe, name, response)

    return run


def gemini_edit_input(probe: Probe) -> str:
    from google.genai import types

    source, media_type = probe.images["gemini"]
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(inline_data=types.Blob(data=source, mime_type=media_type)),
                types.Part(text=EDIT),
            ],
        )
    ]
    response = probe.gemini.models.generate_content(
        model=GEMINI_LITE, contents=contents, config=_gemini_config(response_modalities=["IMAGE"])
    )
    return _parts_outcome(probe, "gemini-edit", response)


def gemini_stream(probe: Probe) -> str:
    chunks = []
    for chunk in probe.gemini.models.generate_content_stream(
        model=GEMINI_LITE, contents=PROMPT, config=_gemini_config(response_modalities=["IMAGE"])
    ):
        parts = []
        for candidate in chunk.candidates or []:
            for part in (candidate.content.parts if candidate.content else None) or []:
                if part.inline_data and part.inline_data.data:
                    kind = "thought-image" if part.thought else "image"
                    parts.append(f"{kind}({describe_image(part.inline_data.data)})")
                elif part.text is not None:
                    parts.append(f"text({len(part.text)})")
        chunks.append(f"[{', '.join(parts)}] usage={dump(chunk.usage_metadata)}")
    return f"{len(chunks)} chunks: " + " | ".join(chunks)


# ---------------------------------------------------------------------------
# xAI (question 6)
# ---------------------------------------------------------------------------


def _xai_outcome(probe: Probe, name: str, responses: Sequence[Any]) -> str:
    notes = []
    for index, response in enumerate(responses):
        encoded = response.base64
        data = base64.b64decode(encoded.split(",", 1)[-1])
        probe.save(f"{name}-{index}" if index else name, data)
        notes.append(
            f"{describe_image(data)}, base64 prefix={encoded[:22]!r}, "
            f"cost_usd={response.cost_usd}, model={response.model}, "
            f"respect_moderation={response.respect_moderation}, usage={dump(response.usage)}"
        )
    return f"{len(responses)} image(s): " + " | ".join(notes)


def xai_sample(model: str, **options: Any) -> Callable[[Probe], str]:
    def run(probe: Probe) -> str:
        response = probe.xai.image.sample(
            prompt=PROMPT, model=model, image_format="base64", **options
        )
        return _xai_outcome(probe, f"xai-{model}", [response])

    return run


def xai_edit(probe: Probe) -> str:
    source, media_type = probe.images["xai"]
    response = probe.xai.image.sample(
        prompt=EDIT, model=XAI_V1, image_format="base64", image_url=data_url(source, media_type)
    )
    return _xai_outcome(probe, "xai-edit", [response])


def xai_batch(probe: Probe) -> str:
    responses = probe.xai.image.sample_batch(
        prompt=PROMPT, model=XAI_V1, n=2, image_format="base64"
    )
    return _xai_outcome(probe, "xai-batch", list(responses))


# ---------------------------------------------------------------------------
# Meta (question 7)
# ---------------------------------------------------------------------------


def _meta_outcome(probe: Probe, name: str, response: Any) -> str:
    kinds = [item.type for item in response.output]
    notes = []
    for index, item in enumerate(response.output):
        if item.type != "image_generation_call":
            continue
        result = getattr(item, "result", None)
        if result:
            data = base64.b64decode(result)
            probe.save(f"{name}-{index}", data)
            notes.append(describe_image(data))
        notes.append(f"id length={len(item.id)} status={getattr(item, 'status', None)}")
    return f"output={kinds}; {'; '.join(notes)}; usage={dump(response.usage)}"


def meta_turn(name: str, **options: Any) -> Callable[[Probe], str]:
    def run(probe: Probe) -> str:
        response = probe.meta.responses.create(
            model=META_IMAGE, input=PROMPT, store=False, **options
        )
        probe.kept.setdefault("meta-turn", response)
        return _meta_outcome(probe, name, response)

    return run


def meta_stateless_edit(probe: Probe) -> str:
    first = probe.kept["meta-turn"]
    items: list[Any] = [{"role": "user", "content": PROMPT}]
    for item in first.output:
        if item.type == "image_generation_call":
            items.append(
                {
                    "type": "image_generation_call",
                    "id": item.id,
                    "status": "completed",
                    "result": None,
                }
            )
        elif item.type == "message":
            items.append(item.model_dump(mode="json", exclude_none=True, by_alias=True))
    items.append({"role": "user", "content": EDIT})
    response = probe.meta.responses.create(model=META_IMAGE, input=items, store=False)
    return _meta_outcome(probe, "meta-replay", response)


def meta_images_endpoint_sized(size: str) -> Callable[[Probe], str]:
    def run(probe: Probe) -> str:
        response = probe.meta.images.generate(
            model=META_IMAGE, prompt=PROMPT, n=1, size=size, response_format="b64_json"
        )
        return _images_outcome(probe, f"meta-images-{size}", response)

    return run


def meta_images_endpoint(probe: Probe) -> str:
    response = probe.meta.images.generate(
        model=META_IMAGE, prompt=PROMPT, n=1, response_format="b64_json"
    )
    return _images_outcome(probe, "meta-images", response)


# ---------------------------------------------------------------------------
# The checks, in order
# ---------------------------------------------------------------------------

CHECKS: tuple[Check, ...] = (
    Check(
        number="1a",
        provider="openai",
        question=f"{OPENAI_IMAGE}: usage?",
        worst_usd=0.03,
        run=openai_generate(OPENAI_IMAGE),
    ),
    Check(
        number="1b",
        provider="openai",
        question="gpt-image-2.5-sunburst: usage?",
        worst_usd=0.03,
        run=openai_generate("gpt-image-2.5-sunburst"),
    ),
    Check(
        number="1c",
        provider="openai",
        question="gpt-image-2: usage?",
        worst_usd=0.03,
        run=openai_generate("gpt-image-2"),
    ),
    Check(
        number="1d",
        provider="openai",
        question="gpt-image-1-mini: usage?",
        worst_usd=0.02,
        run=openai_generate("gpt-image-1-mini"),
    ),
    Check(
        number="5a",
        provider="openai",
        question="gpt-image-1.5 with size 1536x864?",
        worst_usd=0.03,
        run=openai_generate("gpt-image-1.5", size="1536x864"),
    ),
    Check(
        number="5b",
        provider="openai",
        question=f"{OPENAI_IMAGE} with size 1536x864?",
        worst_usd=0.03,
        run=openai_generate(OPENAI_IMAGE, size="1536x864"),
    ),
    Check(
        number="5c",
        provider="openai",
        question=f"{OPENAI_IMAGE} with n=2?",
        worst_usd=0.06,
        run=openai_generate(OPENAI_IMAGE, n=2),
    ),
    Check(
        number="5d",
        provider="openai",
        question=f"{OPENAI_IMAGE} with output_format=webp?",
        worst_usd=0.03,
        run=openai_generate(OPENAI_IMAGE, output_format="webp"),
    ),
    Check(
        number="5e",
        provider="openai",
        question=f"{OPENAI_IMAGE} edit of the 1a image?",
        worst_usd=0.06,
        run=openai_edit,
    ),
    Check(
        number="2a",
        provider="openai",
        question=f"Responses tool turn on {OPENAI_MAINLINE}: where is the image billed?",
        worst_usd=0.06,
        run=openai_tool_turn,
    ),
    Check(
        number="2b",
        provider="openai",
        question="The same turn without the tool (baseline)",
        worst_usd=0.01,
        run=openai_plain_turn,
    ),
    Check(
        number="2c",
        provider="openai",
        question="The tool turn streamed, partial_images=2",
        worst_usd=0.06,
        run=openai_tool_stream,
    ),
    Check(
        number="3a",
        provider="openai",
        question="Stateless edit: the image_generation_call replayed with its result",
        worst_usd=0.08,
        run=openai_stateless_edit(keep_result=True),
    ),
    Check(
        number="3b",
        provider="openai",
        question="Stateless edit: the image_generation_call replayed with result null",
        worst_usd=0.08,
        run=openai_stateless_edit(keep_result=False),
    ),
    Check(
        number="3c",
        provider="openai",
        question="Stateless edit: the image call left out, the image sent as an input_image",
        worst_usd=0.1,
        run=openai_stateless_edit_as_input,
    ),
    Check(
        number="2d",
        provider="openai",
        question="The Images API streamed, partial_images=2",
        worst_usd=0.04,
        run=openai_images_stream,
    ),
    Check(
        number="4a",
        provider="gemini",
        question=f"{GEMINI_LITE}, IMAGE only, 1:1",
        worst_usd=0.05,
        run=gemini_generate(
            GEMINI_LITE, "gemini-lite", response_modalities=["IMAGE"], image_aspect_ratio="1:1"
        ),
    ),
    Check(
        number="4b",
        provider="gemini",
        question=f"{GEMINI_FLASH}, TEXT+IMAGE, 512: thought images, signatures, usage",
        worst_usd=0.08,
        run=gemini_generate(
            GEMINI_FLASH,
            "gemini-flash",
            response_modalities=["TEXT", "IMAGE"],
            image_image_size="512",
        ),
    ),
    Check(
        number="4c",
        provider="gemini",
        question="The 4b turn replayed, then an edit",
        worst_usd=0.1,
        run=gemini_replay(strip_signatures=False),
    ),
    Check(
        number="4d",
        provider="gemini",
        question="The 4b turn replayed without the signatures on image parts",
        worst_usd=0.1,
        run=gemini_replay(strip_signatures=True),
    ),
    Check(
        number="4e",
        provider="gemini",
        question=f"{GEMINI_LITE} edit from an input image",
        worst_usd=0.05,
        run=gemini_edit_input,
    ),
    Check(
        number="4f",
        provider="gemini",
        question=f"{GEMINI_LITE} streamed: the image whole?",
        worst_usd=0.05,
        run=gemini_stream,
    ),
    Check(
        number="4g",
        provider="gemini",
        question=f"{GEMINI_LITE} with candidate_count=2",
        worst_usd=0.08,
        run=gemini_generate(
            GEMINI_LITE, "gemini-two", response_modalities=["IMAGE"], candidate_count=2
        ),
    ),
    Check(
        number="4h",
        provider="gemini",
        question=f"{GEMINI_LITE} with image_config output_mime_type=image/jpeg",
        worst_usd=0.05,
        run=gemini_generate(
            GEMINI_LITE,
            "gemini-jpeg",
            response_modalities=["IMAGE"],
            image_output_mime_type="image/jpeg",
        ),
    ),
    Check(
        number="4i",
        provider="gemini",
        question=f"{GEMINI_LITE} with image_size=512",
        worst_usd=0.05,
        run=gemini_generate(
            GEMINI_LITE, "gemini-512", response_modalities=["IMAGE"], image_image_size="512"
        ),
    ),
    Check(
        number="6a",
        provider="xai",
        question=f"{XAI_V1}: base64, cost_usd",
        worst_usd=0.03,
        run=xai_sample(XAI_V1),
    ),
    Check(
        number="6b",
        provider="xai",
        question=f"{XAI_V2}: 16:9, 1k, low",
        worst_usd=0.06,
        run=xai_sample(XAI_V2, aspect_ratio="16:9", resolution="1k", quality="low"),
    ),
    Check(
        number="6c",
        provider="xai",
        question=f"{XAI_V1} edit from a data URL",
        worst_usd=0.03,
        run=xai_edit,
    ),
    Check(
        number="6d",
        provider="xai",
        question=f"{XAI_V1} sample_batch n=2",
        worst_usd=0.05,
        run=xai_batch,
    ),
    Check(
        number="7a",
        provider="meta",
        question=f"{META_IMAGE} on the Responses API",
        worst_usd=0.02,
        run=meta_turn("meta-turn"),
    ),
    Check(
        number="7b",
        provider="meta",
        question=f"{META_IMAGE} with the image_generation tool, size 1536x1024",
        worst_usd=0.02,
        run=meta_turn("meta-wide", tools=[{"type": "image_generation", "size": "1536x1024"}]),
    ),
    Check(
        number="7e",
        provider="meta",
        question=f"{META_IMAGE} with the image_generation tool, size 1024x1024 (square?)",
        worst_usd=0.02,
        run=meta_turn("meta-square", tools=[{"type": "image_generation", "size": "1024x1024"}]),
    ),
    Check(
        number="7f",
        provider="meta",
        question=f"{META_IMAGE} on /v1/images/generations with size 1024x1792 (portrait?)",
        worst_usd=0.02,
        run=meta_images_endpoint_sized("1024x1792"),
    ),
    Check(
        number="7c",
        provider="meta",
        question="Stateless second turn, result null",
        worst_usd=0.02,
        run=meta_stateless_edit,
    ),
    Check(
        number="7d",
        provider="meta",
        question=f"{META_IMAGE} on /v1/images/generations",
        worst_usd=0.02,
        run=meta_images_endpoint,
    ),
)


def run_checks(checks: Sequence[Check], probe: Probe, budget: Budget) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for check in checks:
        asked = {"number": check.number, "provider": check.provider, "question": check.question}
        if not budget.admits(check.worst_usd):
            outcomes.append(Outcome(**asked, outcome="skipped: over --max-cost", skipped=True))
            continue
        budget.charge(check.worst_usd)
        started = time.monotonic()
        try:
            outcome = check.run(probe)
        except KeyError as exc:
            outcome = f"not run: needs {exc} from an earlier check"
        except Exception as exc:  # the probe records every failure as an answer
            outcome = f"error {type(exc).__name__}: {redact(str(exc))}"
        seconds = time.monotonic() - started
        outcomes.append(Outcome(**asked, outcome=outcome, seconds=seconds))
        print(f"[{check.number}] {check.question} ({seconds:.1f}s)\n    {outcome}\n", flush=True)
    return outcomes


def render(outcomes: Sequence[Outcome], budget: Budget, started: str) -> str:
    lines = [
        f"# Image probe · {started}",
        "",
        f"Worst cases charged: ${budget.spent:.2f} of ${budget.cap:.2f}.",
        "",
    ]
    for outcome in outcomes:
        lines += [
            f"## {outcome.number} · {outcome.provider} · {outcome.question}",
            "",
            f"{outcome.outcome}" + (f" ({outcome.seconds:.1f}s)" if outcome.seconds else ""),
            "",
        ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-cost", type=float, default=2.0, help="USD cap for the whole run")
    parser.add_argument(
        "--provider",
        action="append",
        choices=sorted({check.provider for check in CHECKS}),
        help="run only this provider's checks (repeatable)",
    )
    parser.add_argument("--only", action="append", help="run only this check number (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_cost <= 0:
        raise SystemExit("--max-cost must be > 0")
    checks = [
        check
        for check in CHECKS
        if (not args.provider or check.provider in args.provider)
        and (not args.only or check.number in args.only)
    ]
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    probe = Probe(out_dir=args.output_dir / f"images-{started}")
    budget = Budget(cap=args.max_cost)
    outcomes = run_checks(checks, probe, budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / f"images-{started}.md"
    report.write_text(render(outcomes, budget, started))
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
