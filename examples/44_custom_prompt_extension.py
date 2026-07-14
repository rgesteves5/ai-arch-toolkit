"""44 — Register a custom resource codec and use it in a prompt. No API key required."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.toolkit.prompts import Prompt
from ai_arch_toolkit.toolkit.resources import DecodedResource, ResourceResolver


class UpperCodec:
    """Example codec for `.upper` files."""

    name = "upper"

    def decode(self, raw: bytes, ref) -> DecodedResource:
        text = raw.decode(ref.encoding).upper()
        return DecodedResource(data=text, text=text)


ASSET = Path(__file__).parent / "assets/prompts/story_writer/emphasis.upper"
resolver = ResourceResolver()
resolver.register_codec("text/x-upper", UpperCodec(), extensions=("upper",))

prompt = Prompt.from_file(ASSET, resolver=resolver)
print(prompt.render().text)
