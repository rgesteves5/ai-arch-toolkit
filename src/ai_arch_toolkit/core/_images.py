"""What an image generation asks for (D47), in the toolkit's portable vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, get_args

from ai_arch_toolkit.core._exceptions import RequestError

__all__ = ["ImageFormat", "ImageRequest", "ImageResolution"]

type ImageResolution = Literal["512", "1K", "2K", "4K"]
type ImageFormat = Literal["png", "jpeg", "webp"]

_RESOLUTIONS = frozenset(get_args(ImageResolution.__value__))
_FORMATS = frozenset(get_args(ImageFormat.__value__))
_ASPECT_RATIO = re.compile(r"^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$")


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageRequest:
    """The options of an image generation; ``None`` leaves the choice to the model.

    Each adapter maps them to its provider and raises ``RequestError`` for what the model does
    not take (a ratio it lacks, ``quality`` on a model without one, ``n > 1`` where only one image
    comes back).

    Attributes:
        n: How many images to generate.
        aspect_ratio: Width to height, as ``"16:9"``.
        resolution: The image's size class: ``"512"``, ``"1K"``, ``"2K"`` or ``"4K"`` pixels on
            its long edge, roughly.
        quality: The provider's quality level (``"low"``, ``"medium"``, ``"high"``…).
        output_format: ``"png"``, ``"jpeg"`` or ``"webp"``.
    """

    n: int = 1
    aspect_ratio: str | None = None
    resolution: ImageResolution | None = None
    quality: str | None = None
    output_format: ImageFormat | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise RequestError(f"n must be a positive integer, got {self.n!r}")
        if self.aspect_ratio is not None and self.ratio is None:
            raise RequestError(
                f"aspect_ratio must be 'W:H' with positive numbers, got {self.aspect_ratio!r}"
            )
        if self.resolution is not None and self.resolution not in _RESOLUTIONS:
            raise RequestError(
                f"resolution must be one of {sorted(_RESOLUTIONS)}, got {self.resolution!r}"
            )
        if self.quality is not None and (not isinstance(self.quality, str) or not self.quality):
            raise RequestError(f"quality must be a non-empty string, got {self.quality!r}")
        if self.output_format is not None and self.output_format not in _FORMATS:
            raise RequestError(
                f"output_format must be one of {sorted(_FORMATS)}, got {self.output_format!r}"
            )

    @property
    def ratio(self) -> float | None:
        """The aspect ratio as width over height, or ``None`` when none was asked (or invalid)."""
        if not isinstance(self.aspect_ratio, str):
            return None
        found = _ASPECT_RATIO.match(self.aspect_ratio)
        if found is None:
            return None
        width, height = float(found.group(1)), float(found.group(2))
        return width / height if width > 0 and height > 0 else None
