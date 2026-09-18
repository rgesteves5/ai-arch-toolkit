"""The import guard of the optional provider SDKs: a missing one names the extra to install."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def require_sdk(extra: str) -> Iterator[None]:
    """Guard the SDK imports of one extra: ``with require_sdk("openai"): import openai``.

    Raises:
        ImportError: An import in the block failed; the message says how to install ``extra``.
    """
    try:
        yield
    except ImportError as missing:
        msg = f"Install the {extra} extra: pip install ai-arch-toolkit[{extra}]"
        raise ImportError(msg) from missing
