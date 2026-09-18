"""Local token counting using tiktoken with provider-specific correction factors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ai_arch_toolkit.core._model_id import family

# Correction factors by model family: tiktoken underestimates non-OpenAI models.
_CORRECTIONS: dict[str, float] = {
    "gpt-": 1.0,  # tiktoken is exact for OpenAI
    "o1": 1.0,
    "o3": 1.0,
    "o4": 1.0,
    "claude-3-": 1.12,  # ~12% underestimate
    "claude-": 1.15,  # Claude 4.x: ~15% underestimate (different tokenizer)
    "gemini-": 1.05,  # ~5% variance
    "grok-": 1.05,  # ~5% variance
    "llama": 1.02,  # Meta Llama (tiktoken-based, ~2%)
    "muse-spark-": 1.0,  # Meta input_tokens count was 0.98x o200k_base on English prose
}

# Families that use the o200k_base encoding (GPT-4o+, o-series, Muse Spark).
_ENCODINGS: dict[str, str] = dict.fromkeys(
    ("gpt-4o", "gpt-5", "o1", "o3", "o4", "muse-spark-"), "o200k_base"
)

# Average chars per token (rough cross-model approximation).
_CHARS_PER_TOKEN = 4


def _get_correction(model: str) -> float:
    """Return the correction factor of *model*'s family (1.0 when unknown)."""
    factor = family(model, _CORRECTIONS)
    return 1.0 if factor is None else factor


def _get_encoding(model: str) -> Any:
    """Select the appropriate tiktoken encoding for *model*."""
    try:
        import tiktoken
    except ImportError as e:
        msg = (
            "Local token counting requires tiktoken. "
            "Install with: pip install ai-arch-toolkit[tokens]"
        )
        raise ImportError(msg) from e

    return tiktoken.get_encoding(family(model, _ENCODINGS) or "cl100k_base")


def count_tokens_local(
    text: str,
    model: str = "gpt-4o",
    *,
    correction: float | None = None,
) -> int:
    """Count tokens locally using tiktoken with provider-specific correction.

    Args:
        text: The text to tokenize.
        model: Model name (used to select encoding + correction factor).
        correction: Override the built-in correction factor. Pass 1.0 for raw tiktoken count.

    Returns:
        Estimated token count.

    Raises:
        ImportError: If tiktoken is not installed.
    """
    enc = _get_encoding(model)
    raw = len(enc.encode(text))
    factor = correction if correction is not None else _get_correction(model)
    return math.ceil(raw * factor)


def count_tokens_local_batch(
    texts: Sequence[str],
    model: str = "gpt-4o",
    *,
    correction: float | None = None,
) -> int:
    """Count tokens across multiple texts. Returns total."""
    enc = _get_encoding(model)
    factor = correction if correction is not None else _get_correction(model)
    total = sum(len(enc.encode(t)) for t in texts)
    return math.ceil(total * factor)


def chars_to_tokens(chars: int, model: str = "gpt-4o") -> int:
    """Fast approximation: ~4 chars per token, with correction factor."""
    raw = chars / _CHARS_PER_TOKEN
    return math.ceil(raw * _get_correction(model))


def tokens_to_chars(tokens: int) -> int:
    """Reverse estimation: ~4 chars per token."""
    return tokens * _CHARS_PER_TOKEN
