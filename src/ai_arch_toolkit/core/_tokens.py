"""Local token counting using tiktoken with provider-specific correction factors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

# Correction factors: tiktoken underestimates non-OpenAI models.
# Keyed by model prefix → correction factor (longest-prefix match).
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
}

# Prefixes that use the o200k_base encoding (GPT-4o+, o-series).
_O200K_PREFIXES = ("gpt-4o", "gpt-5", "o1", "o3", "o4")

# Average chars per token (rough cross-model approximation).
_CHARS_PER_TOKEN = 4


def _get_correction(model: str) -> float:
    """Return the correction factor for *model* via longest-prefix match."""
    best = 1.0
    best_len = 0
    for prefix, factor in _CORRECTIONS.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best = factor
            best_len = len(prefix)
    return best


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

    if any(model.startswith(p) for p in _O200K_PREFIXES):
        return tiktoken.get_encoding("o200k_base")
    return tiktoken.get_encoding("cl100k_base")


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
