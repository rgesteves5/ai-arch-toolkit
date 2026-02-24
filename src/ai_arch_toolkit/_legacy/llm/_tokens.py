"""Token counting utilities with model-aware correction factors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ai_arch_toolkit._legacy.llm._types import (
    Content,
    ConversationItem,
    DocumentPart,
    ImagePart,
    Message,
    ToolResult,
)

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


CLAUDE_3_CORRECTION_FACTOR = 1.12
CLAUDE_4_CORRECTION_FACTOR = 1.15
GEMINI_CORRECTION_FACTOR = 1.05
GROK_CORRECTION_FACTOR = 1.05
META_CORRECTION_FACTOR = 1.02
DEFAULT_CORRECTION_FACTOR = 1.0

RawTokenCounter = Callable[[str, str], int]


@dataclass(frozen=True, slots=True)
class TokenCorrectionConfig:
    """Per-family token correction factors plus exact model overrides."""

    claude_3_factor: float = CLAUDE_3_CORRECTION_FACTOR
    claude_4_factor: float = CLAUDE_4_CORRECTION_FACTOR
    gemini_factor: float = GEMINI_CORRECTION_FACTOR
    grok_factor: float = GROK_CORRECTION_FACTOR
    meta_factor: float = META_CORRECTION_FACTOR
    default_factor: float = DEFAULT_CORRECTION_FACTOR
    model_overrides: Mapping[str, float] = field(default_factory=dict)


DEFAULT_TOKEN_CORRECTION_CONFIG = TokenCorrectionConfig()


def _lookup_model_override(model_name: str, overrides: Mapping[str, float]) -> float | None:
    """Return a case-insensitive exact model override when configured."""
    if model_name in overrides:
        return overrides[model_name]
    for key, factor in overrides.items():
        if key.strip().lower() == model_name:
            return factor
    return None


def get_correction_factor(
    model_name: str,
    *,
    config: TokenCorrectionConfig | None = None,
) -> float:
    """Get token correction factor for a model name.

    Matching order:
    1. Exact model override from ``config.model_overrides``
    2. Family rules (Claude 4, Claude 3, Gemini, Grok/xAI, Meta/Llama 4)
    3. Default factor
    """
    cfg = config or DEFAULT_TOKEN_CORRECTION_CONFIG
    normalized = model_name.strip().lower()
    if not normalized:
        return cfg.default_factor

    override = _lookup_model_override(normalized, cfg.model_overrides)
    if override is not None:
        return override

    if "claude" in normalized:
        if any(
            marker in normalized for marker in ("claude-4", "sonnet-4", "opus-4", "4.5", "4-5")
        ):
            return cfg.claude_4_factor
        return cfg.claude_3_factor

    if "gemini" in normalized:
        return cfg.gemini_factor

    if "grok" in normalized or "xai" in normalized:
        return cfg.grok_factor

    if (
        "llama-4" in normalized
        or "llama 4" in normalized
        or ("meta" in normalized and "llama" in normalized)
    ):
        return cfg.meta_factor

    return cfg.default_factor


def raw_tiktoken_count(text: str, model_name: str) -> int:
    """Return raw token count using ``tiktoken`` when available.

    Falls back to the heuristic ``len(text) // 4`` when ``tiktoken`` is not
    installed or the model encoding cannot be resolved.
    """
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoder = tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return max(1, len(text) // 4)
    try:
        return len(encoder.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_text_tokens_for_model(
    text: str,
    model_name: str,
    *,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> int:
    """Estimate text tokens for a specific model with correction factors.

    Formula:
        ``int(raw_token_count * correction_factor)``
    """
    if not text:
        return 0
    raw_count = (raw_token_counter or raw_tiktoken_count)(text, model_name)
    if raw_count <= 0:
        return 0
    factor = get_correction_factor(model_name, config=correction_config)
    return int(raw_count * factor)


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for plain text using ``len(text) // 4``."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_content_tokens(content: Content) -> int:
    """Estimate token count for multimodal content."""
    if isinstance(content, str):
        return estimate_text_tokens(content)

    total = 0
    for part in content:
        if hasattr(part, "text"):
            total += estimate_text_tokens(part.text)
        elif isinstance(part, ImagePart):
            total += 85
        elif isinstance(part, DocumentPart):
            total += max(50, estimate_text_tokens(part.data))
        else:
            total += 20
    return total


def estimate_content_tokens_for_model(
    content: Content,
    model_name: str,
    *,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> int:
    """Estimate multimodal content tokens for a specific model."""
    if isinstance(content, str):
        return estimate_text_tokens_for_model(
            content,
            model_name,
            correction_config=correction_config,
            raw_token_counter=raw_token_counter,
        )

    total = 0
    for part in content:
        if hasattr(part, "text"):
            total += estimate_text_tokens_for_model(
                part.text,
                model_name,
                correction_config=correction_config,
                raw_token_counter=raw_token_counter,
            )
        elif isinstance(part, ImagePart):
            total += 85
        elif isinstance(part, DocumentPart):
            total += max(
                50,
                estimate_text_tokens_for_model(
                    part.data,
                    model_name,
                    correction_config=correction_config,
                    raw_token_counter=raw_token_counter,
                ),
            )
        else:
            total += 20
    return total


def estimate_message_tokens(message: Message) -> int:
    """Estimate tokens for a message including small structural overhead."""
    total = 4 + estimate_content_tokens(message.content)
    for tool_call in message.tool_calls:
        total += estimate_text_tokens(tool_call.name)
        total += estimate_text_tokens(str(tool_call.arguments))
    return total


def estimate_message_tokens_for_model(
    message: Message,
    model_name: str,
    *,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> int:
    """Estimate message tokens for a specific model."""
    total = 4 + estimate_content_tokens_for_model(
        message.content,
        model_name,
        correction_config=correction_config,
        raw_token_counter=raw_token_counter,
    )
    for tool_call in message.tool_calls:
        total += estimate_text_tokens_for_model(
            tool_call.name,
            model_name,
            correction_config=correction_config,
            raw_token_counter=raw_token_counter,
        )
        total += estimate_text_tokens_for_model(
            str(tool_call.arguments),
            model_name,
            correction_config=correction_config,
            raw_token_counter=raw_token_counter,
        )
    return total


def estimate_item_tokens(item: ConversationItem) -> int:
    """Estimate tokens for a conversation item (message or tool result)."""
    if isinstance(item, ToolResult):
        return 4 + estimate_text_tokens(item.name) + estimate_text_tokens(item.content)
    return estimate_message_tokens(item)


def estimate_item_tokens_for_model(
    item: ConversationItem,
    model_name: str,
    *,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> int:
    """Estimate conversation item tokens for a specific model."""
    if isinstance(item, ToolResult):
        return (
            4
            + estimate_text_tokens_for_model(
                item.name,
                model_name,
                correction_config=correction_config,
                raw_token_counter=raw_token_counter,
            )
            + estimate_text_tokens_for_model(
                item.content,
                model_name,
                correction_config=correction_config,
                raw_token_counter=raw_token_counter,
            )
        )
    return estimate_message_tokens_for_model(
        item,
        model_name,
        correction_config=correction_config,
        raw_token_counter=raw_token_counter,
    )


def estimate_conversation_tokens(items: list[ConversationItem]) -> int:
    """Estimate total tokens for a conversation history."""
    return sum(estimate_item_tokens(item) for item in items)


def estimate_conversation_tokens_for_model(
    items: list[ConversationItem],
    model_name: str,
    *,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> int:
    """Estimate conversation tokens for a specific model."""
    return sum(
        estimate_item_tokens_for_model(
            item,
            model_name,
            correction_config=correction_config,
            raw_token_counter=raw_token_counter,
        )
        for item in items
    )
