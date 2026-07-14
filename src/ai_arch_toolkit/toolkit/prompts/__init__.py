"""Structured, provider-agnostic prompt composition."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.prompts._errors import (
    MissingPromptVariableError,
    PromptError,
    PromptIncludeCycleError,
    PromptLoadError,
    PromptRenderError,
    PromptSecurityError,
    PromptTemplateError,
    PromptValidationError,
    PromptVariableError,
)
from ai_arch_toolkit.toolkit.prompts._layouts import (
    JsonLayout,
    LayoutResult,
    MarkdownLayout,
    PromptLayout,
    SectionSpan,
    SeparatorPolicy,
    TextLayout,
    XmlLayout,
)
from ai_arch_toolkit.toolkit.prompts._manifest import load_prompt
from ai_arch_toolkit.toolkit.prompts._messages import (
    PromptConversation,
    PromptMessage,
    PromptMessageContent,
    PromptMessageRole,
    RenderedPromptConversation,
    RenderedPromptMessage,
)
from ai_arch_toolkit.toolkit.prompts._render import render_prompt, validate_cache_layout
from ai_arch_toolkit.toolkit.prompts._sources import (
    CallableSource,
    KnowledgeSource,
    LiteralSource,
    PromptSource,
    ResourceSource,
    SourceResolution,
    knowledge_source,
)
from ai_arch_toolkit.toolkit.prompts._template_engines import (
    JinjaTemplateEngine,
    StringTemplateEngine,
    TemplateEngine,
)
from ai_arch_toolkit.toolkit.prompts._templates import PromptTemplate, PromptTemplateSection
from ai_arch_toolkit.toolkit.prompts._types import (
    Prompt,
    PromptSection,
    PromptStability,
    RenderedPrompt,
    prompt_from_sections,
)
from ai_arch_toolkit.toolkit.prompts._variables import (
    MISSING,
    PromptVariable,
    PromptVariableType,
)

__all__ = [
    "MISSING",
    "CallableSource",
    "JinjaTemplateEngine",
    "JsonLayout",
    "KnowledgeSource",
    "LayoutResult",
    "LiteralSource",
    "MarkdownLayout",
    "MissingPromptVariableError",
    "Prompt",
    "PromptConversation",
    "PromptError",
    "PromptIncludeCycleError",
    "PromptLayout",
    "PromptLoadError",
    "PromptMessage",
    "PromptMessageContent",
    "PromptMessageRole",
    "PromptRenderError",
    "PromptSection",
    "PromptSecurityError",
    "PromptSource",
    "PromptStability",
    "PromptTemplate",
    "PromptTemplateError",
    "PromptTemplateSection",
    "PromptValidationError",
    "PromptVariable",
    "PromptVariableError",
    "PromptVariableType",
    "RenderedPrompt",
    "RenderedPromptConversation",
    "RenderedPromptMessage",
    "ResourceSource",
    "SectionSpan",
    "SeparatorPolicy",
    "SourceResolution",
    "StringTemplateEngine",
    "TemplateEngine",
    "TextLayout",
    "XmlLayout",
    "knowledge_source",
    "load_prompt",
    "prompt_from_sections",
    "render_prompt",
    "validate_cache_layout",
]
