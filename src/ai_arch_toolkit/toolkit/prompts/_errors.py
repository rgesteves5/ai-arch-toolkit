"""Prompt loading, validation, templating, and rendering exceptions."""

from __future__ import annotations


class PromptError(Exception):
    """Base class for prompt subsystem failures."""


class PromptLoadError(PromptError, OSError):
    """A prompt definition could not be loaded."""


class PromptValidationError(PromptError, ValueError):
    """A prompt definition or manifest is invalid."""


class PromptRenderError(PromptError, ValueError):
    """A resolved prompt could not be rendered."""


class PromptVariableError(PromptValidationError):
    """Prompt variables are missing, unknown, or invalid."""


class MissingPromptVariableError(PromptVariableError):
    """A required prompt variable was not supplied."""


class PromptTemplateError(PromptRenderError):
    """A template engine could not render a section."""


class PromptIncludeCycleError(PromptValidationError):
    """Prompt manifests contain an include or inheritance cycle."""


class PromptSecurityError(PromptError, PermissionError):
    """A prompt operation was rejected by a security policy."""


__all__ = [
    "MissingPromptVariableError",
    "PromptError",
    "PromptIncludeCycleError",
    "PromptLoadError",
    "PromptRenderError",
    "PromptSecurityError",
    "PromptTemplateError",
    "PromptValidationError",
    "PromptVariableError",
]
