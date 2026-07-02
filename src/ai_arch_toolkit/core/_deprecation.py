"""Thin wrapper around ``warnings.deprecated`` (PEP 702) for the project's
deprecation policy.

Usage::

    from ai_arch_toolkit.core import deprecated


    @deprecated("Use new_name() instead.", removed_in="0.3")
    def old_name() -> int:
        return new_name()


    @deprecated("OldClass has been folded into NewClass.", removed_in="0.4")
    class OldClass:
        ...

The decorator emits a ``DeprecationWarning`` on use and is also visible to
type-checkers (pyright/mypy mark calls with a strikethrough). When the
``removed_in`` version arrives, delete the symbol and add a ``Removed`` entry
to ``CHANGELOG.md``.

See the *Breaking changes and deprecation* section of ``CONTRIBUTING.md`` for
the full policy.
"""

from __future__ import annotations

from typing import Any
from warnings import deprecated as _warnings_deprecated


def deprecated(message: str, *, removed_in: str | None = None) -> Any:
    """Decorator that marks a callable or class as deprecated.

    Args:
        message: Short reason and migration hint, e.g. ``"Use new_name() instead."``.
        removed_in: Optional version when the symbol will be removed
            (e.g. ``"0.3"``). Surfaced in the warning text so users have
            time to migrate.

    Returns:
        A decorator. Wrapping the target keeps it callable and preserves
        metadata; type-checkers see it as deprecated.
    """
    full = f"{message} Will be removed in v{removed_in.lstrip('v')}." if removed_in else message
    # ``warnings.deprecated`` requires a LiteralString to keep the warning
    # text auditable at type-check time. Our wrapper builds the final string
    # at runtime from a caller-supplied literal plus an optional version, so
    # the suppression is intentional — both inputs are project-controlled.
    return _warnings_deprecated(full)  # type: ignore[arg-type]
