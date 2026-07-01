"""The toolkit's budget exception — an opinionated subclass of the neutral core denial."""

from __future__ import annotations

from ai_arch_toolkit.core._metering._admission import AdmissionDenied

__all__ = ["BudgetExceeded"]


class BudgetExceeded(AdmissionDenied):
    """A run hit a configured budget cap.

    Subclasses the neutral :class:`~ai_arch_toolkit.core.AdmissionDenied`, so a caller that
    catches the core type also catches this, and the store's hard re-validation (which raises the
    neutral base under a race) is handled by the same ``except``. Carries the structured
    ``dimension``/``limit``/``current``/``attempted`` fields inherited from the base.
    """
