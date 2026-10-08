"""nanope's own update, pending after the tools contract (T05 to T09).

nanope imports toolkit tools by name, and eight of them are gone: ``define_word``,
``wikipedia_search``, ``wikipedia_article`` and ``wikipedia_related`` (the wiki family replaced
them, T05), ``reverse_geocode``, ``get_weather_by_coords``, ``get_forecast_by_coords`` and
``weather_units`` (merged into the others of their source, T08; the CHANGELOG's migration table).
nanope belongs to its owner, who updates it (R00, T00). Until then the tests that build its
tools skip; they run again on their own once nanope builds them.
"""

from __future__ import annotations

from types import ModuleType

import pytest

_REASON = (
    "nanope imports tools the tools contract replaced (define_word, wikipedia_*, "
    "reverse_geocode, get_weather_by_coords, get_forecast_by_coords, weather_units); "
    "its owner updates it (T05, T08)"
)


def _nanope_builds_its_tools() -> bool:
    try:
        from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._tools import (
            built_in_tool_registry,
        )

        built_in_tool_registry()
    except ImportError:
        return False
    return True


UNTIL_NANOPE_BUILDS_ITS_TOOLS = pytest.mark.skipif(not _nanope_builds_its_tools(), reason=_REASON)


def research_center_agents() -> ModuleType:
    """``nanope.research_center._agents``, or a skip of the module that asks for it."""
    return pytest.importorskip(
        "ai_arch_toolkit.nanope.research_center._agents", reason=_REASON, exc_type=ImportError
    )
