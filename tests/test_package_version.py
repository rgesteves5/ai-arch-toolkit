"""Tests for package version export."""

from __future__ import annotations

from importlib import metadata

import ai_arch_toolkit


def test_version_matches_distribution_metadata() -> None:
    try:
        expected = metadata.version("ai-arch-toolkit")
    except metadata.PackageNotFoundError:
        assert ai_arch_toolkit.__version__ == "0.0.0"
    else:
        assert ai_arch_toolkit.__version__ == expected
