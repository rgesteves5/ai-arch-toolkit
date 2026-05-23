"""Tests for the @deprecated decorator helper."""

from __future__ import annotations

import warnings

import pytest

from ai_arch_toolkit import deprecated


class TestDeprecatedFunction:
    def test_warns_when_called(self) -> None:
        @deprecated("Use new_thing() instead.")
        def old_thing() -> int:
            return 42

        with pytest.warns(DeprecationWarning, match="Use new_thing"):
            assert old_thing() == 42

    def test_removed_in_added_to_message(self) -> None:
        @deprecated("Switch to new_api.", removed_in="0.3")
        def stale() -> str:
            return "value"

        with pytest.warns(DeprecationWarning, match=r"removed in v0\.3") as record:
            stale()
        assert "Switch to new_api." in str(record[0].message)

    def test_removed_in_strips_leading_v(self) -> None:
        @deprecated("Replaced.", removed_in="v1.0")
        def x() -> None: ...

        with pytest.warns(DeprecationWarning, match=r"removed in v1\.0"):
            x()

    def test_no_removed_in_message_unchanged(self) -> None:
        @deprecated("Just a heads-up.")
        def y() -> None: ...

        with pytest.warns(DeprecationWarning) as record:
            y()
        msg = str(record[0].message)
        assert "Just a heads-up." in msg
        assert "Will be removed" not in msg

    def test_function_still_callable(self) -> None:
        @deprecated("noop")
        def echo(x: int) -> int:
            return x * 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert echo(3) == 6


class TestDeprecatedClass:
    def test_warns_on_instantiation(self) -> None:
        @deprecated("OldThing has been folded into NewThing.", removed_in="0.4")
        class OldThing:
            def __init__(self, value: int) -> None:
                self.value = value

        with pytest.warns(DeprecationWarning, match="folded into NewThing"):
            instance = OldThing(7)
        assert instance.value == 7

    def test_subclassing_still_works(self) -> None:
        @deprecated("legacy")
        class Base:
            def hello(self) -> str:
                return "hi"

        # Subclassing emits one warning, then the new class works as usual.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            class Child(Base):
                pass

            assert Child().hello() == "hi"
