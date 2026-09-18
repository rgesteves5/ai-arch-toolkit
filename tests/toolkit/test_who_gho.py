"""Tests for toolkit/tools/_who_gho.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._who_gho import who_indicator, who_indicators, who_series
from tests.toolkit.http_fakes import HTTP_OPEN, respond


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestWhoGho:
    @patch(HTTP_OPEN)
    def test_indicators_and_indicator(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"value": [{"IndicatorCode": "WHOSIS_000001", "IndicatorName": "Life expectancy"}]}
        )

        result = who_indicators("life")

        assert "WHOSIS_000001 — Life expectancy" in result
        assert "$filter" in _params(mock_urlopen)

        mock_urlopen.return_value = respond(
            {
                "value": [
                    {
                        "IndicatorCode": "WHOSIS_000001",
                        "IndicatorName": "Life expectancy",
                        "Language": "EN",
                    }
                ]
            }
        )
        assert "WHO GHO indicator WHOSIS_000001:" in who_indicator("WHOSIS_000001")

    @patch(HTTP_OPEN)
    def test_series_and_validation(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "value": [
                    {
                        "SpatialDim": "PRT",
                        "TimeDim": 2020,
                        "Value": "80.0",
                        "ParentLocation": "Europe",
                        "Dim1": "SEX_BTSX",
                    }
                ]
            }
        )

        result = who_series("WHOSIS_000001", country="PRT", from_year="2020")

        assert "PRT 2020: 80.0" in result
        assert "SpatialDim eq 'PRT'" in _params(mock_urlopen)["$filter"][0]
        assert "invalid country" in who_series("WHOSIS_000001", country="PT")
