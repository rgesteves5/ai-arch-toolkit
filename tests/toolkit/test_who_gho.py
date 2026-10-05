"""Tests for toolkit/tools/_who_gho.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._who_gho import who_indicator, who_indicators, who_series
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


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

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: who_series("WHOSIS_000001", country="PT"), "invalid country 'PT'"),
            (lambda: who_series("WHOSIS_000001", from_year="20"), "invalid from_year '20'"),
            (
                lambda: who_series("WHOSIS_000001", from_year="2020", to_year="2010"),
                "from_year 2020 is after to_year 2010",
            ),
            (lambda: who_series("WHOSIS_000001", skip=-1), "invalid skip -1"),
            (lambda: who_series("bad code!"), "invalid indicator_code 'bad code!'"),
            (lambda: who_indicator("bad code!"), "invalid indicator_code"),
            (lambda: who_indicators("a;b"), "invalid query"),
            (lambda: who_indicators(skip=-2), "invalid skip -2"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_arguments_do_not_call_api(self, mock_urlopen, call, words):
        with pytest.raises(ToolFailure) as caught:
            call()

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_unknown_indicator_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({"value": []})

        with pytest.raises(ToolFailure) as caught:
            who_indicator("NOPE_1")

        assert caught.value.error.type == "not_found"
        assert "who_indicators" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_no_results_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"value": []})
        assert who_indicators("zzzz") == "No WHO GHO indicators found."

        mock_urlopen.return_value = respond({"value": []})
        assert who_series("WHOSIS_000001") == "No WHO GHO observations found for WHOSIS_000001."

    @patch(HTTP_OPEN)
    def test_a_server_error_is_retryable_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        with pytest.raises(ToolFailure) as caught:
            who_series("WHOSIS_000001")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
