"""Tests for toolkit/tools/_who_gho.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolFailure, ToolGroup, ToolResult
from ai_arch_toolkit.toolkit.tools._who_gho import who_indicator, who_indicators, who_series
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok
    assert isinstance(result.value, str)
    return result.value


def _indicators(count: int, start: int = 1) -> dict[str, object]:
    return {
        "value": [
            {"IndicatorCode": f"CODE_{number}", "IndicatorName": f"Indicator {number}"}
            for number in range(start, start + count)
        ]
    }


# A row of an indicator's entity set, with the fields the API sends.
_ROW = {
    "Id": 1,
    "IndicatorCode": "WHOSIS_000001",
    "SpatialDimType": "COUNTRY",
    "SpatialDim": "PRT",
    "ParentLocationCode": "EUR",
    "ParentLocation": "Europe",
    "TimeDimType": "YEAR",
    "TimeDim": 2019,
    "Dim1Type": "SEX",
    "Dim1": "SEX_BTSX",
    "Value": "81.3 [80.9-81.7]",
    "NumericValue": 81.31234,
    "Low": 80.9,
    "High": 81.7,
    "Date": "2020-12-04T16:59:42.523+01:00",
}


class TestWhoIndicators:
    @patch(HTTP_OPEN)
    def test_lists_indicators_with_their_codes(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"value": [{"IndicatorCode": "WHOSIS_000001", "IndicatorName": "Life expectancy"}]}
        )

        text = _text(who_indicators("life"))

        assert text == (
            "WHO GHO indicators that match 'life' (who_series reads one):\n"
            "1. WHOSIS_000001: Life expectancy"
        )
        assert "contains(tolower(IndicatorName),'life')" in _params(mock_urlopen)["$filter"][0]

    @patch(HTTP_OPEN)
    def test_one_more_than_shown_says_there_is_more(self, mock_urlopen):
        # They paged with no total and no sign of more; the API gives no count, so the tool asks
        # for one more than it shows.
        mock_urlopen.return_value = respond(_indicators(4, start=3))

        result = who_indicators(max_results=3, skip=2)
        text = _text(result)

        assert _params(mock_urlopen)["$top"] == ["4"]
        assert _params(mock_urlopen)["$skip"] == ["2"]
        assert text.splitlines()[1] == "3. CODE_3: Indicator 3"
        assert "CODE_6" not in text
        assert text.endswith("[results 3-5 | next: skip=5]")
        assert result.metadata["window"]["next_call"] == {"skip": 5}

    @patch(HTTP_OPEN)
    def test_the_last_page_has_no_next_call(self, mock_urlopen):
        mock_urlopen.return_value = respond(_indicators(2, start=6))

        text = _text(who_indicators(max_results=3, skip=5))

        assert text.endswith("[results 6-7 | end]")

    @patch(HTTP_OPEN)
    def test_an_odata_next_link_also_says_there_is_more(self, mock_urlopen):
        # OData's server paging (https://docs.oasis-open.org/odata/odata-json-format/v4.01/
        # odata-json-format-v4.01.html#sec_ControlInformationnextLinkodatanextLink).
        mock_urlopen.return_value = respond(
            {**_indicators(2), "@odata.nextLink": "https://ghoapi.azureedge.net/api/Indicator?x"}
        )

        assert _text(who_indicators(max_results=3)).endswith("[results 1-2 | next: skip=2]")

    @patch(HTTP_OPEN)
    def test_no_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond({"value": []})

        assert _text(who_indicators("zzzz")) == "No WHO GHO indicators match 'zzzz'."

    def test_the_limits_are_in_the_schema(self):
        group = ToolGroup(who_indicators, who_series)
        for name, args in (
            ("who_indicators", {"max_results": 101}),
            ("who_indicators", {"skip": -1}),
            ("who_series", {"indicator_code": "X", "max_results": 0}),
        ):
            result = group.execute(ToolCall(id="c", name=name, input=args))
            assert not result.ok and result.error is not None
            assert result.error.type == "validation_error", args


class TestWhoIndicator:
    @patch(HTTP_OPEN)
    def test_reads_an_indicator(self, mock_urlopen):
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

        text = _text(who_indicator("WHOSIS_000001"))

        assert text.startswith("WHO GHO indicator WHOSIS_000001:\nLife expectancy")

    @patch(HTTP_OPEN)
    def test_an_unknown_indicator_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({"value": []})

        with pytest.raises(ToolFailure) as caught:
            who_indicator("NOPE_1")

        assert caught.value.error.type == "not_found"
        assert "who_indicators" in caught.value.error.message


class TestWhoSeries:
    @patch(HTTP_OPEN)
    def test_rows_name_each_codes_dimension_and_the_labels_the_answer_brings(self, mock_urlopen):
        mock_urlopen.return_value = respond({"value": [_ROW]})

        text = _text(who_series("WHOSIS_000001", country="PRT", from_year="2019"))

        assert text == (
            "WHO GHO observations of WHOSIS_000001 (country=PRT, from 2019):\n"
            "1. PRT (country, region Europe EUR), 2019: 81.31234 (80.9 to 81.7) | sex: SEX_BTSX"
        )
        assert "SpatialDim eq 'PRT'" in _params(mock_urlopen)["$filter"][0]

    @pytest.mark.parametrize("place", ["SEAR", "GLOBAL", "wb_lmi", "EUR"])
    @patch(HTTP_OPEN)
    def test_every_place_code_a_row_shows_is_a_filter(self, mock_urlopen, place):
        # Rows show regions, the globe and income groups (SpatialDim); only ISO3 was taken.
        mock_urlopen.return_value = respond({"value": [_ROW]})

        who_series("WHOSIS_000001", country=place)

        assert _params(mock_urlopen)["$filter"] == [f"SpatialDim eq '{place.upper()}'"]

    @patch(HTTP_OPEN)
    def test_a_row_without_a_number_keeps_the_displayed_value(self, mock_urlopen):
        row = {"SpatialDimType": "REGION", "SpatialDim": "EUR", "TimeDim": 2019, "Value": "<0.1"}
        mock_urlopen.return_value = respond({"value": [row]})

        assert _text(who_series("X")).splitlines()[1] == "1. EUR (region), 2019: <0.1"

    @patch(HTTP_OPEN)
    def test_series_read_on_with_skip(self, mock_urlopen):
        mock_urlopen.return_value = respond({"value": [_ROW] * 3})

        text = _text(who_series("WHOSIS_000001", max_results=2, skip=4))

        assert _params(mock_urlopen)["$top"] == ["3"]
        assert text.splitlines()[1].startswith("5. PRT")
        assert text.endswith("[results 5-6 | next: skip=6]")

    @patch(HTTP_OPEN)
    def test_no_observations_say_so_with_the_filters(self, mock_urlopen):
        mock_urlopen.return_value = respond({"value": []})

        assert _text(who_series("WHOSIS_000001", country="PRT")) == (
            "No WHO GHO observations of WHOSIS_000001 match country=PRT."
        )

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: who_series("WHOSIS_000001", country="P'T"), 'invalid country "P\'T"'),
            (lambda: who_series("WHOSIS_000001", country="X"), "use a place code"),
            (lambda: who_series("WHOSIS_000001", from_year="20"), "invalid from_year '20'"),
            (
                lambda: who_series("WHOSIS_000001", from_year="2020", to_year="2010"),
                "from_year 2020 is after to_year 2010",
            ),
            (lambda: who_series("bad code!"), "invalid indicator_code 'bad code!'"),
            (lambda: who_indicator("bad code!"), "invalid indicator_code"),
            (lambda: who_indicators("a;b"), "invalid query"),
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
    def test_a_server_error_is_retryable_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        with pytest.raises(ToolFailure) as caught:
            who_series("WHOSIS_000001")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable

    @patch(HTTP_OPEN)
    def test_a_404_on_an_indicators_series_is_not_found(self, mock_urlopen):
        # Each indicator is an entity set of its own (/api/{code}).
        mock_urlopen.side_effect = http_error(404, "Not Found")

        with pytest.raises(ToolFailure) as caught:
            who_series("NOPE_1")

        assert caught.value.error.type == "not_found"
        assert caught.value.error.message == (
            "WHO GHO has no indicator NOPE_1; search for one with who_indicators"
        )


@pytest.mark.parametrize(
    "call", [lambda: who_indicators("life"), lambda: who_indicator("WHOSIS_000001")]
)
@patch(HTTP_OPEN)
def test_a_404_on_the_indicator_collection_is_an_endpoint_that_moved(mock_urlopen, call):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    with pytest.raises(ToolFailure) as caught:
        call()

    assert caught.value.error.type == "upstream"
    assert caught.value.error.message.startswith("WHO GHO: endpoint not found (HTTP 404)")


@pytest.mark.parametrize(
    "call",
    [
        lambda: who_indicators("life"),
        lambda: who_indicator("WHOSIS_000001"),
        lambda: who_series("WHOSIS_000001", dim1="SEX_MLE"),
    ],
)
@patch(HTTP_OPEN)
def test_a_query_the_service_refuses_is_a_validation_error_in_its_words(mock_urlopen, call):
    # OData's error object (https://docs.oasis-open.org/odata/odata-json-format/v4.01/
    # odata-json-format-v4.01.html#sec_ErrorResponse), with a 400 for a request it refuses.
    mock_urlopen.side_effect = http_error(
        400,
        "Bad Request",
        body=b'{"error": {"code": "", "message": "The query specified in the URI is not valid. '
        b"Could not find a property named 'Dim9' on type 'Default.FACT'.\"}}",
    )

    with pytest.raises(ToolFailure) as caught:
        call()

    assert caught.value.error.type == "validation_error"
    assert caught.value.error.message == (
        "The query specified in the URI is not valid. Could not find a property named 'Dim9' on "
        "type 'Default.FACT'; check the code and the filters (who_indicators lists the codes)"
    )
