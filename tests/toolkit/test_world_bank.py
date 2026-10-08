"""Tests for toolkit/tools/_world_bank.py."""

from __future__ import annotations

import urllib.error
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolFailure, ToolGroup, ToolResult
from ai_arch_toolkit.toolkit.tools._world_bank import (
    world_bank_countries,
    world_bank_indicator,
    world_bank_indicators,
    world_bank_series,
    world_bank_sources,
    world_bank_topics,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(call):
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value.error


def _refused(call, words):
    error = _failure(call)
    assert error.type == "validation_error"
    assert words in error.message


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok
    assert isinstance(result.value, str)
    return result.value


_TOPIC = {
    "id": "3",
    "value": "Economy & Growth",
    "sourceNote": "Economic indicators and growth measures. " * 30,
}
_SOURCE = {
    "id": "2",
    "lastupdated": "2026-04-08",
    "name": "World Development Indicators",
    "code": "WDI",
    "description": "Primary World Bank development indicators.",
    "dataavailability": "Y",
    "metadataavailability": "Y",
}
_COUNTRY = {
    "id": "PRT",
    "iso2Code": "PT",
    "name": "Portugal",
    "region": {"id": "ECS", "value": "Europe & Central Asia"},
    "incomeLevel": {"id": "HIC", "value": "High income"},
    "lendingType": {"id": "LNX", "value": "Not classified"},
    "capitalCity": "Lisbon",
    "longitude": "-9.13552",
    "latitude": "38.7072",
}
_INDICATOR = {
    "id": "FP.CPI.TOTL.ZG",
    "name": "Inflation, consumer prices (annual %)",
    "unit": "",
    "source": {"id": "2", "value": "World Development Indicators"},
    "sourceNote": "Inflation as measured by the consumer price index reflects annual change. "
    * 20,
    "sourceOrganization": "International Monetary Fund, International Financial Statistics.",
    "topics": [{"id": str(number), "value": f"Topic {number}"} for number in range(1, 9)],
}


def _point(country: str, iso3: str, name: str, date: str, value: object) -> dict[str, Any]:
    return {
        "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
        "country": {"id": country, "value": name},
        "countryiso3code": iso3,
        "date": date,
        "value": value,
        "unit": "",
        "obs_status": "",
        "decimal": 0,
    }


def _payload(items, *, page=1, pages=1, per_page=50, total=None):
    return [
        {"page": page, "pages": pages, "per_page": str(per_page), "total": total or len(items)},
        items,
    ]


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestWorldBankCatalog:
    @patch(HTTP_OPEN)
    def test_topics_read_on_page_by_page_with_their_whole_notes(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            _payload([_TOPIC, _TOPIC], page=2, pages=11, per_page=2, total=21)
        )

        result = world_bank_topics(max_results=2, page=2)
        text = _text(result)

        assert text.splitlines()[:2] == [
            "World Bank topics (world_bank_indicators(topic=ID) lists one's indicators):",
            "3. Economy & Growth (ID 3)",
        ]
        assert ("Economic indicators and growth measures. " * 30).strip() in text  # whole
        assert text.endswith("[results 3-4 of 21 | next: page=3, max_results=2]")
        assert result.metadata["window"]["next_call"] == {"page": 3, "max_results": 2}
        assert _called_params(mock_urlopen)["per_page"] == ["2"]
        assert _called_params(mock_urlopen)["page"] == ["2"]

    @patch(HTTP_OPEN)
    def test_the_next_page_keeps_the_pages_size(self, mock_urlopen):
        # A page counts from its size: "next: page=2" called with the default 100 per page read
        # "[no results from 101 of 64 | end]", and skipped results 11-64.
        mock_urlopen.return_value = respond(
            _payload([_TOPIC] * 10, page=1, pages=7, per_page=10, total=64)
        )

        result = world_bank_topics(max_results=10)

        assert result.metadata["window"]["next_call"] == {"page": 2, "max_results": 10}

    @patch(HTTP_OPEN)
    def test_sources(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_SOURCE], per_page=2, total=1))

        text = _text(world_bank_sources(max_results=2))

        assert "1. World Development Indicators (ID 2, code WDI)" in text
        assert "last updated: 2026-04-08 | data: Y | metadata: Y" in text

    @patch(HTTP_OPEN)
    def test_countries_filter_locally_and_read_on(self, mock_urlopen):
        spain = {**_COUNTRY, "id": "ESP", "iso2Code": "ES", "name": "Spain", "capitalCity": "x"}
        mock_urlopen.return_value = respond(_payload([_COUNTRY, spain], per_page=1000, total=2))

        text = _text(world_bank_countries(query="port", region="ECS", income_level="HIC"))

        assert "1. Portugal (PRT, ISO2 PT)" in text
        assert "Spain" not in text
        assert "region: Europe & Central Asia (ECS)" in text
        params = _called_params(mock_urlopen)
        assert params["per_page"] == ["1000"]
        assert params["page"] == ["1"]

    @patch(HTTP_OPEN)
    def test_filtered_countries_read_on_by_page(self, mock_urlopen):
        countries = [
            {**_COUNTRY, "id": f"C{number:02d}", "name": f"Country {number}"}
            for number in range(1, 8)
        ]
        mock_urlopen.return_value = respond(_payload(countries, per_page=1000))

        text = _text(world_bank_countries(region="ECS", max_results=3, page=2))

        assert text.splitlines()[1] == "4. Country 4 (C04, ISO2 PT)"
        assert text.endswith("[results 4-6 of 7 | next: page=3, max_results=3]")

    @patch(HTTP_OPEN)
    def test_no_countries_say_so_with_the_filters(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_COUNTRY], per_page=1000))

        assert _text(world_bank_countries(query="zzz")) == (
            "No World Bank country or aggregate matches query='zzz'."
        )


class TestWorldBankIndicators:
    @patch(HTTP_OPEN)
    def test_browses_indicators_by_topic(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            _payload([_INDICATOR], per_page=3, total=306, pages=102)
        )

        text = _text(world_bank_indicators(topic="3", max_results=3))

        assert text.splitlines()[1] == (
            "1. FP.CPI.TOTL.ZG: Inflation, consumer prices (annual %) | source: World "
            "Development Indicators (2) | topics: Topic 1 (1), Topic 2 (2), Topic 3 (3), Topic 4 "
            "(4), Topic 5 (5), Topic 6 (6), Topic 7 (7), Topic 8 (8)"
        )
        assert text.endswith("[results 1-1 of 306 | next: page=2, max_results=3]")
        assert urlparse(_called_request(mock_urlopen).full_url).path == "/v2/topic/3/indicator"

    @patch(HTTP_OPEN)
    def test_a_search_ranks_the_scanned_catalogue_and_pages_its_matches(self, mock_urlopen):
        matches = [
            {**_INDICATOR, "id": f"CPI.{number}", "name": f"Consumer prices, inflation {number}"}
            for number in range(1, 6)
        ]
        other = {**_INDICATOR, "id": "SP.POP.TOTL", "name": "Population", "sourceNote": "x"}
        mock_urlopen.return_value = respond(
            _payload([other, *matches], page=1, pages=30, per_page=1000, total=29313)
        )

        result = world_bank_indicators(query="inflation", max_results=2, page=2, scan_pages=1)
        text = _text(result)

        assert text.splitlines()[0] == (
            "World Bank indicators that match 'inflation' (searched 6 of 29313 indicators; "
            "scan_pages=30 searches them all; world_bank_indicator gives a definition):"
        )
        assert text.splitlines()[1].startswith("3. CPI.3: Consumer prices, inflation 3")
        assert "SP.POP.TOTL" not in text
        assert text.endswith("[results 3-4 of 5 | next: page=3, max_results=2]")
        assert _called_params(mock_urlopen)["page"] == ["1"]
        assert _called_params(mock_urlopen)["per_page"] == ["1000"]

    @patch(HTTP_OPEN)
    def test_a_search_with_no_match_says_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_INDICATOR], per_page=1000))

        assert _text(world_bank_indicators(query="zzqq", scan_pages=1)) == (
            "No World Bank indicators that match 'zzqq' (searched 1 of 1 indicators)."
        )

    @patch(HTTP_OPEN)
    def test_indicator_lookup_gives_the_whole_definition(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_INDICATOR]))

        text = _text(world_bank_indicator("FP.CPI.TOTL.ZG"))

        assert text.startswith(
            "World Bank indicator FP.CPI.TOTL.ZG: Inflation, consumer prices (annual %)"
        )
        assert _INDICATOR["sourceNote"].strip() in text  # not cut at 500 characters
        assert "Source organization: International Monetary Fund" in text
        assert "Topic 8 (8)" in text  # every topic, not the first 6

    @patch(HTTP_OPEN)
    def test_invalid_indicator_options_do_not_call_api(self, mock_urlopen):
        _refused(lambda: world_bank_indicator("bad/id"), "invalid indicator ID 'bad/id'")
        mock_urlopen.assert_not_called()

    def test_the_limits_are_in_the_schema(self):
        group = ToolGroup(world_bank_indicators, world_bank_series, world_bank_topics)
        for name, args in (
            ("world_bank_topics", {"page": 0}),
            ("world_bank_topics", {"max_results": 101}),
            ("world_bank_indicators", {"scan_pages": 31}),
            ("world_bank_series", {"country": "PRT", "indicator": "X", "max_results": 0}),
        ):
            result = group.execute(ToolCall(id="c", name=name, input=args))
            assert not result.ok and result.error is not None
            assert result.error.type == "validation_error", args

    @patch(HTTP_OPEN)
    def test_an_indicator_the_api_does_not_list_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([]))

        error = _failure(lambda: world_bank_indicator("NO.SUCH"))

        assert error.type == "not_found"
        assert "no indicator NO.SUCH" in error.message
        assert "world_bank_indicators" in error.message

    @patch(HTTP_OPEN)
    def test_a_404_indicator_is_not_found_and_other_statuses_stay_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")
        error = _failure(lambda: world_bank_indicator("NO.SUCH"))
        assert error.type == "not_found"
        assert error.message == (
            "the World Bank has no indicator NO.SUCH; search for one with world_bank_indicators"
        )

        mock_urlopen.side_effect = http_error(502, "Bad Gateway")
        error = _failure(lambda: world_bank_indicator("NO.SUCH"))
        assert error.type == "upstream"
        assert error.retryable

    @patch(HTTP_OPEN)
    def test_no_indicators_under_a_topic_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([]))

        assert _text(world_bank_indicators(topic="3")) == (
            "No World Bank indicators are listed under topic 3."
        )


class TestWorldBankSeries:
    @patch(HTTP_OPEN)
    def test_series_values_are_whole_numbers_not_scientific_notation(self, mock_urlopen):
        # "2.91849e+13", rounded to six digits without a word.
        mock_urlopen.return_value = respond(
            _payload(
                [
                    _point("PT", "PRT", "Portugal", "2023", 2.91849123456e13),
                    _point("PT", "PRT", "Portugal", "2022", None),
                    _point("PT", "PRT", "Portugal", "2021M07", 1.23456789012),
                ],
                per_page=100,
            )
        )

        text = _text(world_bank_series("PRT", "NY.GDP.MKTP.CD", start_year="2020"))

        assert text.splitlines() == [
            "World Bank, NY.GDP.MKTP.CD: GDP (current US$):",
            "1. Portugal (PRT), 2023: 29184912345600.0",
            "2. Portugal (PRT), 2022: no value",
            "3. Portugal (PRT), 2021-07: 1.23456789012",
        ]
        assert urlparse(_called_request(mock_urlopen).full_url).path == (
            "/v2/country/PRT/indicator/NY.GDP.MKTP.CD"
        )
        assert _called_params(mock_urlopen)["date"] == ["2020:2020"]

    @patch(HTTP_OPEN)
    def test_several_countries_compare_and_read_on_to_page_two(self, mock_urlopen):
        # world_bank_compare read page 1 only; world_bank_series takes several countries now.
        page_two = [
            _point("ES", "ESP", "Spain", str(year), 1000 + year) for year in range(2019, 2021)
        ]
        mock_urlopen.return_value = respond(
            _payload(page_two, page=2, pages=3, per_page=2, total=6)
        )

        result = world_bank_series(
            "PRT, ESP;DEU",
            "NY.GDP.MKTP.CD",
            start_year="2019",
            end_year="2020",
            max_results=2,
            page=2,
        )
        text = _text(result)

        assert urlparse(_called_request(mock_urlopen).full_url).path == (
            "/v2/country/PRT;ESP;DEU/indicator/NY.GDP.MKTP.CD"
        )
        assert _called_params(mock_urlopen)["page"] == ["2"]
        assert text.splitlines()[1] == "3. Spain (ESP), 2019: 3019"
        assert text.endswith("[results 3-4 of 6 | next: page=3, max_results=2]")
        assert result.metadata["window"]["next_call"] == {"page": 3, "max_results": 2}

    @patch(HTTP_OPEN)
    def test_no_observations_say_so(self, mock_urlopen):
        # The API answers a series with no data with a null page (2026-09-29).
        mock_urlopen.return_value = respond(
            [{"page": 1, "pages": 0, "per_page": "100", "total": 0}, None]
        )

        assert _text(world_bank_series("PRT", "SP.POP.TOTL", "1900", "1901")) == (
            "No World Bank observations of SP.POP.TOTL for PRT from 1900 to 1901."
        )

    @patch(HTTP_OPEN)
    def test_invalid_series_options_do_not_call_api(self, mock_urlopen):
        _refused(lambda: world_bank_series("bad/code", "SP.POP.TOTL"), "invalid country code")
        _refused(lambda: world_bank_series("", "SP.POP.TOTL"), "invalid country code")
        _refused(lambda: world_bank_series("PRT,all", "SP.POP.TOTL"), "'all' alone")
        _refused(lambda: world_bank_series("PRT", "bad/id"), "invalid indicator ID")
        _refused(lambda: world_bank_series("PRT", "SP.POP.TOTL", "20"), "invalid start_year")
        _refused(
            lambda: world_bank_series("PRT", "SP.POP.TOTL", "2024", "2020"),
            "start_year 2024 is after end_year 2020",
        )
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_api_failure_and_parse_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.worldbank.org/v2/topic",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        error = _failure(world_bank_topics)
        assert error.type == "rate_limited"
        assert "rate limited" in error.message

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = respond("not json")
        error = _failure(world_bank_topics)
        assert error.type == "upstream"
        assert "could not parse" in error.message

    @patch(HTTP_OPEN)
    def test_an_unexpected_shape_is_upstream(self, mock_urlopen):
        mock_urlopen.return_value = respond([{"page": 1}])

        error = _failure(world_bank_topics)

        assert error.type == "upstream"
        assert "unexpected World Bank response shape" in error.message


# What the API answers, with HTTP 200, for an indicator or source ID it does not know
# (api.worldbank.org, 2026-09-29): error 120 (https://datahelpdesk.worldbank.org/knowledgebase/
# articles/898620-api-error-codes).
_INVALID_VALUE = [
    {
        "message": [
            {
                "id": "120",
                "key": "Invalid value",
                "value": "The provided parameter value is not valid",
            }
        ]
    }
]


@pytest.mark.parametrize(
    "call",
    [
        lambda: world_bank_indicators(source="99999"),
        lambda: world_bank_indicators(query="gdp", scan_pages=2),
        world_bank_topics,
        world_bank_sources,
        world_bank_countries,
    ],
)
@patch(HTTP_OPEN)
def test_an_invalid_value_on_a_list_is_a_validation_error_in_its_words(mock_urlopen, call):
    mock_urlopen.return_value = respond(_INVALID_VALUE)

    error = _failure(call)

    assert error.type == "validation_error"
    assert error.message == (
        "Invalid value: The provided parameter value is not valid; check the codes and IDs "
        "given (world_bank_countries, world_bank_indicators, world_bank_sources and "
        "world_bank_topics list them)"
    )
    assert mock_urlopen.call_count == 1


@patch(HTTP_OPEN)
def test_a_series_of_an_unknown_indicator_or_country_is_not_found(mock_urlopen):
    mock_urlopen.return_value = respond(_INVALID_VALUE)

    error = _failure(lambda: world_bank_series("PRT;ESP", "NOT.AN.INDICATOR"))

    assert error.type == "not_found"
    assert error.message == (
        "the World Bank has no indicator NOT.AN.INDICATOR or no country or aggregate among "
        "PRT;ESP (error 120, Invalid value: The provided parameter value is not valid); find "
        "the indicator with world_bank_indicators and the codes with world_bank_countries"
    )


@patch(HTTP_OPEN)
def test_a_service_the_api_says_is_unavailable_is_worth_a_retry(mock_urlopen):
    mock_urlopen.return_value = respond(
        [
            {
                "message": [
                    {
                        "id": "105",
                        "key": "Service currently unavailable",
                        "value": "The requested service is temporarily unavailable.",
                    }
                ]
            }
        ]
    )

    error = _failure(world_bank_topics)

    assert error.type == "upstream"
    assert error.retryable
    assert error.message == (
        "Service currently unavailable: The requested service is temporarily unavailable; try "
        "again later"
    )


@patch(HTTP_OPEN)
def test_every_message_the_api_reports_is_kept_with_a_next_step(mock_urlopen):
    mock_urlopen.return_value = respond(
        [
            {
                "message": [
                    {"id": "199", "key": "Unexpected error", "value": "Bad country"},
                    {"id": "175", "key": "Invalid format", "value": "Bad indicator"},
                ]
            }
        ]
    )

    error = _failure(lambda: world_bank_series("PRT", "SP.POP.TOTL"))

    assert error.type == "upstream"
    assert error.message == (
        "Unexpected error: Bad country; Invalid format: Bad indicator; check the codes and IDs "
        "given (world_bank_countries, world_bank_indicators, world_bank_sources and "
        "world_bank_topics list them), or try again later"
    )


# What the API answers, with HTTP 200, for a data query of an indicator it does not hold
# (api.worldbank.org, seen by the review of T08a, 2026-10-08): error 175, which its error table
# (https://datahelpdesk.worldbank.org/knowledgebase/articles/898620-api-error-codes) does not
# list.
_INDICATOR_NOT_FOUND = [
    {
        "message": [
            {
                "id": "175",
                "key": "Invalid format",
                "value": "The indicator was not found. It may have been deleted or archived.",
            }
        ]
    }
]


@patch(HTTP_OPEN)
def test_a_series_of_an_indicator_the_api_does_not_hold_is_not_found(mock_urlopen):
    mock_urlopen.return_value = respond(_INDICATOR_NOT_FOUND)

    error = _failure(lambda: world_bank_series("PRT", "NOT.AN.INDICATOR"))

    assert error.type == "not_found"
    assert error.message == (
        "the World Bank has no indicator NOT.AN.INDICATOR or no country or aggregate among PRT "
        "(error 175, Invalid format: The indicator was not found. It may have been deleted or "
        "archived); find the indicator with world_bank_indicators and the codes with "
        "world_bank_countries"
    )


@patch(HTTP_OPEN)
def test_an_indicator_lookup_of_one_the_api_does_not_hold_is_not_found(mock_urlopen):
    mock_urlopen.return_value = respond(_INDICATOR_NOT_FOUND)

    error = _failure(lambda: world_bank_indicator("NOT.AN.INDICATOR"))

    assert error.type == "not_found"
    assert error.message.startswith(
        "the World Bank has no indicator NOT.AN.INDICATOR (error 175, Invalid format:"
    )
    assert error.message.endswith("; search for one with world_bank_indicators")


@patch(HTTP_OPEN)
def test_an_indicator_lookup_the_api_calls_an_invalid_value_is_not_found(mock_urlopen):
    # Error 120 is the World Bank's answer, with HTTP 200, for an indicator ID it does not know.
    mock_urlopen.return_value = respond(_INVALID_VALUE)

    error = _failure(lambda: world_bank_indicator("NOT.AN.INDICATOR"))

    assert error.type == "not_found"
    assert error.message == (
        "the World Bank has no indicator NOT.AN.INDICATOR (error 120, Invalid value: The "
        "provided parameter value is not valid); search for one with world_bank_indicators"
    )


@patch(HTTP_OPEN)
def test_another_error_on_an_indicator_lookup_stays_the_sources_words(mock_urlopen):
    mock_urlopen.return_value = respond(
        [{"message": [{"id": "199", "key": "Unexpected error", "value": "is not supported"}]}]
    )

    error = _failure(lambda: world_bank_indicator("SP.POP.TOTL"))

    assert error.type == "upstream"
    assert error.message.startswith("Unexpected error: is not supported; check the codes")


@pytest.mark.parametrize(
    "call",
    [world_bank_topics, lambda: world_bank_series("PRT", "SP.POP.TOTL")],
)
@patch(HTTP_OPEN)
def test_a_404_on_a_list_is_an_endpoint_that_moved(mock_urlopen, call):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    error = _failure(call)

    assert error.type == "upstream"
    assert error.message.startswith("World Bank: endpoint not found (HTTP 404)")
