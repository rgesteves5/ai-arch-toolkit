"""Tests for toolkit/tools/_world_bank.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._world_bank import (
    world_bank_compare,
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


_TOPIC = {
    "id": "3",
    "value": "Economy & Growth",
    "sourceNote": "Economic indicators and growth measures.",
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
    "sourceNote": "Inflation as measured by the consumer price index reflects annual change.",
    "sourceOrganization": "International Monetary Fund, International Financial Statistics.",
    "topics": [{"id": "3", "value": "Economy & Growth"}],
}
_SERIES_POINT = {
    "indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
    "country": {"id": "PT", "value": "Portugal"},
    "countryiso3code": "PRT",
    "date": "2023",
    "value": 10578174,
    "unit": "",
    "obs_status": "",
    "decimal": 0,
}
_SPAIN_POINT = {
    "indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
    "country": {"id": "ES", "value": "Spain"},
    "countryiso3code": "ESP",
    "date": "2023",
    "value": 48352528,
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
    def test_topics(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_TOPIC], per_page=2, total=21))

        result = world_bank_topics(max_results=2)

        assert "World Bank topics (page 1/1, per_page 2, total 21):" in result
        assert "Economy & Growth (3)" in result
        assert "Economic indicators and growth measures." in result
        assert _called_params(mock_urlopen)["per_page"] == ["2"]

    @patch(HTTP_OPEN)
    def test_sources(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_SOURCE], per_page=2, total=71))

        result = world_bank_sources(max_results=2)

        assert "World Development Indicators (2) [WDI]" in result
        assert "last updated: 2026-04-08" in result
        assert "data: Y | metadata: Y" in result

    @patch(HTTP_OPEN)
    def test_countries_filters_locally(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            _payload(
                [
                    _COUNTRY,
                    {
                        **_COUNTRY,
                        "id": "ESP",
                        "iso2Code": "ES",
                        "name": "Spain",
                        "capitalCity": "Madrid",
                    },
                ],
                per_page=500,
                total=2,
            )
        )

        result = world_bank_countries(query="port", region="ECS", income_level="HIC")

        assert "Portugal (PRT)" in result
        assert "Spain" not in result
        assert "ISO2: PT" in result
        assert "region: Europe & Central Asia (ECS)" in result

        params = _called_params(mock_urlopen)
        assert params["per_page"] == ["500"]
        assert params["page"] == ["1"]

    @patch(HTTP_OPEN)
    def test_invalid_catalog_options_do_not_call_api(self, mock_urlopen):
        _refused(lambda: world_bank_topics(page=0), "invalid page 0")
        _refused(lambda: world_bank_sources(page=0), "invalid page 0")
        _refused(lambda: world_bank_countries(page=0), "invalid page 0")
        mock_urlopen.assert_not_called()


class TestWorldBankIndicators:
    @patch(HTTP_OPEN)
    def test_browses_indicators_by_topic(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_INDICATOR], per_page=3, total=306))

        result = world_bank_indicators(topic="3", max_results=3)

        assert "World Bank indicators" in result
        assert "FP.CPI.TOTL.ZG — Inflation, consumer prices" in result
        assert "source: World Development Indicators (2)" in result
        assert "topics: Economy & Growth (3)" in result
        assert urlparse(_called_request(mock_urlopen).full_url).path == "/v2/topic/3/indicator"

    @patch(HTTP_OPEN)
    def test_searches_indicators_client_side(self, mock_urlopen):
        first_page = _payload(
            [
                {
                    **_INDICATOR,
                    "id": "SP.POP.TOTL",
                    "name": "Population, total",
                    "sourceNote": "Total population.",
                },
                _INDICATOR,
            ],
            page=1,
            pages=1,
            per_page=1000,
            total=2,
        )
        mock_urlopen.return_value = respond(first_page)

        result = world_bank_indicators(query="inflation consumer", max_results=5, scan_pages=2)

        assert "FP.CPI.TOTL.ZG — Inflation, consumer prices" in result
        assert "SP.POP.TOTL" not in result
        assert "scanned_pages: 2" in result
        assert _called_params(mock_urlopen)["per_page"] == ["1000"]

    @patch(HTTP_OPEN)
    def test_indicator_lookup(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_INDICATOR]))

        result = world_bank_indicator("FP.CPI.TOTL.ZG")

        assert result.startswith("World Bank indicator FP.CPI.TOTL.ZG:")
        assert "ID: FP.CPI.TOTL.ZG" in result
        assert "Definition: Inflation as measured" in result
        assert "Source organization: International Monetary Fund" in result

    @patch(HTTP_OPEN)
    def test_invalid_indicator_options_do_not_call_api(self, mock_urlopen):
        _refused(lambda: world_bank_indicators(page=0), "invalid page 0")
        _refused(lambda: world_bank_indicators(scan_pages=0), "invalid scan_pages 0")
        _refused(lambda: world_bank_indicator("bad/id"), "invalid indicator ID 'bad/id'")
        mock_urlopen.assert_not_called()

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
        assert _failure(lambda: world_bank_indicator("NO.SUCH")).type == "not_found"

        mock_urlopen.side_effect = http_error(502, "Bad Gateway")
        error = _failure(lambda: world_bank_indicator("NO.SUCH"))
        assert error.type == "upstream"
        assert error.retryable

    @patch(HTTP_OPEN)
    def test_no_matches_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([]))

        assert world_bank_indicators(topic="3") == "No World Bank indicators found."


class TestWorldBankSeries:
    @patch(HTTP_OPEN)
    def test_series(self, mock_urlopen):
        mock_urlopen.return_value = respond(_payload([_SERIES_POINT], per_page=5, total=1))

        result = world_bank_series("PRT", "SP.POP.TOTL", start_year="2020", end_year="2023")

        assert "World Bank series" in result
        assert "SP.POP.TOTL — Population, total for Portugal (PRT)" in result
        assert "2023: 10578174" in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path == "/v2/country/PRT/indicator/SP.POP.TOTL"
        assert _called_params(mock_urlopen)["date"] == ["2020:2023"]

    @patch(HTTP_OPEN)
    def test_compare(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            _payload([_SERIES_POINT, _SPAIN_POINT], per_page=100, total=2)
        )

        result = world_bank_compare("SP.POP.TOTL", "PRT, ESP", year="2023")

        assert "SP.POP.TOTL — Population, total comparison:" in result
        assert "Portugal (PRT): 10578174" in result
        assert "Spain (ESP): 48352528" in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path == "/v2/country/PRT;ESP/indicator/SP.POP.TOTL"
        assert _called_params(mock_urlopen)["date"] == ["2023:2023"]

    @patch(HTTP_OPEN)
    def test_invalid_series_options_do_not_call_api(self, mock_urlopen):
        _refused(lambda: world_bank_series("bad/code", "SP.POP.TOTL"), "invalid country code")
        _refused(lambda: world_bank_series("PRT", "bad/id"), "invalid indicator ID")
        _refused(lambda: world_bank_series("PRT", "SP.POP.TOTL", "20"), "invalid start_year")
        _refused(
            lambda: world_bank_series("PRT", "SP.POP.TOTL", "2024", "2020"),
            "start_year 2024 is after end_year 2020",
        )
        _refused(lambda: world_bank_series("PRT", "SP.POP.TOTL", page=0), "invalid page 0")
        _refused(lambda: world_bank_compare("bad/id", "PRT"), "invalid indicator ID")
        _refused(lambda: world_bank_compare("SP.POP.TOTL", ""), "no valid country code")
        _refused(
            lambda: world_bank_compare("SP.POP.TOTL", "A,B,C,D,E,F,G,H,I,J,K"),
            "compare at most 10 per call",
        )
        _refused(lambda: world_bank_compare("SP.POP.TOTL", "PRT", year="23"), "invalid year '23'")
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
# (api.worldbank.org, 2026-09-29).
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
        lambda: world_bank_series("PRT", "NOT.AN.INDICATOR"),
        lambda: world_bank_compare("NOT.AN.INDICATOR", "PRT,ESP"),
        lambda: world_bank_indicator("NOT.AN.INDICATOR"),
        lambda: world_bank_indicators(source="99999"),
        lambda: world_bank_indicators(query="gdp", scan_pages=2),
        world_bank_topics,
        world_bank_sources,
        world_bank_countries,
    ],
)
@patch(HTTP_OPEN)
def test_an_error_the_api_reports_is_the_tools_error_not_an_empty_page(mock_urlopen, call):
    mock_urlopen.return_value = respond(_INVALID_VALUE)

    error = _failure(call)

    assert error.type == "upstream"
    assert error.message == "Invalid value: The provided parameter value is not valid"
    assert mock_urlopen.call_count == 1


@patch(HTTP_OPEN)
def test_every_message_the_api_reports_is_kept(mock_urlopen):
    mock_urlopen.return_value = respond(
        [
            {
                "message": [
                    {"id": "120", "key": "Invalid value", "value": "Bad country"},
                    {"id": "175", "key": "Invalid format", "value": "Bad indicator"},
                ]
            }
        ]
    )

    assert _failure(lambda: world_bank_series("PRT", "SP.POP.TOTL")).message == (
        "Invalid value: Bad country; Invalid format: Bad indicator"
    )
