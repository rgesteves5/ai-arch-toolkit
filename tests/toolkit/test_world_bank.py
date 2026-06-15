"""Tests for toolkit/tools/_world_bank.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._world_bank import (
    world_bank_compare,
    world_bank_countries,
    world_bank_indicator,
    world_bank_indicators,
    world_bank_series,
    world_bank_sources,
    world_bank_topics,
)

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


def _mock_urlopen(data: dict | list | str):
    resp = MagicMock()
    if isinstance(data, str):
        resp.read.return_value = data.encode()
    else:
        resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestWorldBankCatalog:
    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_topics(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_payload([_TOPIC], per_page=2, total=21))

        result = world_bank_topics(max_results=2)

        assert "World Bank topics (page 1/1, per_page 2, total 21):" in result
        assert "Economy & Growth (3)" in result
        assert "Economic indicators and growth measures." in result
        assert _called_params(mock_urlopen)["per_page"] == ["2"]

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_sources(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_payload([_SOURCE], per_page=2, total=71))

        result = world_bank_sources(max_results=2)

        assert "World Development Indicators (2) [WDI]" in result
        assert "last updated: 2026-04-08" in result
        assert "data: Y | metadata: Y" in result

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_countries_filters_locally(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_invalid_catalog_options_do_not_call_api(self, mock_urlopen):
        assert "page must" in world_bank_topics(page=0)
        assert "page must" in world_bank_sources(page=0)
        assert "page must" in world_bank_countries(page=0)
        mock_urlopen.assert_not_called()


class TestWorldBankIndicators:
    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_browses_indicators_by_topic(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_payload([_INDICATOR], per_page=3, total=306))

        result = world_bank_indicators(topic="3", max_results=3)

        assert "World Bank indicators" in result
        assert "FP.CPI.TOTL.ZG — Inflation, consumer prices" in result
        assert "source: World Development Indicators (2)" in result
        assert "topics: Economy & Growth (3)" in result
        assert urlparse(_called_request(mock_urlopen).full_url).path == "/v2/topic/3/indicator"

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
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
        mock_urlopen.return_value = _mock_urlopen(first_page)

        result = world_bank_indicators(query="inflation consumer", max_results=5, scan_pages=2)

        assert "FP.CPI.TOTL.ZG — Inflation, consumer prices" in result
        assert "SP.POP.TOTL" not in result
        assert "scanned_pages: 2" in result
        assert _called_params(mock_urlopen)["per_page"] == ["1000"]

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_indicator_lookup(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_payload([_INDICATOR]))

        result = world_bank_indicator("FP.CPI.TOTL.ZG")

        assert result.startswith("World Bank indicator FP.CPI.TOTL.ZG:")
        assert "ID: FP.CPI.TOTL.ZG" in result
        assert "Definition: Inflation as measured" in result
        assert "Source organization: International Monetary Fund" in result

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_invalid_indicator_options_do_not_call_api(self, mock_urlopen):
        assert "page must" in world_bank_indicators(page=0)
        assert "scan_pages must" in world_bank_indicators(scan_pages=0)
        assert "invalid indicator ID" in world_bank_indicator("bad/id")
        mock_urlopen.assert_not_called()


class TestWorldBankSeries:
    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_series(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_payload([_SERIES_POINT], per_page=5, total=1))

        result = world_bank_series("PRT", "SP.POP.TOTL", start_year="2020", end_year="2023")

        assert "World Bank series" in result
        assert "SP.POP.TOTL — Population, total for Portugal (PRT)" in result
        assert "2023: 10578174" in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path == "/v2/country/PRT/indicator/SP.POP.TOTL"
        assert _called_params(mock_urlopen)["date"] == ["2020:2023"]

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_compare(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            _payload([_SERIES_POINT, _SPAIN_POINT], per_page=100, total=2)
        )

        result = world_bank_compare("SP.POP.TOTL", "PRT, ESP", year="2023")

        assert "SP.POP.TOTL — Population, total comparison:" in result
        assert "Portugal (PRT): 10578174" in result
        assert "Spain (ESP): 48352528" in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path == "/v2/country/PRT;ESP/indicator/SP.POP.TOTL"
        assert _called_params(mock_urlopen)["date"] == ["2023:2023"]

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_invalid_series_options_do_not_call_api(self, mock_urlopen):
        assert "invalid country code" in world_bank_series("bad/code", "SP.POP.TOTL")
        assert "invalid indicator ID" in world_bank_series("PRT", "bad/id")
        assert "invalid start_year" in world_bank_series("PRT", "SP.POP.TOTL", "20")
        assert "start_year must" in world_bank_series("PRT", "SP.POP.TOTL", "2024", "2020")
        assert "provide at least one country" in world_bank_compare("SP.POP.TOTL", "")
        assert "at most 10 countries" in world_bank_compare(
            "SP.POP.TOTL",
            "A,B,C,D,E,F,G,H,I,J,K",
        )
        assert "invalid year" in world_bank_compare("SP.POP.TOTL", "PRT", year="23")
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._world_bank.urllib.request.urlopen")
    def test_api_failure_and_parse_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.worldbank.org/v2/topic",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        assert "rate limited" in world_bank_topics()

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _mock_urlopen("not json")
        assert "could not parse" in world_bank_topics()
