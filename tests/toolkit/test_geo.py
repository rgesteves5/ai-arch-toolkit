"""Tests for toolkit/tools/_geo.py."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._geo import country_info, geocode, ip_lookup


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestGeocode:
    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({
            "results": [
                {
                    "name": "Tokyo",
                    "country": "Japan",
                    "admin1": "Tokyo",
                    "latitude": 35.6762,
                    "longitude": 139.6503,
                    "population": 13960000,
                    "timezone": "Asia/Tokyo",
                }
            ]
        })
        result = geocode("Tokyo")
        assert "Tokyo" in result
        assert "Japan" in result
        assert "35.6762" in result
        assert "13,960,000" in result
        assert "Asia/Tokyo" in result

    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"results": None})
        result = geocode("Nonexistentville")
        assert "No results" in result

    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_no_admin(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({
            "results": [
                {
                    "name": "Monaco",
                    "country": "Monaco",
                    "latitude": 43.73,
                    "longitude": 7.42,
                }
            ]
        })
        result = geocode("Monaco")
        assert "Monaco, Monaco" in result

    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = geocode("Tokyo")
        assert "failed" in result.lower()


class TestIpLookup:
    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_returns_info(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({
            "status": "success",
            "query": "8.8.8.8",
            "city": "Mountain View",
            "regionName": "California",
            "country": "United States",
            "lat": 37.386,
            "lon": -122.084,
            "timezone": "America/Los_Angeles",
            "isp": "Google LLC",
            "org": "Google LLC",
        })
        result = ip_lookup("8.8.8.8")
        assert "8.8.8.8" in result
        assert "Mountain View" in result
        assert "Google" in result
        assert "America/Los_Angeles" in result

    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_failed_status(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({
            "status": "fail",
            "message": "invalid query",
        })
        result = ip_lookup("not-an-ip")
        assert "failed" in result.lower()
        assert "invalid query" in result

    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = ip_lookup("8.8.8.8")
        assert "failed" in result.lower()


class TestCountryInfo:
    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_returns_info(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen([{
            "name": {"common": "Japan", "official": "Japan"},
            "capital": ["Tokyo"],
            "population": 125800000,
            "area": 377975,
            "region": "Asia",
            "subregion": "Eastern Asia",
            "languages": {"jpn": "Japanese"},
            "currencies": {"JPY": {"name": "Japanese yen", "symbol": "¥"}},
            "timezones": ["UTC+09:00"],
        }])
        result = country_info("Japan")
        assert "Japan" in result
        assert "Tokyo" in result
        assert "125,800,000" in result
        assert "Japanese" in result
        assert "yen" in result
        assert "Eastern Asia" in result

    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_country_not_found(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, BytesIO()
        )
        result = country_info("Xyzland")
        assert "not found" in result.lower()

    @patch("ai_arch_toolkit.toolkit.tools._geo.urllib.request.urlopen")
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = country_info("Japan")
        assert "failed" in result.lower()
