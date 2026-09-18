"""Tests for toolkit/tools/_geo.py."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest

from ai_arch_toolkit.toolkit.tools._geo import (
    country_info,
    distance_between,
    geocode,
    ip_lookup,
    reverse_geocode,
    timezone_lookup,
)
from ai_arch_toolkit.toolkit.tools._osm import osm_search_place
from tests.toolkit.http_fakes import HTTP_OPEN, respond


class TestGeocode:
    @patch(HTTP_OPEN)
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
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
            }
        )
        result = geocode("Tokyo")
        assert "Tokyo" in result
        assert "Japan" in result
        assert "35.6762" in result
        assert "13,960,000" in result
        assert "Asia/Tokyo" in result

    @patch(HTTP_OPEN)
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"results": None})
        result = geocode("Nonexistentville")
        assert "No results" in result

    @patch(HTTP_OPEN)
    def test_no_admin(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "results": [
                    {
                        "name": "Monaco",
                        "country": "Monaco",
                        "latitude": 43.73,
                        "longitude": 7.42,
                    }
                ]
            }
        )
        result = geocode("Monaco")
        assert "Monaco, Monaco" in result

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = geocode("Tokyo")
        assert "failed" in result.lower()


class TestIpLookup:
    # Response shapes from https://ipwhois.io/documentation (free endpoint https://ipwho.is/{IP}).
    @patch(HTTP_OPEN)
    def test_returns_info(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "ip": "8.8.8.8",
                "success": True,
                "type": "IPv4",
                "country": "United States",
                "region": "California",
                "city": "Mountain View",
                "latitude": 37.386,
                "longitude": -122.084,
                "connection": {"asn": 15169, "org": "Google LLC", "isp": "Google LLC"},
                "timezone": {"id": "America/Los_Angeles", "utc": "-07:00"},
            }
        )
        result = ip_lookup("8.8.8.8")
        assert result == (
            "IP: 8.8.8.8\n"
            "Location: Mountain View, California, United States\n"
            "Coordinates: 37.386°N, -122.084°E\n"
            "Timezone: America/Los_Angeles\n"
            "ISP: Google LLC\n"
            "Organization: Google LLC"
        )

    @patch(HTTP_OPEN)
    def test_the_lookup_goes_over_https_to_ipwhois(self, mock_urlopen):
        mock_urlopen.return_value = respond({"success": False, "message": "x"})

        ip_lookup("2001:4860:4860::8888")

        url = mock_urlopen.call_args.args[0].full_url
        assert url == "https://ipwho.is/2001:4860:4860::8888"

    @patch(HTTP_OPEN)
    def test_failed_status(self, mock_urlopen):
        mock_urlopen.return_value = respond({"success": False, "message": "Reserved range"})
        result = ip_lookup("10.0.0.1")
        assert result == "IP lookup failed: Reserved range"

    @patch(HTTP_OPEN)
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = ip_lookup("8.8.8.8")
        assert "failed" in result.lower()


class TestReverseGeocode:
    @patch(HTTP_OPEN)
    def test_returns_location(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "display_name": "Tokyo, Japan",
                "address": {
                    "city": "Tokyo",
                    "state": "Tokyo",
                    "country": "Japan",
                },
            }
        )
        result = reverse_geocode(35.6762, 139.6503)
        assert "Tokyo, Japan" in result
        assert "City: Tokyo" in result
        assert "Country: Japan" in result

    def test_invalid_coordinates(self):
        result = reverse_geocode(100.0, 10.0)
        assert "out of range" in result.lower()


class TestTimezoneLookup:
    @patch(HTTP_OPEN)
    def test_returns_timezone(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "timezone": "Asia/Tokyo",
                "utc_offset_seconds": 32400,
            }
        )
        result = timezone_lookup(35.6762, 139.6503)
        assert "Asia/Tokyo" in result
        assert "UTC+09:00" in result

    @patch(HTTP_OPEN)
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = timezone_lookup(35.6762, 139.6503)
        assert "failed" in result.lower()


class TestDistanceBetween:
    def test_distance_in_km(self):
        result = distance_between(0.0, 0.0, 0.0, 1.0)
        assert "111." in result
        assert result.endswith(" km")

    def test_distance_in_miles(self):
        result = distance_between(0.0, 0.0, 0.0, 1.0, unit="mi")
        assert "69." in result
        assert result.endswith(" mi")

    def test_invalid_unit(self):
        result = distance_between(0.0, 0.0, 0.0, 1.0, unit="meters")
        assert "Invalid unit" in result


class TestCountryInfo:
    @patch(HTTP_OPEN)
    def test_returns_info(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            [
                {
                    "name": {"common": "Japan", "official": "Japan"},
                    "capital": ["Tokyo"],
                    "population": 125800000,
                    "area": 377975,
                    "region": "Asia",
                    "subregion": "Eastern Asia",
                    "languages": {"jpn": "Japanese"},
                    "currencies": {"JPY": {"name": "Japanese yen", "symbol": "¥"}},
                    "timezones": ["UTC+09:00"],
                }
            ]
        )
        result = country_info("Japan")
        assert "Japan" in result
        assert "Tokyo" in result
        assert "125,800,000" in result
        assert "Japanese" in result
        assert "yen" in result
        assert "Eastern Asia" in result

    @patch(HTTP_OPEN)
    def test_country_not_found(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError("url", 404, "Not Found", {}, BytesIO())
        result = country_info("Xyzland")
        assert "not found" in result.lower()

    @patch(HTTP_OPEN)
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = country_info("Japan")
        assert "failed" in result.lower()


@pytest.mark.parametrize("ip", ["", "a b?c", "not-an-ip"])
@patch(HTTP_OPEN)
def test_ip_lookup_rejects_invalid_ip_before_request(mock_urlopen, ip):
    mock_urlopen.return_value = respond({"status": "success"})
    result = ip_lookup(ip)
    assert "IP lookup failed" in result
    mock_urlopen.assert_not_called()


@patch(HTTP_OPEN)
def test_reverse_geocode_shares_nominatims_one_request_per_second_with_the_osm_tools(
    mock_urlopen, throttle_waits
):
    mock_urlopen.side_effect = [respond([]), respond({"display_name": "Lisboa"})]

    osm_search_place("Lisbon")
    reverse_geocode(38.7, -9.1)

    assert throttle_waits[0] == 0
    assert 1.0 < throttle_waits[1] <= 1.1
