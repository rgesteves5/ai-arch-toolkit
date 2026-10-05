"""Tests for toolkit/tools/_geo.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
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


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


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
        failure = _failure(lambda: geocode("Tokyo"))
        assert failure.error.type == "upstream"
        assert failure.error.retryable
        assert "timed out" in str(failure)


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

        with pytest.raises(ToolFailure):
            ip_lookup("2001:4860:4860::8888")

        url = mock_urlopen.call_args.args[0].full_url
        assert url == "https://ipwho.is/2001:4860:4860::8888"

    @patch(HTTP_OPEN)
    def test_failed_status(self, mock_urlopen):
        mock_urlopen.return_value = respond({"success": False, "message": "Reserved range"})
        failure = _failure(lambda: ip_lookup("10.0.0.1"))
        assert failure.error.type == "upstream"
        assert str(failure) == "ipwho.is could not look up 10.0.0.1: Reserved range."

    @patch(HTTP_OPEN)
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        failure = _failure(lambda: ip_lookup("8.8.8.8"))
        assert failure.error.type == "upstream"


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
        failure = _failure(lambda: reverse_geocode(100.0, 10.0))
        assert failure.error.type == "validation_error"
        assert "latitude out of range" in str(failure)

    @patch(HTTP_OPEN)
    def test_a_place_with_no_address_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"error": "Unable to geocode"})
        assert (
            reverse_geocode(0.0, -30.0)
            == "No reverse geocoding result for coordinates: 0.0, -30.0"
        )


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
        failure = _failure(lambda: timezone_lookup(35.6762, 139.6503))
        assert failure.error.type == "upstream"

    def test_invalid_coordinates(self):
        failure = _failure(lambda: timezone_lookup(10.0, 200.0))
        assert failure.error.type == "validation_error"
        assert "longitude out of range" in str(failure)


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
        failure = _failure(lambda: distance_between(0.0, 0.0, 0.0, 1.0, unit="meters"))
        assert failure.error.type == "validation_error"
        assert "invalid unit" in str(failure)

    def test_invalid_coordinates_say_which_end(self):
        failure = _failure(lambda: distance_between(0.0, 0.0, 91.0, 1.0))
        assert failure.error.type == "validation_error"
        assert str(failure).startswith("end latitude out of range")


def _fact(qid, prop, value, *, label=None, extra=None):
    """One row of the country query: an item value is a URI, anything else a literal."""
    row = {
        "c": {"type": "uri", "value": f"http://www.wikidata.org/entity/{qid}"},
        "prop": {"type": "literal", "value": prop},
        "value": (
            {"type": "uri", "value": f"http://www.wikidata.org/entity/{value}"}
            if value.startswith("Q") and value[1:].isdigit()
            else {"type": "literal", "value": value}
        ),
    }
    if label is not None:
        row["label"] = {"type": "literal", "xml:lang": "en", "value": label}
    if extra is not None:
        row["extra"] = {"type": "literal", "value": extra}
    return row


_JAPAN = [
    _fact("Q17", "name", "Japan"),
    _fact("Q17", "iso2", "JP"),
    _fact("Q17", "iso3", "JPN"),
    _fact("Q17", "capital", "Q1490", label="Tokyo"),
    _fact("Q17", "population", "125800000", extra="2020-10-01T00:00:00Z"),
    _fact("Q17", "population", "123975371", extra="2024-10-01T00:00:00Z"),
    _fact("Q17", "area", "377975000000"),
    _fact("Q17", "continent", "Q48", label="Asia"),
    _fact("Q17", "language", "Q5287", label="Japanese"),
    _fact("Q17", "language", "Q999999"),
    _fact("Q17", "currency", "Q8146", label="Japanese yen", extra="JPY"),
    _fact("Q17", "currency", "Q4916", extra="EUR"),
    _fact("Q17", "timezone", "Q7", label="Asia/Tokyo"),
    _fact("Q17", "timezone", "Q8", label="UTC+09:00"),
    _fact("Q17", "timezone", "Q9", label="UTC\u221201:00"),
]


def _sparql(rows):
    return respond(
        {"head": {"vars": ["c", "prop", "value", "label", "extra"]}, "results": {"bindings": rows}}
    )


class TestCountryInfo:
    @patch(HTTP_OPEN)
    def test_the_search_order_picks_the_country_and_lists_the_others(self, mock_urlopen):
        # Q1 is no country (the query returns nothing for it); a malformed ID never reaches it.
        search = {"search": [{"id": "Q17"}, {"id": "Q1"}, {"id": "../Q2"}, {"id": "Q183"}]}
        germany = [_fact("Q183", "name", "Germany"), _fact("Q183", "iso2", "DE")]
        mock_urlopen.side_effect = [respond(search), _sparql(germany + _JAPAN)]

        lines = country_info(" Japan ").splitlines()

        assert lines[0] == "Japan:"
        assert lines[-1] == "Other matches: Germany (DE)"
        first, second = (call.args[0].full_url for call in mock_urlopen.call_args_list)
        assert first.startswith("https://www.wikidata.org/w/api.php?")
        assert parse_qs(urlparse(first).query)["search"] == ["Japan"]
        assert second.startswith("https://query.wikidata.org/sparql?")
        assert "VALUES ?c { wd:Q17 wd:Q1 wd:Q183 }" in parse_qs(urlparse(second).query)["query"][0]

    @patch(HTTP_OPEN)
    def test_formats_the_latest_population_the_area_in_km2_and_the_utc_offsets(self, mock_urlopen):
        mock_urlopen.side_effect = [respond({"search": [{"id": "Q17"}]}), _sparql(_JAPAN)]

        result = country_info("Japan")

        assert result.splitlines() == [
            "Japan:",
            "  ISO 3166-1: JP, JPN",
            "  Capital: Tokyo",
            "  Population: 123,975,371 (2024)",
            "  Area: 377,975 km²",
            "  Continent: Asia",
            "  Official languages: Japanese",
            "  Currencies: EUR, Japanese yen (JPY)",
            "  Timezones: UTC\u221201:00, UTC+09:00",
            "  Wikidata: https://www.wikidata.org/wiki/Q17",
        ]

    @patch(HTTP_OPEN)
    def test_an_empty_search_asks_no_query(self, mock_urlopen):
        mock_urlopen.return_value = respond({"search": []})

        failure = _failure(lambda: country_info("Xyzland"))
        assert failure.error.type == "not_found"
        assert "'Xyzland'" in str(failure)
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_no_country_among_the_matches(self, mock_urlopen):
        mock_urlopen.side_effect = [respond({"search": [{"id": "Q1"}]}), _sparql([])]

        failure = _failure(lambda: country_info("Xyzland"))
        assert failure.error.type == "not_found"

    @patch(HTTP_OPEN)
    def test_an_invalid_name_asks_nothing(self, mock_urlopen):
        for name in ('Japan" } DELETE', ""):
            failure = _failure(lambda name=name: country_info(name))
            assert failure.error.type == "validation_error"
            assert "invalid name" in str(failure)
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        failure = _failure(lambda: country_info("Japan"))
        assert failure.error.type == "upstream"
        assert str(failure) == "request timed out."

    @patch(HTTP_OPEN)
    def test_an_error_the_search_api_reports_is_not_a_missing_country(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"error": {"code": "ratelimited", "info": "You've exceeded your rate limit."}}
        )

        failure = _failure(lambda: country_info("Japan"))
        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert str(failure) == (
            "the wiki asked to slow down (ratelimited: You've exceeded your rate limit); "
            "try again later"
        )
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_another_error_the_search_api_reports_is_upstream(self, mock_urlopen):
        mock_urlopen.return_value = respond({"error": {"code": "readonly", "info": "Read-only."}})

        failure = _failure(lambda: country_info("Japan"))
        assert failure.error.type == "upstream"
        assert str(failure) == "readonly: Read-only."
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_a_malformed_query_answer_fails_cleanly(self, mock_urlopen):
        bad_row = _fact("Q17", "population", "many")
        mock_urlopen.side_effect = [respond({"search": [{"id": "Q17"}]}), _sparql([bad_row])]

        failure = _failure(lambda: country_info("Japan"))
        assert failure.error.type == "upstream"
        assert str(failure).startswith("could not parse")


@pytest.mark.parametrize("ip", ["", "a b?c", "not-an-ip"])
@patch(HTTP_OPEN)
def test_ip_lookup_rejects_invalid_ip_before_request(mock_urlopen, ip):
    mock_urlopen.return_value = respond({"status": "success"})
    failure = _failure(lambda: ip_lookup(ip))
    assert failure.error.type == "validation_error"
    assert "invalid IP address" in str(failure)
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
