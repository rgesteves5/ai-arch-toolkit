"""Tests for toolkit/tools/_geo.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._geo import (
    country_info,
    distance_between,
    geocode,
    ip_lookup,
    timezone_lookup,
)
from tests.toolkit import geo_answers
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _query(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestGeocode:
    @patch(HTTP_OPEN)
    def test_lists_the_places_labelled_with_signed_coordinates(self, mock_urlopen):
        mock_urlopen.return_value = respond(geo_answers.geocoding(1, name="Tokyo"))

        text = _text(geocode("Tokyo"))

        assert text == (
            "Places named 'Tokyo' (Open-Meteo geocoding):\n"
            "1. Tokyo, State 1, United States (US) | latitude 39.80172, longitude -89.64371 | "
            "time zone America/Chicago | population 116250 | elevation 182.0 m"
        )

    @patch(HTTP_OPEN)
    def test_shows_more_than_three_and_pages_on(self, mock_urlopen):
        # It always showed 3, and nothing told the agent there were more.
        mock_urlopen.return_value = respond(geo_answers.geocoding(6))

        result = geocode("Springfield", max_results=5)

        text = _text(result)
        assert "5. Springfield, State 5" in text
        assert "6. Springfield" not in text
        assert text.endswith("[results 1-5 | next: offset=5]")
        assert _query(mock_urlopen)["count"] == ["6"]  # the page and one more
        assert isinstance(result, ToolResult)
        assert result.metadata["window"]["next_call"] == {"offset": 5}

    @patch(HTTP_OPEN)
    def test_the_next_page_says_the_total_when_the_source_has_no_more(self, mock_urlopen):
        mock_urlopen.return_value = respond(geo_answers.geocoding(7))

        text = _text(geocode("Springfield", max_results=5, offset=5))

        assert _query(mock_urlopen)["count"] == ["11"]
        lines = text.splitlines()
        assert [line.split(" |")[0] for line in lines[1:3]] == [
            "6. Springfield, State 6, United States (US)",
            "7. Springfield, State 7, United States (US)",
        ]
        assert lines[3:] == ["[results 6-7 of 7 | end]"]

    @patch(HTTP_OPEN)
    def test_past_the_hundred_places_it_says_how_to_narrow(self, mock_urlopen):
        mock_urlopen.return_value = respond(geo_answers.geocoding(100))

        text = _text(geocode("Springfield", max_results=10, offset=90))

        assert _query(mock_urlopen)["count"] == ["100"]
        assert "(the source returns no more than 100 results; add the region or country" in text

    @patch(HTTP_OPEN)
    def test_no_place_is_a_success_that_names_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(geo_answers.NO_PLACES)

        assert _text(geocode("Nonexistentville")) == (
            "No places named 'Nonexistentville' in Open-Meteo's geocoding; check the spelling, "
            "or search OpenStreetMap with osm_search_place."
        )

    @patch(HTTP_OPEN)
    def test_open_meteos_refusal_is_a_validation_error_in_its_words(self, mock_urlopen):
        body = b'{"error": true, "reason": "Parameter count must be between 1 and 100."}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(lambda: geocode("Tokyo"))

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "Open-Meteo rejected the request: Parameter count must be between 1 and 100; "
            "correct the argument it names and call again"
        )

    def test_an_empty_name_asks_nothing(self):
        failure = _failure(lambda: geocode("  "))
        assert failure.error.type == "validation_error"

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
                "country_code": "US",
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
            "Location: Mountain View, California, United States (US)\n"
            "Coordinates (approximate): latitude 37.386, longitude -122.084\n"
            "Time zone: America/Los_Angeles (UTC-07:00)\n"
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
    def test_a_reserved_address_is_the_callers_to_change(self, mock_urlopen):
        # It was an upstream failure, which reads as "try again later".
        mock_urlopen.return_value = respond({"success": False, "message": "Reserved range"})
        failure = _failure(lambda: ip_lookup("10.0.0.1"))
        assert failure.error.type == "validation_error"
        assert str(failure) == (
            "ipwho.is cannot locate this address (Reserved range): a private, reserved or "
            "malformed address has no public location; give a public IPv4 or IPv6 address"
        )

    @patch(HTTP_OPEN)
    def test_another_reason_is_the_sources_failure_in_its_words(self, mock_urlopen):
        mock_urlopen.return_value = respond({"success": False, "message": "Server busy"})
        failure = _failure(lambda: ip_lookup("8.8.8.8"))
        assert failure.error.type == "upstream"
        assert str(failure) == "Server busy"

    @patch(HTTP_OPEN)
    def test_the_daily_limit_is_a_rate_limit(self, mock_urlopen):
        body = b'{"success": false, "message": "Rate limit exceeded"}'
        mock_urlopen.side_effect = http_error(429, "Too Many Requests", body=body)
        failure = _failure(lambda: ip_lookup("8.8.8.8"))
        assert failure.error.type == "rate_limited"
        assert "ipwho.is said: Rate limit exceeded" in str(failure)

    @patch(HTTP_OPEN)
    def test_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        failure = _failure(lambda: ip_lookup("8.8.8.8"))
        assert failure.error.type == "upstream"


class TestTimezoneLookup:
    @patch(HTTP_OPEN)
    def test_returns_timezone(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "timezone": "Asia/Tokyo",
                "timezone_abbreviation": "JST",
                "utc_offset_seconds": 32400,
            }
        )
        assert timezone_lookup(35.6762, 139.6503) == (
            "Coordinates: latitude 35.6762, longitude 139.6503\n"
            "Time zone: Asia/Tokyo (JST)\n"
            "UTC offset now: UTC+09:00"
        )

    @patch(HTTP_OPEN)
    def test_an_answer_without_a_time_zone_is_no_answer(self, mock_urlopen):
        mock_urlopen.return_value = respond({"utc_offset_seconds": 0})
        failure = _failure(lambda: timezone_lookup(0.0, 0.0))
        assert failure.error.type == "upstream"
        assert "without a time zone" in str(failure)

    @patch(HTTP_OPEN)
    def test_open_meteos_refusal_keeps_its_reason(self, mock_urlopen):
        body = b'{"error": true, "reason": "Latitude must be in range of -90 to 90."}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)
        failure = _failure(lambda: timezone_lookup(10.0, 10.0))
        assert failure.error.type == "validation_error"
        assert "Latitude must be in range of -90 to 90" in str(failure)

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
