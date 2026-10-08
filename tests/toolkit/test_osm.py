"""Tests for toolkit/tools/_osm.py: the Nominatim family, with reverse geocoding fused into
``osm_reverse_geocode`` (T08b, D41)."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit import tools
from ai_arch_toolkit.toolkit.tools._osm import osm_reverse_geocode, osm_search_place
from tests.toolkit import geo_answers
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_PLACE = {
    "place_id": 123,
    "osm_type": "relation",
    "osm_id": 540,
    "display_name": "Lisbon, Portugal",
    "category": "boundary",
    "type": "administrative",
    "importance": "0.7",
    "lat": "38.7077507",
    "lon": "-9.1365919",
    "boundingbox": ["38.6", "38.8", "-9.3", "-9.0"],
    "address": {
        "city": "Lisbon",
        "ISO3166-2-lvl6": "PT-11",
        "postcode": "1100",
        "country": "Portugal",
        "country_code": "pt",
    },
    "extratags": {f"tag{index:02d}": f"value {index}" for index in range(12)},
}


def _invalid(call, *args, **kwargs) -> str:
    """The message of the validation_error ``call`` raises."""
    with pytest.raises(ToolFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.error.type == "validation_error"
    return caught.value.error.message


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestOsmSearchPlace:
    @patch(HTTP_OPEN)
    def test_returns_places_with_their_osm_object_address_box_and_tags(self, mock_urlopen):
        mock_urlopen.return_value = respond([_PLACE])

        text = _text(
            osm_search_place(
                "Lisbon",
                max_results=2,
                country_codes="pt",
                layer="address",
                include_extra_tags=True,
            )
        )

        assert text.splitlines()[:5] == [
            "OpenStreetMap places for 'Lisbon' (Nominatim; data © OpenStreetMap contributors, "
            "ODbL):",
            "1. Lisbon, Portugal",
            "   OSM: relation/540 | type: boundary/administrative | latitude 38.7077507, "
            "longitude -9.1365919",
            "   Address: city: Lisbon | ISO3166-2-lvl6: PT-11 | postcode: 1100 | country: "
            "Portugal | country_code: pt",
            "   Bounding box: south 38.6, north 38.8, west -9.3, east -9.0",
        ]
        assert "tag00: value 0" in text and "tag11: value 11" in text  # all 12, not the first 8

        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        params = _called_params(mock_urlopen)
        assert params["format"] == ["jsonv2"]
        assert params["q"] == ["Lisbon"]
        assert params["limit"] == ["3"]  # the page and one more
        assert params["countrycodes"] == ["pt"]
        assert params["layer"] == ["address"]
        assert params["extratags"] == ["1"]

    @patch(HTTP_OPEN)
    def test_more_than_ten_places_page_on_up_to_the_forty_nominatim_gives(self, mock_urlopen):
        # It stopped at 10, with nothing past them; Nominatim gives up to 40.
        mock_urlopen.side_effect = [
            respond(geo_answers.nominatim_places(31)),
            respond(geo_answers.nominatim_places(40)),
        ]

        page = _text(osm_search_place("Springfield", max_results=15, offset=15))
        last = _text(osm_search_place("Springfield", max_results=10, offset=30))

        assert page.endswith("[results 16-30 | next: offset=30]")
        assert "31. Springfield" not in page
        assert last.endswith(
            "(the source returns no more than 40 results; add country_codes or layer, or a more "
            "precise query)\n[results 31-40 | end]"
        )
        assert _called_params(mock_urlopen)["limit"] == ["40"]

    def test_the_forty_is_in_the_schema(self):
        properties = osm_search_place.__tool_definition__.schema.input_schema["properties"]
        assert properties["max_results"]["maximum"] == 40
        assert properties["offset"]["maximum"] == 39

    @patch(HTTP_OPEN)
    def test_no_place_is_a_success_that_names_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond([])
        assert _text(osm_search_place("Nowhereville")) == (
            "No OpenStreetMap places match 'Nowhereville'; try fewer words, another spelling, or "
            "no country or layer filter."
        )

    @patch(HTTP_OPEN)
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in _invalid(osm_search_place, "")
        assert "country_codes" in _invalid(osm_search_place, "Lisbon", country_codes="portugal")
        assert "invalid layer" in _invalid(osm_search_place, "Lisbon", layer="bad")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_parameter_nominatim_refuses_is_a_validation_error_in_its_words(self, mock_urlopen):
        body = b'{"error":{"code":400,"message":"Parameter \'lat\' must be a number."}}'
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        message = _invalid(osm_reverse_geocode, 1.0, 2.0)

        assert message == (
            "Nominatim refused the request: Parameter 'lat' must be a number; correct that "
            "argument"
        )

    @patch(HTTP_OPEN)
    def test_api_and_parse_failures(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://nominatim.openstreetmap.org/search",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        with pytest.raises(ToolFailure) as caught:
            osm_search_place("Lisbon")
        assert caught.value.error.type == "rate_limited"
        assert "rate limited" in caught.value.error.message

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = respond("not json")
        with pytest.raises(ToolFailure) as caught:
            osm_search_place("Lisbon")
        assert caught.value.error.type == "upstream"
        assert "could not parse" in caught.value.error.message


class TestOsmReverseGeocode:
    @patch(HTTP_OPEN)
    def test_returns_the_place_with_its_full_address(self, mock_urlopen):
        mock_urlopen.return_value = respond(_PLACE)

        text = osm_reverse_geocode(38.7077507, -9.1365919, layer="address,poi")

        assert text.splitlines()[:2] == [
            "OpenStreetMap place at latitude 38.7077507, longitude -9.1365919, zoom 18 "
            "(Nominatim; data © OpenStreetMap contributors, ODbL):",
            "Lisbon, Portugal",
        ]
        params = _called_params(mock_urlopen)
        assert params["lat"] == ["38.7077507"]
        assert params["lon"] == ["-9.1365919"]
        assert params["zoom"] == ["18"]
        assert params["layer"] == ["address,poi"]

    @patch(HTTP_OPEN)
    def test_zoom_ten_answers_what_reverse_geocode_did_the_city(self, mock_urlopen):
        # reverse_geocode (zoom 10) and osm_reverse_geocode (zoom 18) answered the same point
        # differently; one tool now takes the detail as an argument.
        mock_urlopen.return_value = respond(_PLACE)

        text = osm_reverse_geocode(38.7, -9.1, zoom=10)

        assert _called_params(mock_urlopen)["zoom"] == ["10"]
        assert "Address: city: Lisbon" in text
        assert "reverse_geocode" not in tools.__all__

    def test_the_zoom_nominatim_takes_is_in_the_schema(self):
        zoom = osm_reverse_geocode.__tool_definition__.schema.input_schema["properties"]["zoom"]
        assert (zoom["minimum"], zoom["maximum"]) == (0, 18)

    @patch(HTTP_OPEN)
    def test_invalid_reverse_options_do_not_call_api(self, mock_urlopen):
        assert "must be between -90 and 90" in _invalid(osm_reverse_geocode, -91, 0)
        assert "must be between -180 and 180" in _invalid(osm_reverse_geocode, 0, 181)
        assert "invalid layer" in _invalid(osm_reverse_geocode, 0, 0, layer="bad")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_nothing_at_the_point_is_a_success_that_says_so(self, mock_urlopen):
        mock_urlopen.return_value = respond({"error": "Unable to geocode"})

        assert osm_reverse_geocode(0, -30) == (
            "No OpenStreetMap place at latitude 0, longitude -30 (zoom 18): Nominatim has no "
            "data there, e.g. open sea."
        )

    @patch(HTTP_OPEN)
    def test_another_error_in_a_success_is_the_sources_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond({"error": "Database unavailable"})

        with pytest.raises(ToolFailure) as caught:
            osm_reverse_geocode(0, 0)

        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == "Database unavailable"


@patch(HTTP_OPEN)
def test_both_tools_share_nominatims_one_request_per_second(mock_urlopen, throttle_waits):
    mock_urlopen.side_effect = [respond([]), respond(_PLACE)]

    osm_search_place("Lisbon")
    osm_reverse_geocode(38.7, -9.1)

    assert throttle_waits[0] == 0
    assert 1.0 < throttle_waits[1] <= 1.1
