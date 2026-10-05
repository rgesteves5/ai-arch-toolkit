"""Tests for toolkit/tools/_osm.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._osm import osm_reverse_geocode, osm_search_place
from tests.toolkit.http_fakes import HTTP_OPEN, respond

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
    "address": {"city": "Lisbon", "country": "Portugal", "postcode": "1100"},
    "extratags": {"wikidata": "Q597", "website": "https://www.lisboa.pt"},
}


def _invalid(call, *args, **kwargs) -> str:
    """The message of the validation_error ``call`` raises."""
    with pytest.raises(ToolFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.error.type == "validation_error"
    return caught.value.error.message


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestOsmSearchPlace:
    @patch(HTTP_OPEN)
    def test_returns_places(self, mock_urlopen):
        mock_urlopen.return_value = respond([_PLACE])

        result = osm_search_place(
            "Lisbon",
            max_results=2,
            country_codes="pt",
            layer="address",
            include_extra_tags=True,
        )

        assert "OSM places for 'Lisbon'" in result
        assert "Lisbon, Portugal" in result
        assert "OSM: relation/540" in result
        assert "coords: 38.7077507, -9.1365919" in result
        assert "city: Lisbon" in result
        assert "wikidata: Q597" in result
        assert "© OpenStreetMap contributors" in result

        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        params = _called_params(mock_urlopen)
        assert params["format"] == ["jsonv2"]
        assert params["q"] == ["Lisbon"]
        assert params["limit"] == ["2"]
        assert params["countrycodes"] == ["pt"]
        assert params["layer"] == ["address"]
        assert params["extratags"] == ["1"]

    @patch(HTTP_OPEN)
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in _invalid(osm_search_place, "")
        assert "country_codes" in _invalid(osm_search_place, "Lisbon", country_codes="portugal")
        assert "invalid layer" in _invalid(osm_search_place, "Lisbon", layer="bad")
        mock_urlopen.assert_not_called()


class TestOsmReverseGeocode:
    @patch(HTTP_OPEN)
    def test_returns_reverse_result(self, mock_urlopen):
        mock_urlopen.return_value = respond(_PLACE)

        result = osm_reverse_geocode(38.7077507, -9.1365919, zoom=18, layer="address,poi")

        assert "OSM reverse geocode for 38.7077507, -9.1365919" in result
        assert "Lisbon, Portugal" in result

        params = _called_params(mock_urlopen)
        assert params["lat"] == ["38.7077507"]
        assert params["lon"] == ["-9.1365919"]
        assert params["zoom"] == ["18"]
        assert params["layer"] == ["address,poi"]

    @patch(HTTP_OPEN)
    def test_invalid_reverse_options_do_not_call_api(self, mock_urlopen):
        assert "must be between -90 and 90" in _invalid(osm_reverse_geocode, -91, 0)
        assert "must be between -180 and 180" in _invalid(osm_reverse_geocode, 0, 181)
        assert "invalid layer" in _invalid(osm_reverse_geocode, 0, 0, layer="bad")
        mock_urlopen.assert_not_called()

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

    @patch(HTTP_OPEN)
    def test_no_place_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond([])
        assert osm_search_place("Nowhereville") == "No OSM places found for: 'Nowhereville'"

        mock_urlopen.return_value = respond({"error": "Unable to geocode"})
        assert osm_reverse_geocode(0, 0) == "No OSM reverse geocode result for: 0, 0"
