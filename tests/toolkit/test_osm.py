"""Tests for toolkit/tools/_osm.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._osm import osm_reverse_geocode, osm_search_place

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


class TestOsmSearchPlace:
    @patch("ai_arch_toolkit.toolkit.tools._osm._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._osm.urllib.request.urlopen")
    def test_returns_places(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen([_PLACE])

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

    @patch("ai_arch_toolkit.toolkit.tools._osm.urllib.request.urlopen")
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in osm_search_place("")
        assert "country_codes" in osm_search_place("Lisbon", country_codes="portugal")
        assert "invalid layer" in osm_search_place("Lisbon", layer="bad")
        mock_urlopen.assert_not_called()


class TestOsmReverseGeocode:
    @patch("ai_arch_toolkit.toolkit.tools._osm._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._osm.urllib.request.urlopen")
    def test_returns_reverse_result(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen(_PLACE)

        result = osm_reverse_geocode(38.7077507, -9.1365919, zoom=18, layer="address,poi")

        assert "OSM reverse geocode for 38.7077507, -9.1365919" in result
        assert "Lisbon, Portugal" in result

        params = _called_params(mock_urlopen)
        assert params["lat"] == ["38.7077507"]
        assert params["lon"] == ["-9.1365919"]
        assert params["zoom"] == ["18"]
        assert params["layer"] == ["address,poi"]

    @patch("ai_arch_toolkit.toolkit.tools._osm.urllib.request.urlopen")
    def test_invalid_reverse_options_do_not_call_api(self, mock_urlopen):
        assert "latitude must" in osm_reverse_geocode(-91, 0)
        assert "longitude must" in osm_reverse_geocode(0, 181)
        assert "invalid layer" in osm_reverse_geocode(0, 0, layer="bad")
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._osm._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._osm.urllib.request.urlopen")
    def test_api_and_parse_failures(self, mock_urlopen, _mock_throttle):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://nominatim.openstreetmap.org/search",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        assert "rate limited" in osm_search_place("Lisbon")

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _mock_urlopen("not json")
        assert "could not parse" in osm_search_place("Lisbon")
