"""Tests for toolkit/tools/_overpass.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._overpass import overpass_pois, overpass_query

_DATA = {
    "elements": [
        {
            "type": "node",
            "id": 1,
            "lat": 38.7,
            "lon": -9.1,
            "tags": {"name": "Cafe A", "amenity": "cafe", "opening_hours": "Mo-Fr"},
        }
    ]
}


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestOverpass:
    @patch("ai_arch_toolkit.toolkit.tools._overpass.urllib.request.urlopen")
    def test_query_and_pois(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_DATA)

        result = overpass_query('[out:json];node["amenity"="cafe"](38,-10,39,-9);out tags;')

        assert "Cafe A | node/1" in result

        mock_urlopen.return_value = _mock_urlopen(_DATA)
        result = overpass_pois("amenity", "cafe", latitude=38.7, longitude=-9.1, radius_m=500)
        assert "amenity=cafe" in result
        body = mock_urlopen.call_args.args[0].data.decode()
        assert "around%3A500%2C38.7%2C-9.1" in body

    @patch("ai_arch_toolkit.toolkit.tools._overpass.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "include [out:json]" in overpass_query("node;")
        assert "provide bbox" in overpass_pois("amenity", "cafe")
        assert "invalid tag_key" in overpass_pois("bad key")
        mock_urlopen.assert_not_called()
