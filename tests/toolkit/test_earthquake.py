"""Tests for toolkit/tools/_earthquake.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._earthquake import (
    earthquake_count,
    earthquake_event,
    earthquake_search,
)

_FEATURE = {
    "type": "Feature",
    "id": "us1",
    "properties": {
        "title": "M 5.0 - Portugal",
        "mag": 5.0,
        "type": "earthquake",
        "time": 1710000000000,
        "place": "Portugal",
        "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us1",
    },
    "geometry": {"type": "Point", "coordinates": [-9.1, 38.7, 10]},
}


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestEarthquake:
    @patch("ai_arch_toolkit.toolkit.tools._earthquake.urllib.request.urlopen")
    def test_search(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"metadata": {"count": 1}, "features": [_FEATURE]}
        )

        result = earthquake_search(start_time="2024-01-01", min_magnitude=4.5)

        assert "M 5.0 - Portugal | id: us1" in result
        assert _params(mock_urlopen)["minmagnitude"] == ["4.5"]

    @patch("ai_arch_toolkit.toolkit.tools._earthquake.urllib.request.urlopen")
    def test_event_and_count(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_FEATURE)
        assert "USGS earthquake us1:" in earthquake_event("us1")

        mock_urlopen.return_value = _mock_urlopen("42")
        assert earthquake_count(start_time="2024-01-01") == "USGS earthquake count: 42"

    @patch("ai_arch_toolkit.toolkit.tools._earthquake.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "invalid start_time" in earthquake_search(start_time="2024")
        assert "offset must" in earthquake_search(offset=0)
        assert "invalid event_id" in earthquake_event("bad/id")
        mock_urlopen.assert_not_called()
