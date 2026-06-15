"""Tests for toolkit/tools/_eonet.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._eonet import eonet_categories, eonet_event, eonet_events

_EVENT = {
    "id": "EONET_1",
    "title": "Example wildfire",
    "description": "A natural event.",
    "categories": [{"id": "wildfires", "title": "Wildfires"}],
    "geometry": [{"date": "2026-06-01T00:00:00Z", "coordinates": [-9.1, 38.7]}],
    "sources": [{"id": "InciWeb", "url": "https://example.com"}],
}


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestEonet:
    @patch("ai_arch_toolkit.toolkit.tools._eonet.urllib.request.urlopen")
    def test_categories_events_and_event(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {
                "categories": [
                    {"id": "wildfires", "title": "Wildfires", "description": "Fire events"}
                ]
            }
        )
        assert "wildfires — Wildfires" in eonet_categories()

        mock_urlopen.return_value = _mock_urlopen({"events": [_EVENT]})
        result = eonet_events(category="wildfires", days=7)
        assert "Example wildfire | id: EONET_1" in result
        assert _params(mock_urlopen)["category"] == ["wildfires"]

        mock_urlopen.return_value = _mock_urlopen(_EVENT)
        detail = eonet_event("EONET_1")
        assert "sources: InciWeb: https://example.com" in detail

    @patch("ai_arch_toolkit.toolkit.tools._eonet.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "status must" in eonet_events(status="bad")
        assert "invalid start_date" in eonet_events(start_date="2026")
        assert "invalid event_id" in eonet_event("bad/id")
        mock_urlopen.assert_not_called()
