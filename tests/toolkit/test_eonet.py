"""Tests for toolkit/tools/_eonet.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._eonet import eonet_categories, eonet_event, eonet_events
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_EVENT = {
    "id": "EONET_1",
    "title": "Example wildfire",
    "description": "A natural event.",
    "categories": [{"id": "wildfires", "title": "Wildfires"}],
    "geometry": [{"date": "2026-06-01T00:00:00Z", "coordinates": [-9.1, 38.7]}],
    "sources": [{"id": "InciWeb", "url": "https://example.com"}],
}


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestEonet:
    @patch(HTTP_OPEN)
    def test_categories_events_and_event(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "categories": [
                    {"id": "wildfires", "title": "Wildfires", "description": "Fire events"}
                ]
            }
        )
        assert "wildfires — Wildfires" in eonet_categories()

        mock_urlopen.return_value = respond({"events": [_EVENT]})
        result = eonet_events(category="wildfires", days=7)
        assert "Example wildfire | id: EONET_1" in result
        assert _params(mock_urlopen)["category"] == ["wildfires"]

        mock_urlopen.return_value = respond(_EVENT)
        detail = eonet_event("EONET_1")
        assert "sources: InciWeb: https://example.com" in detail

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "status must" in eonet_events(status="bad")
        assert "invalid start_date" in eonet_events(start_date="2026")
        assert "invalid event_id" in eonet_event("bad/id")
        mock_urlopen.assert_not_called()


@patch(HTTP_OPEN)
def test_the_500_eonet_sends_for_an_unknown_event_says_what_it_may_mean(mock_urlopen):
    # EONET answered every unknown ID with this page (2026-09-30).
    mock_urlopen.side_effect = http_error(
        500, "Internal Server Error", body=b"<!DOCTYPE html><title>Server Error</title>"
    )

    assert eonet_event("EONET_0") == (
        "NASA EONET event failed: HTTP error 500: Internal Server Error (EONET answers this for "
        "an event ID it does not know; eonet_events lists the current IDs)"
    )


@patch(HTTP_OPEN)
def test_an_answer_without_an_event_is_not_a_blank_event(mock_urlopen):
    mock_urlopen.return_value = respond({})

    assert eonet_event("EONET_1") == (
        "NASA EONET event failed: could not parse API response: no event in the answer"
    )
