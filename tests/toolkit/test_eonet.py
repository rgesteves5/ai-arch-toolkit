"""Tests for toolkit/tools/_eonet.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
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
        for call, words in (
            (lambda: eonet_events(status="bad"), "status must"),
            (lambda: eonet_events(start_date="2026"), "invalid start_date"),
            (lambda: eonet_event("bad/id"), "invalid event_id"),
        ):
            with pytest.raises(ToolFailure) as caught:
                call()
            assert caught.value.error.type == "validation_error"
            assert words in str(caught.value)
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_events_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"events": []})

        assert eonet_events() == "No NASA EONET events found."

    @patch(HTTP_OPEN)
    def test_a_rate_limit_is_a_retryable_failure(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        with pytest.raises(ToolFailure) as caught:
            eonet_categories()

        assert caught.value.error.type == "rate_limited"
        assert caught.value.error.retryable


@patch(HTTP_OPEN)
def test_the_500_eonet_sends_for_an_unknown_event_says_what_it_may_mean(mock_urlopen):
    # EONET answered every unknown ID with this page (2026-09-30).
    mock_urlopen.side_effect = http_error(
        500, "Internal Server Error", body=b"<!DOCTYPE html><title>Server Error</title>"
    )

    with pytest.raises(ToolFailure) as caught:
        eonet_event("EONET_0")

    # A 500 may be an outage too, so it stays upstream.
    assert caught.value.error.type == "upstream"
    assert caught.value.error.retryable
    assert str(caught.value) == (
        "HTTP error 500: Internal Server Error (NASA EONET also answers this for an event ID it "
        "does not know: check 'EONET_0' with eonet_events, which lists the current IDs, or try "
        "again later)"
    )


@patch(HTTP_OPEN)
def test_other_http_errors_on_an_event_stay_upstream(mock_urlopen):
    mock_urlopen.side_effect = http_error(503, "Service Unavailable")

    with pytest.raises(ToolFailure) as caught:
        eonet_event("EONET_1")

    assert caught.value.error.type == "upstream"
    assert caught.value.error.retryable


@patch(HTTP_OPEN)
def test_an_answer_without_an_event_is_not_a_blank_event(mock_urlopen):
    mock_urlopen.return_value = respond({})

    with pytest.raises(ToolFailure) as caught:
        eonet_event("EONET_1")

    assert caught.value.error.type == "upstream"
    assert "without an event" in str(caught.value)


@patch(HTTP_OPEN)
def test_a_404_for_an_event_is_not_found(mock_urlopen):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    with pytest.raises(ToolFailure) as caught:
        eonet_event("EONET_0")

    assert caught.value.error.type == "not_found"
    assert caught.value.error.message == (
        "NASA EONET has no event with ID 'EONET_0'; list the current IDs with eonet_events."
    )


@pytest.mark.parametrize("call", [eonet_categories, eonet_events])
@patch(HTTP_OPEN)
def test_a_404_on_a_list_is_endpoint_not_found(mock_urlopen, call):
    # It read as "no matching records found." (upstream), like an empty list.
    mock_urlopen.side_effect = http_error(404, "Not Found")

    with pytest.raises(ToolFailure) as caught:
        call()

    assert caught.value.error.type == "upstream"
    assert "NASA EONET: endpoint not found (HTTP 404)" in caught.value.error.message
