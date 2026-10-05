"""Tests for toolkit/tools/_earthquake.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._earthquake import (
    earthquake_count,
    earthquake_event,
    earthquake_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

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


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestEarthquake:
    @patch(HTTP_OPEN)
    def test_search(self, mock_urlopen):
        mock_urlopen.return_value = respond({"metadata": {"count": 1}, "features": [_FEATURE]})

        result = earthquake_search(start_time="2024-01-01", min_magnitude=4.5)

        assert "M 5.0 - Portugal | id: us1" in result
        assert _params(mock_urlopen)["minmagnitude"] == ["4.5"]

    @patch(HTTP_OPEN)
    def test_event_and_count(self, mock_urlopen):
        mock_urlopen.return_value = respond(_FEATURE)
        assert "USGS earthquake us1:" in earthquake_event("us1")

        mock_urlopen.return_value = respond("42")
        assert earthquake_count(start_time="2024-01-01") == "USGS earthquake count: 42"

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: earthquake_search(start_time="2024"), "invalid start_time '2024'"),
            (lambda: earthquake_search(offset=0), "offset must be greater than or equal to 1"),
            (lambda: earthquake_search(order_by="size"), "invalid order_by 'size'"),
            (lambda: earthquake_search(latitude=10.0), "must be provided together"),
            (
                lambda: earthquake_search(latitude=91.0, longitude=0.0, max_radius_km=10.0),
                "latitude must be between -90 and 90",
            ),
            (lambda: earthquake_event("bad/id"), "invalid event_id 'bad/id'"),
            (
                lambda: earthquake_count(min_magnitude=6.0, max_magnitude=5.0),
                "must be less than or equal to max_magnitude",
            ),
            (
                lambda: earthquake_count(start_time="2024-02-01", end_time="2024-01-01"),
                "must be before or equal to end_time",
            ),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen, call, words):
        with pytest.raises(ToolFailure) as caught:
            call()

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_events_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"metadata": {"count": 0}, "features": []})

        assert earthquake_search() == "No USGS earthquakes found."

    @patch(HTTP_OPEN)
    def test_an_empty_event_answer_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({})

        with pytest.raises(ToolFailure) as caught:
            earthquake_event("us0")

        assert caught.value.error.type == "not_found"
        assert "USGS has no earthquake with ID us0" in caught.value.error.message
        assert "earthquake_search" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_upstream_failure_propagates(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        with pytest.raises(ToolFailure) as caught:
            earthquake_count()

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable

    @patch(HTTP_OPEN)
    def test_an_unknown_event_id_is_not_found(self, mock_urlopen):
        # USGS answers an unknown eventid with a 404 and a text page.
        mock_urlopen.side_effect = http_error(
            404, "Not Found", body=b"Error 404: Not Found\n\nUnknown eventid=us0\n"
        )

        with pytest.raises(ToolFailure) as caught:
            earthquake_event("us0")

        assert caught.value.error.type == "not_found"
        assert caught.value.error.message == (
            "USGS has no earthquake with ID us0; search with earthquake_search."
        )

    @pytest.mark.parametrize("call", [earthquake_search, earthquake_count])
    @patch(HTTP_OPEN)
    def test_404_on_a_query_is_endpoint_not_found(self, mock_urlopen, call):
        # It read as "no matching records found." (upstream), like an empty search.
        mock_urlopen.side_effect = http_error(404, "Not Found")

        with pytest.raises(ToolFailure) as caught:
            call()

        assert caught.value.error.type == "upstream"
        assert "USGS: endpoint not found (HTTP 404)" in caught.value.error.message
