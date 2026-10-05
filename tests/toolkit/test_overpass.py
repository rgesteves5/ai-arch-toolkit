"""Tests for toolkit/tools/_overpass.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._overpass import overpass_pois, overpass_query
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _page(*errors: str) -> bytes:
    """An Overpass error page, as the server sends it, with one paragraph per error."""
    paragraphs = "".join(
        f'<p><strong style="color:#FF0000">Error</strong>: {error} </p>\n' for error in errors
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n<html><head>'
        "<title>OSM3S Response</title></head>\n<body>\n"
        "<p>The data included in this document is from www.openstreetmap.org.</p>\n"
        f"{paragraphs}</body>\n</html>\n"
    ).encode()


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


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


class TestOverpass:
    @patch(HTTP_OPEN)
    def test_query_and_pois(self, mock_urlopen):
        mock_urlopen.return_value = respond(_DATA)

        result = overpass_query('[out:json];node["amenity"="cafe"](38,-10,39,-9);out tags;')

        assert "Cafe A | node/1" in result

        mock_urlopen.return_value = respond(_DATA)
        result = overpass_pois("amenity", "cafe", latitude=38.7, longitude=-9.1, radius_m=500)
        assert "amenity=cafe" in result
        body = mock_urlopen.call_args.args[0].data.decode()
        assert "around%3A500%2C38.7%2C-9.1" in body

    @patch(HTTP_OPEN)
    def test_a_runtime_error_is_the_tools_error_even_with_partial_elements(self, mock_urlopen):
        # Sent with HTTP 200 (overpass-api.de, 2026-09-29), after the elements found so far.
        timeout = 'runtime error: Query timed out in "query" at line 1 after 4 seconds.'
        said = (
            'runtime error: Query timed out in "query" at line 1 after 4 seconds; narrow the '
            "query (a smaller area, fewer elements)"
        )
        mock_urlopen.return_value = respond({**_DATA, "remark": timeout})

        with pytest.raises(ToolFailure) as caught:
            overpass_query("[out:json][timeout:1];node[amenity];out;")
        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == said

        mock_urlopen.return_value = respond({"elements": [], "remark": timeout})
        with pytest.raises(ToolFailure) as caught:
            overpass_pois("amenity", latitude=38.7, longitude=-9.1)
        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == said

    @patch(HTTP_OPEN)
    def test_no_element_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"elements": []})

        assert overpass_query("[out:json];node(1);out;") == "No Overpass elements found."

    @patch(HTTP_OPEN)
    def test_another_remark_is_a_note(self, mock_urlopen):
        mock_urlopen.return_value = respond({**_DATA, "remark": "a note that is no error"})

        assert "Cafe A | node/1" in overpass_query("[out:json];node(1);out;")

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        for call, args, words in (
            (overpass_query, ("node;",), "[out:json]"),
            (overpass_pois, ("amenity", "cafe"), "provide bbox"),
            (overpass_pois, ("bad key",), "invalid tag_key"),
        ):
            with pytest.raises(ToolFailure) as caught:
                call(*args)
            assert caught.value.error.type == "validation_error"
            assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_query_overpass_cannot_parse_is_a_validation_error(self, mock_urlopen):
        page = _page("line 1: parse error: Unknown type &quot;nod&quot;", "line 1: static error")
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=page)

        failure = _failure(overpass_query, "[out:json];nod(1);out;")

        assert failure.error.type == "validation_error"
        assert not failure.error.retryable
        assert failure.error.message == (
            "Overpass could not read the query (HTTP 400): "
            'line 1: parse error: Unknown type "nod"; line 1: static error; '
            "correct the Overpass QL (overpass_query) or the tag and area (overpass_pois)"
        )

    @patch(HTTP_OPEN)
    def test_a_busy_server_keeps_its_words(self, mock_urlopen):
        busy = (
            "runtime error: open64: 0 Success /osm3s_osm_base Dispatcher_Client::"
            "request_read_and_idx::timeout. The server is probably too busy to handle your "
            "request."
        )
        mock_urlopen.side_effect = http_error(504, "Gateway Timeout", body=_page(busy))

        failure = _failure(overpass_query, "[out:json];node(1);out;")

        assert failure.error.type == "upstream"
        assert failure.error.retryable
        assert failure.error.message == f"HTTP error 504: {busy}"

    @patch(HTTP_OPEN)
    def test_a_504_without_a_page_says_what_it_means(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(504, "Gateway Timeout")

        failure = _failure(overpass_query, "[out:json];node(1);out;")

        assert failure.error.type == "upstream"
        assert failure.error.retryable
        assert "Overpass is overloaded or the query timed out" in failure.error.message

    @patch(HTTP_OPEN)
    def test_the_quota_is_a_rate_limit(self, mock_urlopen):
        quota = "runtime error: ... rate_limited. Please check /api/status for the quota."
        mock_urlopen.side_effect = http_error(429, "Too Many Requests", body=_page(quota))

        failure = _failure(overpass_query, "[out:json];node(1);out;")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert "rate_limited. Please check /api/status" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_404_is_an_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(overpass_query, "[out:json];node(1);out;")

        assert failure.error.type == "upstream"
        assert "Overpass: endpoint not found (HTTP 404)" in failure.error.message
