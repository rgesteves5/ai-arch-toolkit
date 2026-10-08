"""Tests for toolkit/tools/_overpass.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._overpass import overpass_pois, overpass_query
from tests.toolkit import geo_answers
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


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


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

        assert _text(result).splitlines()[1:] == [
            "1. Cafe A | node/1 | latitude 38.7, longitude -9.1",
            "   tags: amenity=cafe; opening_hours=Mo-Fr",
        ]

        mock_urlopen.return_value = respond(_DATA)
        result = overpass_pois("amenity", "cafe", latitude=38.7, longitude=-9.1, radius_m=500)
        assert _text(result).splitlines()[0] == (
            "OpenStreetMap elements tagged amenity=cafe within 500 m of latitude 38.7, longitude "
            "-9.1 (Overpass API; data © OpenStreetMap contributors, ODbL):"
        )
        body = mock_urlopen.call_args.args[0].data.decode()
        assert "around%3A500%2C38.7%2C-9.1" in body

    @patch(HTTP_OPEN)
    def test_every_element_is_reachable_through_the_window(self, mock_urlopen):
        # It showed "returned 25 of 60" and nothing could read the other 35.
        mock_urlopen.side_effect = lambda *_: respond(geo_answers.overpass_elements(60))

        first = overpass_pois("amenity", "cafe", bbox="38.6,-9.3,38.8,-9.0")
        last = overpass_pois("amenity", "cafe", bbox="38.6,-9.3,38.8,-9.0", offset=50)

        assert isinstance(first, ToolResult)
        assert _text(first).endswith("[results 1-25 of 60 | next: offset=25]")
        assert _text(last).splitlines()[-3:] == [
            "60. Café Way | way/77 | latitude 38.71, longitude -9.14",
            "   tags: amenity=cafe; opening_hours=Mo-Fr 08:00-18:00",
            "[results 51-60 of 60 | end]",
        ]

    def test_the_limits_are_in_the_schema(self):
        properties = overpass_pois.__tool_definition__.schema.input_schema["properties"]
        assert (properties["radius_m"]["minimum"], properties["radius_m"]["maximum"]) == (
            1,
            50000,
        )
        assert properties["max_results"]["maximum"] == 50
        assert properties["offset"]["minimum"] == 0

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

        assert _text(overpass_query("[out:json];node(1);out;")) == (
            "No OpenStreetMap elements match the Overpass query '[out:json];node(1);out;'."
        )

    @patch(HTTP_OPEN)
    def test_another_remark_is_a_note(self, mock_urlopen):
        mock_urlopen.return_value = respond({**_DATA, "remark": "a note that is no error"})

        assert "Cafe A | node/1" in _text(overpass_query("[out:json];node(1);out;"))

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

    @pytest.mark.parametrize(
        ("query", "words"),
        [
            # No out statement: Overpass answers no elements, which read as "nothing matches".
            ("[out:json];node(1);", "no out statement"),
            ("[out:json];node[name=Shout](1);", "no out statement"),
            # Another format: the answer could not be parsed, an upstream failure to retry.
            ("[out:xml];node(1);out;", "start it with [out:json]"),
            ("[out:csv(name)];node(1);out;", "start it with [out:json]"),
            ("node(1);out;", "start it with [out:json]"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_query_the_tool_cannot_read_is_refused_before_any_request(
        self, mock_urlopen, query, words
    ):
        mock_urlopen.side_effect = AssertionError("no request")

        failure = _failure(overpass_query, query)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        assert "out;" in failure.error.message  # the example to follow
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        "query",
        [
            '[out:json][timeout:25];node["amenity"="cafe"](38.7,-9.2,38.8,-9.1);out;',
            "[out:json];(node(1);way(2););out center tags;",
            "[ out : json ];node(1)->.a;.a out body;",
            "[out:json];\nnode(1);\nout\n  count;",
            "[out:json];foreach(node(1)){out;}",
        ],
    )
    @patch(HTTP_OPEN)
    def test_every_form_of_the_out_statement_is_taken(self, mock_urlopen, query):
        mock_urlopen.return_value = respond(_DATA)

        assert "Cafe A | node/1" in _text(overpass_query(query))

    @patch(HTTP_OPEN)
    def test_coordinates_go_in_decimal_notation(self, mock_urlopen):
        # str(0.00001) is "1e-05", which Overpass QL cannot parse.
        mock_urlopen.side_effect = lambda *_: respond(_DATA)

        around = overpass_pois("amenity", latitude=0.00001, longitude=-0.00002, radius_m=10)
        box = overpass_pois("amenity", bbox="0.00001,-0.00002,0.00003,0.00004")

        sent = [call.args[0].data.decode() for call in mock_urlopen.call_args_list]
        assert "around%3A10%2C0.00001%2C-0.00002" in sent[0]
        assert "%280.00001%2C-0.00002%2C0.00003%2C0.00004%29" in sent[1]
        assert "e-0" not in _text(around) + _text(box)

    def test_the_coordinates_are_bounded_in_the_schema(self):
        properties = overpass_pois.__tool_definition__.schema.input_schema["properties"]
        assert (properties["latitude"]["minimum"], properties["latitude"]["maximum"]) == (-90, 90)
        assert (properties["longitude"]["minimum"], properties["longitude"]["maximum"]) == (
            -180,
            180,
        )

    @patch(HTTP_OPEN)
    def test_coordinates_out_of_range_are_refused_by_the_executor(self, mock_urlopen):
        mock_urlopen.side_effect = AssertionError("no request")

        result = ToolGroup(overpass_pois).execute(
            ToolCall(
                id="c",
                name="overpass_pois",
                input={"tag_key": "amenity", "latitude": 91.0, "longitude": 0.0},
            )
        )

        assert result.error is not None
        assert result.error.type == "validation_error"
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
