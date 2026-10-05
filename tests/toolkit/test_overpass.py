"""Tests for toolkit/tools/_overpass.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._overpass import overpass_pois, overpass_query
from tests.toolkit.http_fakes import HTTP_OPEN, respond

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
        mock_urlopen.return_value = respond({**_DATA, "remark": timeout})

        with pytest.raises(ToolFailure) as caught:
            overpass_query("[out:json][timeout:1];node[amenity];out;")
        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == timeout

        mock_urlopen.return_value = respond({"elements": [], "remark": timeout})
        with pytest.raises(ToolFailure) as caught:
            overpass_pois("amenity", latitude=38.7, longitude=-9.1)
        assert caught.value.error.type == "upstream"
        assert caught.value.error.message == timeout

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
