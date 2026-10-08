"""Tests for toolkit/tools/_earthquake.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._earthquake import (
    earthquake_count,
    earthquake_event,
    earthquake_search,
)
from ai_arch_toolkit.toolkit.tools._values import plain
from tests.toolkit import geo_answers
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _sent(mock_urlopen: MagicMock, call: int = -1) -> tuple[str, dict[str, list[str]]]:
    """The path and the query of a request the tool sent."""
    url = urlparse(mock_urlopen.call_args_list[call].args[0].full_url)
    return url.path, parse_qs(url.query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestEarthquakeSearch:
    @patch(HTTP_OPEN)
    def test_the_total_is_the_count_methods_not_the_pages(self, mock_urlopen):
        # The header's "count" was metadata.count: the events of the page.
        mock_urlopen.side_effect = [respond("1234\n"), respond(geo_answers.usgs_features(1, 2))]

        result = earthquake_search(start_time="2024-01-01", min_magnitude=4.5, max_results=2)

        text = _text(result)
        assert text.endswith("[results 1-2 of 1234 | next: offset=3]")
        assert isinstance(result, ToolResult)
        assert result.metadata["window"]["total"] == 1234
        count_path, count_query = _sent(mock_urlopen, 0)
        page_path, page_query = _sent(mock_urlopen, 1)
        assert count_path.endswith("/count") and page_path.endswith("/query")
        assert count_query == {
            "minmagnitude": ["4.5"],
            "maxmagnitude": [plain(10.0)],
            "starttime": ["2024-01-01"],
        }
        assert page_query["limit"] == ["2"]
        assert page_query["offset"] == ["1"]
        assert page_query["format"] == ["geojson"]

    @patch(HTTP_OPEN)
    def test_times_are_iso_8601_utc_and_positions_are_labelled(self, mock_urlopen):
        # Times came as epoch milliseconds (1710000000000).
        mock_urlopen.side_effect = [respond("1"), respond(geo_answers.usgs_features(1))]

        text = _text(earthquake_search())

        assert text.splitlines() == [
            f"USGS earthquakes for minmagnitude={plain(0.0)}, maxmagnitude={plain(10.0)}:",
            "1. M 5.1 - Portugal | id: us1",
            "   time: 2024-03-09T16:00:00Z | magnitude 5.1 mww | type: earthquake",
            f"   latitude 38.7, longitude -9.1, depth {plain(10.0)} km",
        ]

    @patch(HTTP_OPEN)
    def test_the_next_page_numbers_on_from_the_offset(self, mock_urlopen):
        mock_urlopen.side_effect = [respond("3"), respond(geo_answers.usgs_features(3))]

        text = _text(earthquake_search(max_results=2, offset=3))

        assert "3. M 5.3 - Portugal | id: us3" in text
        assert text.endswith("[results 3-3 of 3 | end]")
        assert _sent(mock_urlopen)[1]["offset"] == ["3"]

    @patch(HTTP_OPEN)
    def test_no_events_is_a_success_that_names_the_query_without_asking_for_a_page(
        self, mock_urlopen
    ):
        mock_urlopen.return_value = respond("0")

        assert _text(earthquake_search(start_time="2024-01-01")) == (
            f"No USGS earthquakes match minmagnitude={plain(0.0)}, maxmagnitude={plain(10.0)}, "
            "starttime=2024-01-01; widen the dates, the magnitudes or the area."
        )
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_a_page_with_no_content_is_an_empty_page_not_a_parse_error(self, mock_urlopen):
        # USGS answers "no data" with 204 No Content by default (nodata=204,
        # https://earthquake.usgs.gov/fdsnws/event/1/), which read as "could not parse".
        mock_urlopen.side_effect = [respond("2"), respond(b"", status=204)]

        text = _text(earthquake_search(offset=5))

        assert text.endswith("[no results from 5 of 2 | end]")

    @patch(HTTP_OPEN)
    def test_a_query_usgs_rejects_is_a_validation_error_with_its_detail(self, mock_urlopen):
        body = geo_answers.fdsn_error(400, "Bad Request", geo_answers.USGS_BAD_START)
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(earthquake_search, start_time="2024-01-01")

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            'USGS rejected the query (HTTP 400): Bad starttime value "2024-13-01"; correct the '
            "dates (YYYY-MM-DD), the magnitudes, the depths or the area"
        )

    @patch(HTTP_OPEN)
    def test_too_much_data_says_to_narrow(self, mock_urlopen):
        detail = "Query exceeds the 20000 event limit."
        body = geo_answers.fdsn_error(413, "Request Entity Too Large", detail)
        mock_urlopen.side_effect = http_error(413, "Request Entity Too Large", body=body)

        failure = _failure(earthquake_search)

        assert failure.error.type == "validation_error"
        assert "Query exceeds the 20000 event limit; narrow the dates" in failure.error.message

    @patch(HTTP_OPEN)
    def test_the_end_day_is_included(self, mock_urlopen):
        # USGS reads a bare date as the start of that day, 00:00:00
        # (https://earthquake.usgs.gov/fdsnws/event/1/), so the end day's events were left out:
        # a one-day search counted 0 and said to widen the dates.
        mock_urlopen.side_effect = [respond("2"), respond(geo_answers.usgs_features(1, 2))]

        text = _text(earthquake_search(start_time="2024-01-01", end_time="2024-01-01"))

        for call in (0, 1):
            query = _sent(mock_urlopen, call)[1]
            assert query["starttime"] == ["2024-01-01"]
            assert query["endtime"] == ["2024-01-01T23:59:59.999999"]
        assert "endtime=2024-01-01T23:59:59.999999" in text.splitlines()[0]

    @patch(HTTP_OPEN)
    def test_numbers_go_in_decimal_notation(self, mock_urlopen):
        # FDSN services refuse scientific notation (Commonalities 1.2, "Float type parameters"),
        # and str(0.00001) is "1e-05".
        mock_urlopen.return_value = respond("0")

        text = _text(
            earthquake_search(
                min_magnitude=0.00001,
                latitude=0.00001,
                longitude=-0.00002,
                max_radius_km=0.00003,
                min_depth_km=0.00004,
                max_depth_km=0.00005,
            )
        )

        query = _sent(mock_urlopen)[1]
        assert {name: query[name] for name in ("minmagnitude", "latitude", "longitude")} == {
            "minmagnitude": ["0.00001"],
            "latitude": ["0.00001"],
            "longitude": ["-0.00002"],
        }
        assert (query["maxradiuskm"], query["mindepth"], query["maxdepth"]) == (
            ["0.00003"],
            ["0.00004"],
            ["0.00005"],
        )
        assert "e-0" not in text

    def test_the_limits_are_in_the_schema(self):
        properties = earthquake_search.__tool_definition__.schema.input_schema["properties"]
        bounds = {
            name: (properties[name].get("minimum"), properties[name].get("maximum"))
            for name in (
                "max_results",
                "offset",
                "latitude",
                "longitude",
                "max_radius_km",
                "min_depth_km",
                "max_depth_km",
            )
        }
        # USGS's documented ranges (https://earthquake.usgs.gov/fdsnws/event/1/).
        assert bounds == {
            "max_results": (1, 50),
            "offset": (1, None),
            "latitude": (-90, 90),
            "longitude": (-180, 180),
            "max_radius_km": (0, 20001.6),
            "min_depth_km": (-100, 1000),
            "max_depth_km": (-100, 1000),
        }
        assert properties["order_by"]["enum"] == ["time", "time-asc", "magnitude", "magnitude-asc"]

    @pytest.mark.parametrize(
        "args",
        [
            {"latitude": 91.0, "longitude": 0.0, "max_radius_km": 10.0},
            {"latitude": 0.0, "longitude": 181.0, "max_radius_km": 10.0},
            {"latitude": 0.0, "longitude": 0.0, "max_radius_km": -1.0},
            {"max_depth_km": 5000.0},
            {"order_by": "size"},
        ],
    )
    @patch(HTTP_OPEN)
    def test_values_usgs_does_not_take_are_refused_before_any_request(self, mock_urlopen, args):
        mock_urlopen.side_effect = AssertionError("no request")

        result = ToolGroup(earthquake_search).execute(
            ToolCall(id="c", name="earthquake_search", input=args)
        )

        assert result.error is not None
        assert result.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: earthquake_search(start_time="2024"), "invalid start_time '2024'"),
            (lambda: earthquake_search(order_by="size"), "invalid order_by 'size'"),
            (lambda: earthquake_search(latitude=10.0), "must be provided together"),
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
        failure = _failure(call)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()


class TestEarthquakeEvent:
    @patch(HTTP_OPEN)
    def test_an_event_with_its_details_and_times_in_utc(self, mock_urlopen):
        mock_urlopen.return_value = respond(geo_answers.usgs_feature(1))

        assert earthquake_event("us1").splitlines() == [
            "USGS earthquake us1:",
            "M 5.1 - Portugal | id: us1",
            "   time: 2024-03-09T16:00:00Z | magnitude 5.1 mww | type: earthquake",
            f"   latitude 38.7, longitude -9.1, depth {plain(10.0)} km",
            "   review status: reviewed | felt reports: 12 | PAGER alert: green | tsunami flag: 0 "
            "| significance: 400",
            "   updated: 2024-03-09T16:01:00Z",
            "   USGS event page: https://earthquake.usgs.gov/earthquakes/eventpage/us1",
        ]

    @patch(HTTP_OPEN)
    def test_an_unknown_event_id_is_not_found(self, mock_urlopen):
        # USGS answers an unknown eventid with a 404 and a text page.
        mock_urlopen.side_effect = http_error(
            404, "Not Found", body=b"Error 404: Not Found\n\nUnknown eventid=us0\n"
        )

        failure = _failure(earthquake_event, "us0")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "USGS has no earthquake with ID us0; search with earthquake_search."
        )

    @patch(HTTP_OPEN)
    def test_no_content_for_an_event_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond(b"", status=204)

        failure = _failure(earthquake_event, "us0")

        assert failure.error.type == "not_found"
        assert "earthquake_search" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_deleted_event_is_not_found_and_says_so(self, mock_urlopen):
        # Deleted events answer 409 Conflict (https://earthquake.usgs.gov/fdsnws/event/1/).
        body = geo_answers.fdsn_error(409, "Conflict", "Event us0 was deleted.")
        mock_urlopen.side_effect = http_error(409, "Conflict", body=body)

        failure = _failure(earthquake_event, "us0")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "USGS deleted this event (HTTP 409): Event us0 was deleted; find the event that "
            "replaced it with earthquake_search"
        )


class TestEarthquakeCount:
    @patch(HTTP_OPEN)
    def test_count_names_what_it_counted(self, mock_urlopen):
        mock_urlopen.return_value = respond("42")

        assert earthquake_count(start_time="2024-01-01") == (
            f"USGS earthquake count for minmagnitude={plain(0.0)}, maxmagnitude={plain(10.0)}, "
            "starttime=2024-01-01: 42"
        )

    @patch(HTTP_OPEN)
    def test_the_end_day_is_counted(self, mock_urlopen):
        mock_urlopen.return_value = respond("3")

        earthquake_count(start_time="2024-01-01", end_time="2024-01-31")

        assert _sent(mock_urlopen)[1]["endtime"] == ["2024-01-31T23:59:59.999999"]

    @patch(HTTP_OPEN)
    def test_upstream_failure_propagates(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        failure = _failure(earthquake_count)

        assert failure.error.type == "upstream"
        assert failure.error.retryable


@pytest.mark.parametrize("call", [earthquake_search, earthquake_count])
@patch(HTTP_OPEN)
def test_404_on_a_query_is_endpoint_not_found(mock_urlopen, call):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(call)

    assert failure.error.type == "upstream"
    assert "USGS: endpoint not found (HTTP 404)" in failure.error.message
