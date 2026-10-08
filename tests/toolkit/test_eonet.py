"""Tests for toolkit/tools/_eonet.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._eonet import eonet_categories, eonet_event, eonet_events
from ai_arch_toolkit.toolkit.tools._values import plain
from tests.toolkit import geo_answers
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestEonet:
    @patch(HTTP_OPEN)
    def test_categories_list_the_id_events_take(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "categories": [
                    {"id": "wildfires", "title": "Wildfires", "description": "Fire events"}
                ]
            }
        )
        assert eonet_categories() == (
            "NASA EONET categories (the ID first, as eonet_events takes it):\n"
            "1. wildfires: Wildfires\n"
            "   Fire events"
        )

    @patch(HTTP_OPEN)
    def test_events_show_their_category_status_and_track_labelled(self, mock_urlopen):
        mock_urlopen.return_value = respond(geo_answers.eonet_events(1))

        text = _text(eonet_events(category="severeStorms", days=7))

        assert text == (
            "NASA EONET events for status=open, category=severeStorms, days=7:\n"
            "1. Tropical Storm 1 | id: EONET_1 | Severe Storms (severeStorms) | open\n"
            "   track: 2 points, 2026-09-01T00:00:00Z to 2026-09-02T00:00:00Z; latest at "
            f"latitude {plain(16.0)}, longitude {plain(-41.0)}, magnitude {plain(36.0)} kts "
            "(eonet_event reads the track)"
        )
        params = _params(mock_urlopen)
        assert params["category"] == ["severeStorms"]
        assert params["days"] == ["7"]
        assert params["limit"] == ["11"]  # the page and one more

    @patch(HTTP_OPEN)
    def test_events_past_the_page_are_reachable(self, mock_urlopen):
        # Events came with no position and no way to read past max_results.
        mock_urlopen.return_value = respond(geo_answers.eonet_events(6))

        result = eonet_events(max_results=5)

        assert _text(result).endswith("[results 1-5 | next: offset=5]")
        assert "6. Tropical Storm 6" not in _text(result)
        assert _params(mock_urlopen)["limit"] == ["6"]

    @patch(HTTP_OPEN)
    def test_the_box_goes_in_eonets_corner_order(self, mock_urlopen):
        # The tool takes west,south,east,north and sent it as is; EONET reads min lon, max lat,
        # max lon, min lat (https://eonet.gsfc.nasa.gov/docs/v3).
        mock_urlopen.return_value = respond({"events": []})

        eonet_events(bbox="-125,32,-114,42")

        assert _params(mock_urlopen)["bbox"] == ["-125,42,-114,32"]

    @patch(HTTP_OPEN)
    def test_dates_go_as_yyyy_mm_dd(self, mock_urlopen):
        mock_urlopen.return_value = respond({"events": []})

        eonet_events(start_date="20260901", end_date="2026-09-30")

        params = _params(mock_urlopen)
        assert (params["start"], params["end"]) == (["2026-09-01"], ["2026-09-30"])
        assert "days" not in params

    @patch(HTTP_OPEN)
    def test_several_categories_and_sources_go_as_eonet_takes_them(self, mock_urlopen):
        # EONET documents comma-separated lists for both, read as "any of them"
        # (https://eonet.gsfc.nasa.gov/docs/v3); the tool refused the comma.
        mock_urlopen.return_value = respond({"events": []})

        eonet_events(category="wildfires,volcanoes", source="InciWeb,EO")

        params = _params(mock_urlopen)
        assert (params["category"], params["source"]) == (["wildfires,volcanoes"], ["InciWeb,EO"])

    def test_the_statuses_are_in_the_schema(self):
        properties = eonet_events.__tool_definition__.schema.input_schema["properties"]
        assert properties["status"]["enum"] == ["open", "closed", "all"]

    def test_the_365_days_are_in_the_schema(self):
        days = eonet_events.__tool_definition__.schema.input_schema["properties"]["days"]
        assert (days["minimum"], days["maximum"]) == (1, 365)

    def test_more_days_are_refused_not_cut(self):
        # They were cut to 365 without a word.
        result = ToolGroup(eonet_events).execute(
            ToolCall(id="c", name="eonet_events", input={"days": 1000})
        )
        assert result.error is not None
        assert result.error.type == "validation_error"

    @patch(HTTP_OPEN)
    def test_no_events_is_a_success_that_names_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond({"events": []})

        assert _text(eonet_events(category="volcanoes")) == (
            "No NASA EONET events match status=open, category=volcanoes, days=30; widen the "
            "period (days, start_date) or drop a filter."
        )

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        for call, words in (
            (lambda: eonet_events(status="bad"), "status must"),
            (lambda: eonet_events(start_date="2026"), "invalid start_date"),
            (lambda: eonet_events(bbox="1,2,3"), "bbox must"),
            (lambda: eonet_events(category="wildfires,"), "invalid category"),
            (lambda: eonet_events(source="InciWeb,,EO"), "invalid source"),
            (lambda: eonet_event("bad/id"), "invalid event_id"),
        ):
            with pytest.raises(ToolFailure) as caught:
                call()
            assert caught.value.error.type == "validation_error"
            assert words in str(caught.value)
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_rate_limit_is_a_retryable_failure(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        failure = _failure(eonet_categories)

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable


class TestEonetEvent:
    @patch(HTTP_OPEN)
    def test_the_whole_track_is_read_not_only_its_last_point(self, mock_urlopen):
        # The track was reduced to its last point.
        mock_urlopen.side_effect = lambda *_: respond(geo_answers.eonet_event(points=8))

        first = eonet_event("EONET_1", max_points=5)
        rest = eonet_event("EONET_1", max_points=5, offset=5)

        assert _text(first).splitlines()[:7] == [
            "NASA EONET event EONET_1: Tropical Storm 1",
            "  Categories: Severe Storms (severeStorms)",
            "  Status: open",
            "  Description: A storm over the Atlantic.",
            "  Sources: JTWC: https://www.metoc.navy.mil/jtwc/products/al012026.tcw; NOAA_NHC: "
            "https://www.nhc.noaa.gov/",
            "  Track (8 points; date | position | magnitude):",
            f"2026-09-01T00:00:00Z | latitude {plain(15.0)}, longitude {plain(-40.0)}, "
            f"magnitude {plain(35.0)} kts",
        ]
        assert _text(first).endswith("[results 1-5 of 8 | next: offset=5]")
        assert _text(rest).splitlines()[-2:] == [
            f"2026-09-08T00:00:00Z | latitude {plain(22.0)}, longitude {plain(-47.0)}, "
            f"magnitude {plain(42.0)} kts",
            "[results 6-8 of 8 | end]",
        ]

    @patch(HTTP_OPEN)
    def test_a_polygon_shows_its_span_and_a_closed_event_its_date(self, mock_urlopen):
        event = geo_answers.eonet_event(points=1)
        event["closed"] = "2026-09-05T00:00:00Z"
        event["geometry"] = [
            {
                "date": "2026-09-01T00:00:00Z",
                "type": "Polygon",
                "coordinates": [[[-120.5, 38.0], [-120.0, 38.0], [-120.0, 38.4], [-120.5, 38.0]]],
            }
        ]
        mock_urlopen.return_value = respond(event)

        text = _text(eonet_event("EONET_1"))

        assert "  Status: closed 2026-09-05T00:00:00Z" in text
        assert text.endswith(
            f"2026-09-01T00:00:00Z | area latitude {plain(38.0)} to 38.4, longitude -120.5 to "
            f"{plain(-120.0)}"
        )

    @patch(HTTP_OPEN)
    def test_the_500_eonet_sends_for_an_unknown_event_says_what_it_may_mean(self, mock_urlopen):
        # EONET answered every unknown ID with this page (2026-09-30); the event API's error
        # reader explains it, so no tool reads a caught status.
        mock_urlopen.side_effect = http_error(
            500, "Internal Server Error", body=geo_answers.EONET_ERROR_PAGE
        )

        failure = _failure(eonet_event, "EONET_0")

        # A 500 may be an outage too, so it stays upstream.
        assert failure.error.type == "upstream"
        assert failure.error.retryable
        assert str(failure) == (
            "HTTP error 500: NASA EONET failed, or has no event with that ID (it answers an "
            "unknown ID this way); check the ID with eonet_events, or try again later"
        )

    @patch(HTTP_OPEN)
    def test_other_http_errors_on_an_event_stay_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        failure = _failure(eonet_event, "EONET_1")

        assert failure.error.type == "upstream"
        assert failure.error.retryable

    @patch(HTTP_OPEN)
    def test_an_answer_without_an_event_is_not_a_blank_event(self, mock_urlopen):
        mock_urlopen.return_value = respond({})

        failure = _failure(eonet_event, "EONET_1")

        assert failure.error.type == "upstream"
        assert "without an event" in str(failure)

    @patch(HTTP_OPEN)
    def test_a_404_for_an_event_is_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(eonet_event, "EONET_0")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "NASA EONET has no event with ID 'EONET_0'; list the current IDs with eonet_events."
        )


@pytest.mark.parametrize("call", [eonet_categories, eonet_events])
@patch(HTTP_OPEN)
def test_a_404_on_a_list_is_endpoint_not_found(mock_urlopen, call):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(call)

    assert failure.error.type == "upstream"
    assert "NASA EONET: endpoint not found (HTTP 404)" in failure.error.message
