"""Tests for toolkit/tools/_eurostat.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolFailure, ToolGroup, ToolResult
from ai_arch_toolkit.toolkit.tools._eurostat import (
    eurostat_dataset,
    eurostat_dataset_search,
    eurostat_series,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

# A dataflow stub, as ``detail=allstubs`` returns it: the ID and the title.
_DATAFLOW = {
    "link": {
        "item": [
            {
                "class": "dataset",
                "label": "Population on 1 January",
                "extension": {"lang": "EN", "id": "TPS00001", "agencyId": "ESTAT"},
            },
            {
                "class": "dataset",
                "label": "Mean hourly earnings",
                "extension": {"lang": "EN", "id": "EARN_SES_AGT15", "agencyId": "ESTAT"},
            },
        ]
    }
}


def _catalogue(count: int) -> dict[str, Any]:
    return {
        "link": {
            "item": [
                {"label": f"Population table {number}", "extension": {"id": f"POP{number:03d}"}}
                for number in range(1, count + 1)
            ]
        }
    }


# A JSON-stat 2.0 answer (https://json-stat.org/format/): four dimensions, the last fastest.
_DATASET = {
    "version": "2.0",
    "class": "dataset",
    "label": "Population on 1 January",
    "source": "ESTAT",
    "updated": "2026-04-30T23:00:00+0200",
    "id": ["freq", "unit", "geo", "time"],
    "size": [1, 2, 2, 2],
    "value": {
        "0": 10467366,
        "1": 10639726,
        "2": 48085361,
        "3": 48619695,
        "4": 10467.366,
        "5": 10639.726,
        "6": 48085.361,
        "7": 1.5e7,
    },
    "status": {"1": "p"},
    "dimension": {
        "freq": {
            "label": "Time frequency",
            "category": {"index": {"A": 0}, "label": {"A": "Annual"}},
        },
        "unit": {
            "label": "Unit of measure",
            "category": {
                "index": {"NR": 0, "THS": 1},
                "label": {"NR": "Number", "THS": "Thousand"},
            },
        },
        "geo": {
            "label": "Geopolitical entity",
            "category": {"index": {"PT": 0, "ES": 1}, "label": {"PT": "Portugal", "ES": "Spain"}},
        },
        "time": {
            "label": "Time",
            "category": {
                "index": {"2023": 0, "2024": 1},
                "label": {"2023": "2023", "2024": "2024"},
            },
        },
    },
    "extension": {
        "description": "<p>Population on 1 January of each year.</p>" + " More." * 200,
        "annotation": [
            {"type": "OBS_COUNT", "title": "592"},
            {"type": "OBS_PERIOD_OVERALL_OLDEST", "title": "2014"},
            {"type": "OBS_PERIOD_OVERALL_LATEST", "title": "2024"},
        ],
    },
}


def _one_geo() -> dict[str, Any]:
    """``_DATASET`` with one unit and one geo: only time varies."""
    dimension = dict(_DATASET["dimension"])
    dimension["unit"] = {
        "label": "Unit of measure",
        "category": {"index": {"NR": 0}, "label": {"NR": "Number"}},
    }
    dimension["geo"] = {
        "label": "Geopolitical entity",
        "category": {"index": {"PT": 0}, "label": {"PT": "Portugal"}},
    }
    return {
        **_DATASET,
        "size": [1, 1, 1, 2],
        "value": {"0": 10467366, "1": 10639726},
        "dimension": dimension,
    }


def _latest() -> dict[str, Any]:
    """``_DATASET`` as ``lastTimePeriod=1`` answers it: the latest period only."""
    dimension = dict(_DATASET["dimension"])
    dimension["time"] = {"label": "Time", "category": {"index": {"2024": 0}}}
    return {
        **_DATASET,
        "size": [1, 2, 2, 1],
        "value": {"0": 10639726, "1": 48619695, "2": 10639.726, "3": 1.5e7},
        "dimension": dimension,
    }


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok
    assert isinstance(result.value, str)
    return result.value


class TestDatasetSearch:
    @patch(HTTP_OPEN)
    def test_finds_datasets_by_id_or_title(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(_DATAFLOW), respond(_DATAFLOW)]

        text = _text(eurostat_dataset_search("population"))

        assert text == (
            "Eurostat datasets that match 'population' (eurostat_dataset reads one):\n"
            "1. TPS00001: Population on 1 January"
        )
        assert "1. EARN_SES_AGT15: Mean hourly earnings" in _text(eurostat_dataset_search("ses"))
        # The full catalogue is 20 MB, over the response cap; its stubs are about 1.5 MB.
        assert _params(mock_urlopen)["detail"] == ["allstubs"]

    @patch(HTTP_OPEN)
    def test_matches_read_on_with_the_total(self, mock_urlopen):
        mock_urlopen.return_value = respond(_catalogue(30))

        result = eurostat_dataset_search("population", max_results=10, offset=10)
        text = _text(result)

        assert text.splitlines()[1] == "11. POP011: Population table 11"
        assert text.endswith("[results 11-20 of 30 | next: offset=20]")

    @patch(HTTP_OPEN)
    def test_no_datasets_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(_DATAFLOW)

        assert _text(eurostat_dataset_search("zzzz")) == "No Eurostat datasets match 'zzzz'."


class TestDataset:
    @patch(HTTP_OPEN)
    def test_reads_the_details_and_every_dimensions_codes_with_labels(self, mock_urlopen):
        # eurostat_dimensions made the same request: the dataset's codes are here now.
        mock_urlopen.return_value = respond(_latest())

        lines = _text(eurostat_dataset("tps00001")).splitlines()

        assert lines[:3] == [
            "Eurostat dataset TPS00001: Population on 1 January",
            "   updated: 2026-04-30T21:00:00Z | source: ESTAT",
            "   observations: 592 | period: 2014 to 2024",
        ]
        description = lines[3]
        assert description.startswith("   description: Population on 1 January of each year.")
        assert description.endswith("More. More.")  # whole, not cut at 500 characters
        assert lines[4:] == [
            "Dimensions and codes (eurostat_series takes them as filters, e.g. geo=PT+ES):",
            "freq: Time frequency (1 code)",
            "  A: Annual",
            "unit: Unit of measure (2 codes)",
            "  NR: Number",
            "  THS: Thousand",
            "geo: Geopolitical entity (2 codes)",
            "  PT: Portugal",
            "  ES: Spain",
            "time: Time (the latest period here; the data cover 2014 to 2024)",
            "  2024",
        ]
        assert _params(mock_urlopen)["lastTimePeriod"] == ["1"]

    @patch(HTTP_OPEN)
    def test_one_dimensions_codes_read_on(self, mock_urlopen):
        many = {f"G{number:02d}": f"Region {number}" for number in range(70)}
        answer = dict(_DATASET)
        answer["dimension"] = {
            **_DATASET["dimension"],
            "geo": {
                "label": "Geopolitical entity",
                "category": {
                    "index": {code: position for position, code in enumerate(many)},
                    "label": many,
                },
            },
        }
        mock_urlopen.side_effect = [respond(answer), respond(answer)]

        first = eurostat_dataset("TPS00001", dimension="geo")
        second = _text(eurostat_dataset("TPS00001", dimension="geo", offset=60))

        assert _text(first).splitlines()[1] == "geo: Geopolitical entity (70 codes)"
        assert _text(first).endswith('[results 1-60 of 71 | next: offset=60, dimension="geo"]')
        assert first.metadata["window"]["next_call"] == {"offset": 60, "dimension": "geo"}
        # The codes past the first window name their dimension, whose line is in the first.
        assert second.splitlines()[:2] == [
            "Eurostat dataset TPS00001: Population on 1 January, codes (geo: Geopolitical "
            "entity, continued):",
            "  G59: Region 59",
        ]
        assert second.endswith("[results 61-71 of 71 | end]")

    @patch(HTTP_OPEN)
    def test_a_window_that_starts_on_a_dimensions_line_needs_no_name(self, mock_urlopen):
        mock_urlopen.return_value = respond(_latest())

        text = _text(eurostat_dataset("TPS00001", offset=2))

        assert text.splitlines()[:2] == [
            "Eurostat dataset TPS00001: Population on 1 January, codes:",
            "unit: Unit of measure (2 codes)",
        ]

    @patch(HTTP_OPEN)
    def test_a_dimension_the_dataset_does_not_have_names_those_it_has(self, mock_urlopen):
        mock_urlopen.return_value = respond(_DATASET)

        failure = _failure(lambda: eurostat_dataset("TPS00001", dimension="sex"))

        assert failure.error.type == "validation_error"
        assert str(failure) == ("TPS00001 has no dimension 'sex'; it has freq, unit, geo and time")


class TestSeries:
    @patch(HTTP_OPEN)
    def test_codes_come_with_labels_and_open_dimensions_are_named(self, mock_urlopen):
        # eurostat_compare kept the first points in index order: an arbitrary series when a
        # dimension other than geo stayed open. Every series is a row now, named by its codes.
        mock_urlopen.return_value = respond(_DATASET)

        text = _text(eurostat_series("TPS00001", filters="geo=PT+ES", last_time_periods=2))

        assert text.splitlines() == [
            "Eurostat TPS00001: Population on 1 January",
            "fixed: Time frequency (freq) = Annual (A)",
            "unit varies too: each row names its series; filter it (e.g. unit=NR) to compare "
            "one series",
            "rows: unit | geo | time: value",
            "1. Number (NR) | Portugal (PT) | 2023: 10467366",
            "2. Number (NR) | Portugal (PT) | 2024: 10639726 (flag p)",
            "3. Number (NR) | Spain (ES) | 2023: 48085361",
            "4. Number (NR) | Spain (ES) | 2024: 48619695",
            "5. Thousand (THS) | Portugal (PT) | 2023: 10467.366",
            "6. Thousand (THS) | Portugal (PT) | 2024: 10639.726",
            "7. Thousand (THS) | Spain (ES) | 2023: 48085.361",
            "8. Thousand (THS) | Spain (ES) | 2024: 15000000.0",  # a float keeps its point
        ]

    @patch(HTTP_OPEN)
    def test_flags_come_with_the_labels_the_answer_brings(self, mock_urlopen):
        # Eurostat's JSON-stat names its flags in extension.status.label.
        extension = {
            **_DATASET["extension"],
            "status": {"label": {"p": "provisional", "e": "estimated"}},
        }
        status = {"0": "e", "1": "p", "2": "ep", "3": "x"}
        mock_urlopen.return_value = respond(
            {**_one_geo(), "status": status, "extension": extension}
        )

        text = _text(eurostat_series("TPS00001", filters="geo=PT,unit=NR"))

        assert text.splitlines()[-2:] == [
            "1. 2023: 10467366 (flag e: estimated)",
            "2. 2024: 10639726 (flag p: provisional)",
        ]

    @patch(HTTP_OPEN)
    def test_combined_and_unknown_flags(self, mock_urlopen):
        extension = {"status": {"label": {"p": "provisional", "e": "estimated"}}}
        status = {"0": "ep", "1": "x"}
        mock_urlopen.return_value = respond(
            {**_one_geo(), "status": status, "extension": extension}
        )

        text = _text(eurostat_series("TPS00001", filters="geo=PT,unit=NR"))

        assert text.splitlines()[-2:] == [
            "1. 2023: 10467366 (flag ep: estimated, provisional)",
            "2. 2024: 10639726 (flag x)",
        ]

    @patch(HTTP_OPEN)
    def test_geo_codes_are_upper_case(self, mock_urlopen):
        # eurostat_compare upper-cased them; Eurostat's geo codes are (PT, EU27_2020).
        mock_urlopen.return_value = respond(_DATASET)

        eurostat_series("TPS00001", filters="geo=pt+es")

        assert _params(mock_urlopen)["geo"] == ["PT", "ES"]

    @patch(HTTP_OPEN)
    def test_several_codes_go_as_repeated_parameters(self, mock_urlopen):
        # "if several VALUE are required for a dimension several filter must be used"
        # (https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/
        # api-detailed-guidelines/api-statistics).
        mock_urlopen.return_value = respond(_DATASET)

        eurostat_series("TPS00001", filters="geo=PT+ES+FR, unit=NR")

        params = _params(mock_urlopen)
        assert params["geo"] == ["PT", "ES", "FR"]
        assert params["unit"] == ["NR"]
        assert params["lastTimePeriod"] == ["5"]

    @pytest.mark.parametrize("time", ["time=2020", "sinceTimePeriod=2015", "TIME_PERIOD=2019"])
    @patch(HTTP_OPEN)
    def test_a_time_filter_replaces_the_last_periods(self, mock_urlopen, time):
        # Only one time parameter per query (the same guide).
        mock_urlopen.return_value = respond(_DATASET)

        eurostat_series("TPS00001", filters=f"geo=PT,{time}")

        assert "lastTimePeriod" not in _params(mock_urlopen)

    @patch(HTTP_OPEN)
    def test_one_series_has_only_time_in_its_rows(self, mock_urlopen):
        mock_urlopen.return_value = respond(_one_geo())

        text = _text(eurostat_series("TPS00001", filters="geo=PT,unit=NR"))

        assert text.splitlines()[1:] == [
            "fixed: Time frequency (freq) = Annual (A); Unit of measure (unit) = Number (NR); "
            "Geopolitical entity (geo) = Portugal (PT)",
            "rows: time: value",
            "1. 2023: 10467366",
            "2. 2024: 10639726 (flag p)",
        ]

    @patch(HTTP_OPEN)
    def test_observations_read_on(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(_DATASET), respond(_DATASET)]

        first = eurostat_series("TPS00001", max_points=3)
        second = _text(eurostat_series("TPS00001", max_points=3, offset=3))

        assert _text(first).endswith("[results 1-3 of 8 | next: offset=3]")
        assert second.splitlines()[4] == "4. Number (NR) | Spain (ES) | 2024: 48619695"
        assert second.endswith("[results 4-6 of 8 | next: offset=6]")

    @patch(HTTP_OPEN)
    def test_an_answer_without_observations_says_so(self, mock_urlopen):
        mock_urlopen.return_value = respond({**_DATASET, "value": {}})

        assert _text(eurostat_series("TPS00001", filters="geo=PT")) == (
            "Eurostat has no observations of TPS00001 for geo=PT in the last 5 periods."
        )

    def test_the_limits_are_in_the_schema(self):
        group = ToolGroup(eurostat_series, eurostat_dataset, eurostat_dataset_search)
        for name, args in (
            ("eurostat_series", {"dataset_id": "X1", "last_time_periods": 101}),
            ("eurostat_series", {"dataset_id": "X1", "last_time_periods": 0}),
            ("eurostat_series", {"dataset_id": "X1", "max_points": 101}),
            ("eurostat_dataset", {"dataset_id": "X1", "offset": -1}),
            ("eurostat_dataset_search", {"query": "x", "max_results": 51}),
        ):
            result = group.execute(ToolCall(id="c", name=name, input=args))
            assert not result.ok and result.error is not None
            assert result.error.type == "validation_error", args


@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda: eurostat_dataset("bad/id"), "invalid dataset_id"),
        (lambda: eurostat_dataset_search(""), "invalid query"),
        (lambda: eurostat_series("TPS00001", filters="geo"), "use key=value"),
        (lambda: eurostat_series("TPS00001", filters="geo=P T"), "invalid filter value"),
        (lambda: eurostat_series("TPS00001", filters="format=TSV"), "the tool sets format"),
        (lambda: eurostat_series("TPS00001", filters="geo=PT,Lang=fr"), "the tool sets format"),
    ],
)
@patch(HTTP_OPEN)
def test_invalid_arguments_do_not_call_api(mock_urlopen, call, words):
    failure = _failure(call)

    assert failure.error.type == "validation_error"
    assert words in str(failure)
    mock_urlopen.assert_not_called()


# As both APIs answered live (2026-09-30).
_NOT_DISSEMINATED = (
    b'{ "error": [{"status": 404,"id": 100,"label": "ERR_NOT_FOUND_4: NOT_A_DATASET '
    b'(DATA_FLOW:ALL,1.0) is not available for dissemination."}]}'
)
_ASYNCHRONOUS = (
    b'{ "error": [{"status": 413,"id": 413,"label": "ASYNCHRONOUS_RESPONSE. Your request will '
    b'be treated asynchronously. Please try again later."}]}'
)


@pytest.mark.parametrize(
    "call",
    [
        lambda: eurostat_dataset("NOT_A_DATASET"),
        lambda: eurostat_series("NOT_A_DATASET", filters="geo=XX"),
    ],
)
@patch(HTTP_OPEN)
def test_a_dataset_eurostat_does_not_have_is_not_found(mock_urlopen, call):
    # A 404 is "the requested resource is not available" (the API guide's error table).
    mock_urlopen.side_effect = http_error(404, "Not Found", body=_NOT_DISSEMINATED)

    failure = _failure(call)

    assert failure.error.type == "not_found"
    assert not failure.error.retryable
    assert str(failure).startswith(
        "Eurostat has no dataset NOT_A_DATASET to disseminate; find its ID with "
        "eurostat_dataset_search"
    )


@patch(HTTP_OPEN)
def test_a_404_with_filters_names_the_dataset_and_the_filters(mock_urlopen):
    # The guide reads a 404 as the dataset; the answer does not say which of the two it was.
    mock_urlopen.side_effect = http_error(404, "Not Found", body=_NOT_DISSEMINATED)

    failure = _failure(lambda: eurostat_series("NOT_A_DATASET", filters="geo=XX"))

    assert str(failure) == (
        "Eurostat has no dataset NOT_A_DATASET to disseminate; find its ID with "
        "eurostat_dataset_search (a 404 does not say whether the filters geo=XX were at fault: "
        "eurostat_dataset lists the codes)"
    )


@patch(HTTP_OPEN)
def test_a_query_that_matches_no_data_is_not_found(mock_urlopen):
    # Error 100 with a 400: "The result from the query is empty" (the API guide's error table).
    mock_urlopen.side_effect = http_error(
        400,
        "Bad Request",
        body=b'{"error": [{"status": 400, "id": 100, "label": "No results found"}]}',
    )

    failure = _failure(lambda: eurostat_series("TPS00001", filters="geo=PT,time=1960"))

    assert failure.error.type == "not_found"
    assert str(failure) == (
        "Eurostat has no data of TPS00001 for geo=PT, time=1960 (No results found); other codes "
        "or periods may have some: eurostat_dataset lists the codes"
    )


@patch(HTTP_OPEN)
def test_a_code_the_dataset_does_not_have_is_a_validation_error(mock_urlopen):
    # Error 150 (the API's FAQ: a value outside the dataset's constraint).
    mock_urlopen.side_effect = http_error(
        400,
        "Bad Request",
        body=b'{"error": [{"status": 400, "id": 150, "label": "INVALID_QUERY_DIMENSION_VALUE: '
        b"Query is invalid as per its structure's definition. The following values for "
        b'dimension are not allowed: GEO=EU27."}]}',
    )

    failure = _failure(lambda: eurostat_series("TPS00001", filters="geo=EU27"))

    assert failure.error.type == "validation_error"
    assert str(failure) == (
        "INVALID_QUERY_DIMENSION_VALUE: Query is invalid as per its structure's definition. The "
        "following values for dimension are not allowed: GEO=EU27; list the codes with "
        "eurostat_dataset(dataset_id, dimension=...)"
    )


@patch(HTTP_OPEN)
def test_a_400_without_an_id_is_a_validation_error(mock_urlopen):
    # The guide's own example of an error body.
    mock_urlopen.side_effect = http_error(
        400,
        "Bad Request",
        body=b'{"error":{"status":"400","label":"\'geo\' parameter and \'geoLevel\' parameter '
        b'cannot be set at the same time. Please choose one or the other."}}',
    )

    failure = _failure(lambda: eurostat_series("TPS00001", filters="geo=PT,geoLevel=country"))

    assert failure.error.type == "validation_error"
    assert "cannot be set at the same time" in str(failure)


@patch(HTTP_OPEN)
def test_a_404_on_the_catalogue_says_what_eurostat_said(mock_urlopen):
    mock_urlopen.side_effect = http_error(404, "Not Found", body=_NOT_DISSEMINATED)

    failure = _failure(lambda: eurostat_dataset_search("population"))

    assert failure.error.type == "upstream"
    assert str(failure) == (
        "Eurostat: endpoint not found (HTTP 404); the API may have changed: ERR_NOT_FOUND_4: "
        "NOT_A_DATASET (DATA_FLOW:ALL,1.0) is not available for dissemination."
    )


@patch(HTTP_OPEN)
def test_a_request_eurostat_would_only_serve_later_is_worth_a_retry(mock_urlopen):
    mock_urlopen.side_effect = http_error(413, "Request Entity Too Large", body=_ASYNCHRONOUS)

    failure = _failure(lambda: eurostat_series("nama_10_gdp", filters="geo=ZZ"))

    assert failure.error.type == "upstream"
    assert failure.error.retryable
    assert str(failure) == (
        "ASYNCHRONOUS_RESPONSE. Your request will be treated asynchronously. Please try again "
        "later; try again in a few minutes, or narrow the request with filters"
    )


@patch(HTTP_OPEN)
def test_an_asynchronous_warning_in_a_success_is_worth_a_retry(mock_urlopen):
    # The guide shows it as a warning object.
    mock_urlopen.return_value = respond(
        {
            "warning": {
                "status": 413,
                "label": "ASYNCHRONOUS_RESPONSE. Your request will be treated asynchronously. "
                "Please try again later.",
            }
        }
    )

    failure = _failure(lambda: eurostat_dataset("nama_10_gdp"))

    assert failure.error.type == "upstream"
    assert failure.error.retryable
    assert str(failure).startswith("ASYNCHRONOUS_RESPONSE.")


@patch(HTTP_OPEN)
def test_a_rate_limit_is_rate_limited(mock_urlopen):
    mock_urlopen.side_effect = http_error(429, "Too Many Requests")

    failure = _failure(lambda: eurostat_series("TPS00001", filters="geo=PT+ES"))

    assert failure.error.type == "rate_limited"
    assert failure.error.retryable
