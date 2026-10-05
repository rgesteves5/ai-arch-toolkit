"""Tests for toolkit/tools/_eurostat.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._eurostat import (
    eurostat_compare,
    eurostat_dataset,
    eurostat_dataset_search,
    eurostat_dimensions,
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
_DATASET = {
    "label": "Population on 1 January",
    "source": "ESTAT",
    "updated": "2026-04-30T23:00:00+0200",
    "id": ["freq", "geo", "time"],
    "size": [1, 2, 1],
    "value": {"0": 10, "1": 20},
    "dimension": {
        "freq": {
            "label": "Time frequency",
            "category": {"index": {"A": 0}, "label": {"A": "Annual"}},
        },
        "geo": {
            "label": "Geopolitical entity",
            "category": {"index": {"PT": 0, "ES": 1}, "label": {"PT": "Portugal", "ES": "Spain"}},
        },
        "time": {"label": "Time", "category": {"index": {"2025": 0}, "label": {"2025": "2025"}}},
    },
    "extension": {
        "description": "<p>Population description</p>",
        "annotation": [{"type": "OBS_COUNT", "title": "592"}],
    },
}


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestEurostat:
    @patch(HTTP_OPEN)
    def test_dataset_search(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(_DATAFLOW), respond(_DATAFLOW)]

        result = eurostat_dataset_search("population")

        assert result.splitlines()[1:] == ["1. TPS00001 — Population on 1 January"]
        assert "1. EARN_SES_AGT15 — Mean hourly earnings" in eurostat_dataset_search("earn_ses")
        # The full catalogue is 20 MB, over the response cap; its stubs are about 1.5 MB.
        assert _params(mock_urlopen)["detail"] == ["allstubs"]

    @patch(HTTP_OPEN)
    def test_dataset_dimensions_and_series(self, mock_urlopen):
        mock_urlopen.return_value = respond(_DATASET)
        assert "Population description" in eurostat_dataset("TPS00001")

        mock_urlopen.return_value = respond(_DATASET)
        assert "geo — Geopolitical entity" in eurostat_dimensions("TPS00001")

        mock_urlopen.return_value = respond(_DATASET)
        result = eurostat_series("TPS00001", filters="geo=PT", last_time_periods=1)
        assert "10 | freq=A, geo=PT, time=2025" in result
        assert _params(mock_urlopen)["geo"] == ["PT"]

    @patch(HTTP_OPEN)
    def test_compare_and_validation(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(_DATASET), respond(_DATASET)]

        result = eurostat_compare("TPS00001", "PT,ES")

        assert "Eurostat comparison TPS00001:" in result
        assert "geo=PT" in result

    @patch(HTTP_OPEN)
    def test_no_datasets_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond(_DATAFLOW)

        assert eurostat_dataset_search("zzzz") == "No Eurostat datasets found."

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: eurostat_dataset("bad/id"), "invalid dataset_id"),
            (lambda: eurostat_dataset_search(""), "invalid query"),
            (lambda: eurostat_dataset_search("population", offset=-1), "offset must"),
            (lambda: eurostat_series("TPS00001", filters="geo"), "use key=value"),
            (lambda: eurostat_series("TPS00001", filters="geo=P T"), "invalid filter value"),
            (lambda: eurostat_compare("TPS00001", ""), "1-10 comma-separated geo_codes"),
            (lambda: eurostat_compare("TPS00001", "P T"), "invalid geo code"),
            (lambda: eurostat_compare("TPS00001", "PT", filters="geo=ES"), "via geo_codes"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_arguments_do_not_call_api(self, mock_urlopen, call, words):
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
        lambda: eurostat_dimensions("NOT_A_DATASET"),
        lambda: eurostat_series("NOT_A_DATASET"),
    ],
)
@patch(HTTP_OPEN)
def test_a_dataset_eurostat_does_not_have_is_explained(mock_urlopen, call):
    mock_urlopen.side_effect = http_error(404, "Not Found", body=_NOT_DISSEMINATED)

    failure = _failure(call)

    # The 404 stays upstream until T02 declares it per endpoint.
    assert failure.error.type == "upstream"
    assert str(failure) == (
        "HTTP error 404: ERR_NOT_FOUND_4: NOT_A_DATASET (DATA_FLOW:ALL,1.0) is not "
        "available for dissemination."
    )


@patch(HTTP_OPEN)
def test_a_request_eurostat_would_only_serve_later_says_so(mock_urlopen):
    mock_urlopen.side_effect = http_error(413, "Request Entity Too Large", body=_ASYNCHRONOUS)

    failure = _failure(lambda: eurostat_series("nama_10_gdp", filters="geo=ZZ"))

    assert failure.error.type == "upstream"
    assert str(failure) == (
        "HTTP error 413: ASYNCHRONOUS_RESPONSE. Your request will be "
        "treated asynchronously. Please try again later."
    )


@patch(HTTP_OPEN)
def test_a_rate_limit_is_rate_limited(mock_urlopen):
    mock_urlopen.side_effect = http_error(429, "Too Many Requests")

    failure = _failure(lambda: eurostat_compare("TPS00001", "PT,ES"))

    assert failure.error.type == "rate_limited"
    assert failure.error.retryable
