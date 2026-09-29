"""Tests for toolkit/tools/_eurostat.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._eurostat import (
    eurostat_compare,
    eurostat_dataset,
    eurostat_dataset_search,
    eurostat_dimensions,
    eurostat_series,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond

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
        assert "invalid dataset_id" in eurostat_dataset("bad/id")
