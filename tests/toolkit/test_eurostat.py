"""Tests for toolkit/tools/_eurostat.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._eurostat import (
    eurostat_compare,
    eurostat_dataset,
    eurostat_dataset_search,
    eurostat_dimensions,
    eurostat_series,
)

_DATAFLOW = {
    "link": {
        "item": [
            {
                "label": "Population on 1 January",
                "extension": {
                    "id": "TPS00001",
                    "description": "<p>Population dataset</p>",
                    "annotation": [
                        {"type": "OBS_COUNT", "title": "592"},
                        {"type": "OBS_PERIOD_OVERALL_OLDEST", "title": "2014"},
                        {"type": "OBS_PERIOD_OVERALL_LATEST", "title": "2025"},
                    ],
                },
            }
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


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestEurostat:
    @patch("ai_arch_toolkit.toolkit.tools._eurostat.urllib.request.urlopen")
    def test_dataset_search(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_DATAFLOW)

        result = eurostat_dataset_search("population")

        assert "TPS00001 — Population on 1 January" in result
        assert "period: 2014-2025" in result

    @patch("ai_arch_toolkit.toolkit.tools._eurostat.urllib.request.urlopen")
    def test_dataset_dimensions_and_series(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_DATASET)
        assert "Population description" in eurostat_dataset("TPS00001")

        mock_urlopen.return_value = _mock_urlopen(_DATASET)
        assert "geo — Geopolitical entity" in eurostat_dimensions("TPS00001")

        mock_urlopen.return_value = _mock_urlopen(_DATASET)
        result = eurostat_series("TPS00001", filters="geo=PT", last_time_periods=1)
        assert "10 | freq=A, geo=PT, time=2025" in result
        assert _params(mock_urlopen)["geo"] == ["PT"]

    @patch("ai_arch_toolkit.toolkit.tools._eurostat.urllib.request.urlopen")
    def test_compare_and_validation(self, mock_urlopen):
        mock_urlopen.side_effect = [_mock_urlopen(_DATASET), _mock_urlopen(_DATASET)]

        result = eurostat_compare("TPS00001", "PT,ES")

        assert "Eurostat comparison TPS00001:" in result
        assert "geo=PT" in result
        assert "invalid dataset_id" in eurostat_dataset("bad/id")
