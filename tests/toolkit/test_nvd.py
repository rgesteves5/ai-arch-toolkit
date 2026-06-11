"""Tests for toolkit/tools/_nvd.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._nvd import nvd_cve, nvd_cve_search

_CVE = {
    "cve": {
        "id": "CVE-2021-44228",
        "published": "2021-12-10T10:15:09.000",
        "lastModified": "2024-11-21T08:15:28.000",
        "vulnStatus": "Analyzed",
        "descriptions": [{"lang": "en", "value": "Apache Log4j vulnerability"}],
        "metrics": {
            "cvssMetricV31": [{"cvssData": {"baseScore": "10.0", "baseSeverity": "CRITICAL"}}]
        },
        "weaknesses": [{"description": [{"lang": "en", "value": "CWE-502"}]}],
        "configurations": [
            {"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*"}]}]}
        ],
        "references": {"referenceData": [{"url": "https://example.test/advisory"}]},
    }
}


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestNvdCveSearch:
    @patch("ai_arch_toolkit.toolkit.tools._nvd._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._nvd.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen({"vulnerabilities": [_CVE]})

        result = nvd_cve_search(
            query="log4j",
            cvss_severity="critical",
            max_results=2,
            start=3,
            pub_start_date="2021-12-01",
            pub_end_date="2021-12-31",
        )

        assert "NVD CVE results" in result
        assert "CVE-2021-44228" in result
        assert "CVSS: 10" in result
        assert "severity: CRITICAL" in result
        assert "Apache Log4j vulnerability" in result
        assert "CWE-502" in result

        params = _called_params(mock_urlopen)
        assert params["keywordSearch"] == ["log4j"]
        assert params["cvssV3Severity"] == ["CRITICAL"]
        assert params["resultsPerPage"] == ["2"]
        assert params["startIndex"] == ["3"]
        assert params["pubStartDate"] == ["2021-12-01T00:00:00.000"]
        assert params["pubEndDate"] == ["2021-12-31T23:59:59.999"]

    @patch("ai_arch_toolkit.toolkit.tools._nvd.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "provide query" in nvd_cve_search()
        assert "invalid CVE ID" in nvd_cve_search(cve_id="CVE-bad")
        assert "cvss_severity" in nvd_cve_search(cvss_severity="URGENT")
        assert "start must" in nvd_cve_search(query="x", start=-1)
        assert "provided together" in nvd_cve_search(query="x", pub_start_date="2024-01-01")
        assert "invalid pub_start_date" in nvd_cve_search(
            query="x", pub_start_date="01-01-2024", pub_end_date="2024-02-01"
        )
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._nvd._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._nvd.urllib.request.urlopen")
    def test_rate_limited(self, mock_urlopen, _mock_throttle):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://services.nvd.nist.gov/rest/json/cves/2.0",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

        result = nvd_cve_search(query="test")

        assert "rate limited" in result


class TestNvdCve:
    @patch("ai_arch_toolkit.toolkit.tools._nvd._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._nvd.urllib.request.urlopen")
    def test_returns_cve(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen({"vulnerabilities": [_CVE]})

        result = nvd_cve("cve-2021-44228")

        assert result.startswith("NVD CVE CVE-2021-44228:")
        assert "Apache Log4j vulnerability" in result
        assert _called_params(mock_urlopen)["cveId"] == ["CVE-2021-44228"]

    @patch("ai_arch_toolkit.toolkit.tools._nvd.urllib.request.urlopen")
    def test_invalid_cve_id(self, mock_urlopen):
        result = nvd_cve("bad")

        assert "invalid CVE ID" in result
        mock_urlopen.assert_not_called()
