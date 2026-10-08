"""Tests for toolkit/tools/_nvd.py (T06)."""

from __future__ import annotations

import email.message
import io
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._nvd import nvd_cve, nvd_cve_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import NVD_REFUSED_REASON, nvd_item, nvd_page


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok, result
    assert isinstance(result.value, str)
    return result.value


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _invalid(fn: Any, *args: Any, **kwargs: Any) -> str:
    failure = _failure(fn, *args, **kwargs)
    assert failure.error.type == "validation_error"
    return failure.error.message


def _params(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _refused(message: str) -> urllib.error.HTTPError:
    """A 404 that says why in its ``message`` header, as NVD refuses a request."""
    headers = email.message.Message()
    headers["message"] = message
    return urllib.error.HTTPError(
        "https://services.nvd.nist.gov/rest/json/cves/2.0", 404, "", headers, io.BytesIO()
    )


class TestNvdCveSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            nvd_page(nvd_item("CVE-2021-44228"), nvd_item("CVE-2021-45046"), total=321)
        )

        result = nvd_cve_search(query="log4j", max_results=2)

        text = _text(result)
        assert text.splitlines()[:6] == [
            "NVD CVEs that match keyword 'log4j':",
            "1. CVE-2021-44228",
            "   CVSS 3.1: 10.0 CRITICAL | CVSS 2.0: 9.3 HIGH | status: Analyzed | "
            "published: 2021-12-10T10:15:09",
            "   Description: Apache Log4j2 JNDI features do not protect against LDAP.",
            "   Weaknesses: CWE-917, CWE-502",
            "   CPEs: 3 | References: 2 (nvd_cve('CVE-2021-44228') lists them)",
        ]
        assert text.endswith("[results 1-2 of 321 | next: start=2]")
        params = _params(mock_urlopen)
        assert params["keywordSearch"] == ["log4j"]
        assert params["resultsPerPage"] == ["2"]
        assert params["startIndex"] == ["0"]

    @patch(HTTP_OPEN)
    def test_the_last_page_says_end(self, mock_urlopen):
        mock_urlopen.return_value = respond(nvd_page(nvd_item(), total=3, start=2))

        text = _text(nvd_cve_search(query="log4j", max_results=2, start=2))

        assert text.splitlines()[1] == "3. CVE-2021-44228"
        assert text.endswith("[results 3-3 of 3 | end]")

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_filters(self, mock_urlopen):
        mock_urlopen.return_value = respond(nvd_page(total=0))

        text = _text(
            nvd_cve_search(
                query="zzqq",
                cvss_severity="high",
                pub_start_date="2024-01-01",
                pub_end_date="2024-01-31",
            )
        )

        assert text == (
            "No NVD CVEs match keyword 'zzqq', CVSS v3 severity HIGH, published 2024-01-01 to "
            "2024-01-31."
        )

    @patch(HTTP_OPEN)
    def test_filters_go_as_nvd_takes_them(self, mock_urlopen):
        mock_urlopen.return_value = respond(nvd_page(total=0))

        nvd_cve_search(
            query="log4j",
            cpe_name="cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
            cvss_severity="critical",
            pub_start_date="2021-12-01",
            pub_end_date="2022-03-30",
        )

        params = _params(mock_urlopen)
        assert params["cpeName"] == ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"]
        assert params["cvssV3Severity"] == ["CRITICAL"]
        assert params["pubStartDate"] == ["2021-12-01T00:00:00.000"]
        assert params["pubEndDate"] == ["2022-03-30T23:59:59.999"]

    @patch(HTTP_OPEN)
    def test_a_range_over_nvds_120_days_is_refused_before_the_request(self, mock_urlopen):
        # "The maximum allowable range when using any date range parameters is 120 consecutive
        # days" (https://nvd.nist.gov/developers/vulnerabilities).
        message = _invalid(
            nvd_cve_search, query="log4j", pub_start_date="2021-01-01", pub_end_date="2021-05-01"
        )

        assert message == (
            "NVD takes a publication range of at most 120 consecutive days, and 2021-01-01 to "
            "2021-05-01 is 121; split it into ranges of 120 days or fewer"
        )
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        ("max_results", "start", "kept"), [(20, 0, True), (21, 0, False), (5, -1, False)]
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, start, kept):
        mock_urlopen.return_value = respond(nvd_page(total=0))
        call = ToolCall(
            id="c1",
            name="nvd_cve_search",
            input={"query": "x", "max_results": max_results, "start": start},
        )

        assert ToolGroup(nvd_cve_search).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "no filter given" in _invalid(nvd_cve_search)
        assert "invalid CVE ID" in _invalid(nvd_cve_search, cve_id="bad")
        assert "invalid cvss_severity" in _invalid(nvd_cve_search, query="x", cvss_severity="x")
        assert "provided together" in _invalid(
            nvd_cve_search, query="x", pub_start_date="2024-01-01"
        )
        assert "invalid pub_start_date" in _invalid(
            nvd_cve_search, query="x", pub_start_date="01-01-2024", pub_end_date="2024-01-02"
        )
        assert "before or equal" in _invalid(
            nvd_cve_search, query="x", pub_start_date="2024-02-01", pub_end_date="2024-01-01"
        )
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        failure = _failure(nvd_cve_search, query="log4j")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable


class TestNvdErrors:
    @patch(HTTP_OPEN)
    def test_a_refused_request_says_why_from_the_message_header(self, mock_urlopen):
        mock_urlopen.side_effect = _refused(NVD_REFUSED_REASON)

        failure = _failure(nvd_cve_search, query="log4j", cpe_name="cpe:bad")

        assert failure.error.type == "validation_error"
        assert not failure.error.retryable
        assert failure.error.message == (
            f"NVD refused the request: {NVD_REFUSED_REASON}; correct that parameter"
        )

    @patch(HTTP_OPEN)
    def test_a_404_without_a_reason_is_an_endpoint_that_moved(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(nvd_cve_search, query="log4j")

        assert failure.error.type == "upstream"
        assert failure.error.message.startswith("NVD: endpoint not found (HTTP 404)")

    @patch(HTTP_OPEN)
    def test_a_message_header_on_a_success_is_not_an_error(self, mock_urlopen):
        answer = respond(nvd_page(nvd_item(), total=1))
        answer.headers["message"] = "informational"
        mock_urlopen.return_value = answer

        assert _text(nvd_cve("CVE-2021-44228")).startswith("NVD CVE CVE-2021-44228:")


class TestNvdCve:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole_every_cvss_with_its_version(self, mock_urlopen):
        mock_urlopen.return_value = respond(nvd_page(nvd_item(cpes=12, references=7), total=1))

        text = _text(nvd_cve("cve-2021-44228"))

        lines = text.splitlines()
        assert lines[:9] == [
            "NVD CVE CVE-2021-44228:",
            "CVE-2021-44228",
            "status: Analyzed | published: 2021-12-10T10:15:09 | last modified: "
            "2024-11-21T08:15:28",
            "Description: Apache Log4j2 JNDI features do not protect against LDAP.",
            "CVSS (2):",
            "- CVSS 3.1: 10.0 CRITICAL (Primary, nvd@nist.gov) "
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            "- CVSS 2.0: 9.3 HIGH (Primary, nvd@nist.gov) AV:N/AC:M/Au:N/C:C/I:C/A:C",
            "Weaknesses: CWE-917 (Primary, nvd@nist.gov); CWE-502 (Secondary, "
            "security@apache.org)",
            "CPEs (12):",
        ]
        assert (
            "- cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:12 (vulnerable; from 2.0.1 including, "
            "to 2.3.1 excluding)"
        ) in text
        assert "References (7):" in text
        assert "- https://example.test/advisory/7 (Exploit, Third Party Advisory)" in text
        assert text.endswith("URL: https://nvd.nist.gov/vuln/detail/CVE-2021-44228\n")
        assert _params(mock_urlopen)["cveId"] == ["CVE-2021-44228"]

    @patch(HTTP_OPEN)
    def test_a_long_record_reads_on_through_the_window(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(nvd_page(nvd_item(cpes=80), total=1)) for _ in "ab"]

        first = nvd_cve("CVE-2021-44228", max_chars=1000)
        last = first.metadata["window"]["last"]
        second = nvd_cve("CVE-2021-44228", max_chars=1000, offset=last)

        assert _text(first).endswith(f"next: offset={last}]")
        assert second.metadata["window"]["first"] == last
        assert "Description:" in _text(first)

    @patch(HTTP_OPEN)
    def test_invalid_cve_id(self, mock_urlopen):
        assert "invalid CVE ID" in _invalid(nvd_cve, "bad")
        assert "cannot be empty" in _invalid(nvd_cve, " ")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_unknown_cve_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond(nvd_page(total=0))

        failure = _failure(nvd_cve, "CVE-1999-99999")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "NVD has no CVE CVE-1999-99999; check the ID, or search by keyword with nvd_cve_search"
        )


@patch(HTTP_OPEN)
def test_a_publication_date_range_alone_is_a_search(mock_urlopen):
    # NVD takes pubStartDate and pubEndDate as the only filter.
    mock_urlopen.return_value = respond(nvd_page(nvd_item(), total=1))

    text = _text(nvd_cve_search(pub_start_date="2024-01-01", pub_end_date="2024-01-31"))

    assert text.startswith("NVD CVEs that match published 2024-01-01 to 2024-01-31:")
    params = _params(mock_urlopen)
    assert "keywordSearch" not in params
    assert params["pubStartDate"] == ["2024-01-01T00:00:00.000"]
