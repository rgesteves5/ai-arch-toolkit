"""Tests for toolkit/tools/_rxnorm_dailymed.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._rxnorm_dailymed import (
    dailymed_label,
    dailymed_label_search,
    rxnorm_concept,
    rxnorm_drug_search,
    rxnorm_ndcs,
    rxnorm_related,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_SETID = "53c11fb4-ba31-b5e5-e063-6394a90a9c1a"


class TestRxNormDailyMed:
    @patch(HTTP_OPEN)
    def test_rxnorm_tools(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "drugGroup": {
                    "conceptGroup": [
                        {
                            "tty": "IN",
                            "conceptProperties": [{"name": "aspirin", "rxcui": "1191"}],
                        }
                    ]
                }
            }
        )
        assert "aspirin | RxCUI: 1191" in rxnorm_drug_search("aspirin")

        mock_urlopen.return_value = respond(
            {"properties": {"name": "aspirin", "tty": "IN", "language": "ENG"}}
        )
        assert "RxNorm concept 1191:" in rxnorm_concept("1191")

        mock_urlopen.return_value = respond(
            {
                "relatedGroup": {
                    "conceptGroup": [
                        {
                            "tty": "SCD",
                            "conceptProperties": [{"name": "aspirin 81 MG", "rxcui": "243670"}],
                        }
                    ]
                }
            }
        )
        assert "aspirin 81 MG" in rxnorm_related("1191", tty="SCD")

        mock_urlopen.return_value = respond({"ndcGroup": {"ndcList": {"ndc": ["0001-0002"]}}})
        assert "0001-0002" in rxnorm_ndcs("1191")

    @patch(HTTP_OPEN)
    def test_dailymed_tools(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "data": [
                    {"title": "ASPIRIN TABLET", "setid": _SETID, "published_date": "Jun 10, 2026"}
                ],
                "metadata": {"total_elements": 1},
            }
        )
        assert _SETID in dailymed_label_search(drug_name="aspirin")

        xml = f"""<?xml version="1.0"?>
        <document xmlns="urn:hl7-org:v3">
          <title>ASPIRIN</title><effectiveTime value="20260608"/><setId root="{_SETID}"/>
          <author><assignedEntity><representedOrganization>
            <name>Example Pharma</name>
          </representedOrganization></assignedEntity></author>
          <component><structuredBody><component><section><title>INDICATIONS</title></section></component></structuredBody></component>
        </document>"""
        mock_urlopen.return_value = respond(xml)
        result = dailymed_label(_SETID)
        assert "ASPIRIN" in result
        assert "sections: INDICATIONS" in result

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        for call, args, kwargs, words in (
            (rxnorm_concept, ("bad",), {}, "invalid rxcui"),
            (rxnorm_related, ("bad",), {}, "invalid rxcui"),
            (rxnorm_related, ("1191",), {"tty": "S C D"}, "invalid tty"),
            (rxnorm_ndcs, ("bad",), {}, "invalid rxcui"),
            (rxnorm_drug_search, ("",), {}, "invalid name"),
            (dailymed_label_search, (), {}, "provide drug_name or ndc"),
            (dailymed_label_search, (), {"ndc": "abc"}, "invalid ndc"),
            (dailymed_label_search, (), {"drug_name": "aspirin", "page": 0}, "page must be"),
            (dailymed_label, ("bad",), {}, "invalid setid"),
        ):
            with pytest.raises(ToolFailure) as caught:
                call(*args, **kwargs)
            assert caught.value.error.type == "validation_error"
            assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_unknown_rxcui_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({})

        with pytest.raises(ToolFailure) as caught:
            rxnorm_concept("999999999")

        assert caught.value.error.type == "not_found"
        assert "rxnorm_drug_search" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_no_match_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"drugGroup": {"name": "zzqq"}})
        assert rxnorm_drug_search("zzqq") == "No RxNorm drug concepts found."

        mock_urlopen.return_value = respond({"data": [], "metadata": {"total_elements": 0}})
        assert dailymed_label_search(drug_name="zzqq") == "No DailyMed labels found."

    @patch(HTTP_OPEN)
    def test_an_unreadable_label_is_an_upstream_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("<not xml")

        with pytest.raises(ToolFailure) as caught:
            dailymed_label(_SETID)

        assert caught.value.error.type == "upstream"
        assert "could not parse XML" in caught.value.error.message


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


@pytest.mark.parametrize("tool", [rxnorm_concept, rxnorm_related, rxnorm_ndcs])
@patch(HTTP_OPEN)
def test_a_404_for_a_concept_is_not_found(mock_urlopen, tool):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(tool, "999999999")

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        "no RxNorm concept with RxCUI 999999999; search with rxnorm_drug_search."
    )


@patch(HTTP_OPEN)
def test_a_404_for_a_label_is_not_found(mock_urlopen):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(dailymed_label, _SETID)

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        f"DailyMed has no label with set ID {_SETID}; find labels with dailymed_label_search"
    )


@pytest.mark.parametrize(
    ("call", "kwargs", "api"),
    [
        (rxnorm_drug_search, {"name": "ibuprofen"}, "RxNorm"),
        (dailymed_label_search, {"drug_name": "ibuprofen"}, "DailyMed"),
    ],
)
@patch(HTTP_OPEN)
def test_a_404_on_a_search_is_an_endpoint_not_found(mock_urlopen, call, kwargs, api):
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(call, **kwargs)

    assert failure.error.type == "upstream"
    assert f"{api}: endpoint not found (HTTP 404)" in failure.error.message
