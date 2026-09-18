"""Tests for toolkit/tools/_rxnorm_dailymed.py."""

from __future__ import annotations

from unittest.mock import patch

from ai_arch_toolkit.toolkit.tools._rxnorm_dailymed import (
    dailymed_label,
    dailymed_label_search,
    rxnorm_concept,
    rxnorm_drug_search,
    rxnorm_ndcs,
    rxnorm_related,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond

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
        assert "invalid rxcui" in rxnorm_concept("bad")
        assert "provide drug_name" in dailymed_label_search()
        mock_urlopen.assert_not_called()
