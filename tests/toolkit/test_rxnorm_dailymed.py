"""Tests for toolkit/tools/_rxnorm_dailymed.py and _spl.py (T07a).

RxNav answers are shaped as its API pages give them (https://lhncbc.nlm.nih.gov/RxNav/APIs/);
DailyMed's as its v2 web services do
(https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm), and the label is an
SPL document: HL7 v3 sections with a LOINC code, a title, CDA narrative and subsections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._rxnorm_dailymed import (
    dailymed_label,
    dailymed_label_search,
    dailymed_label_text,
    rxnorm_concept,
    rxnorm_drug_search,
    rxnorm_ndcs,
    rxnorm_related,
)
from ai_arch_toolkit.toolkit.tools._spl import spl_label
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

SETID = "53c11fb4-ba31-b5e5-e063-6394a90a9c1a"


def concepts(count: int, *, group: str = "drugGroup", tty: str = "SCD") -> dict[str, Any]:
    """An RxNav answer with ``count`` concepts in one term type's group, and an empty group."""
    properties = [
        {"rxcui": str(1000 + n), "name": f"aspirin {n} MG Oral Tablet", "tty": tty}
        for n in range(1, count + 1)
    ]
    return {
        group: {
            "name": None,
            "conceptGroup": [{"tty": "BPCK"}, {"tty": tty, "conceptProperties": properties}],
        }
    }


def spl(*sections: str, title: str = "ASPIRIN 81 MG tablet") -> str:
    """An SPL document with these top-level sections."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <setId root="{SETID}"/>
  <versionNumber value="5"/>
  <title>{title}</title>
  <effectiveTime value="20260608"/>
  <author><assignedEntity><representedOrganization>
    <name>Example Pharma</name>
  </representedOrganization></assignedEntity></author>
  <component><structuredBody>{"".join(sections)}</structuredBody></component>
</document>"""


def section(title: str, text: str = "", *subsections: str, code: str = "42229-5") -> str:
    names = {"34067-9": "INDICATIONS &amp; USAGE SECTION", "42229-5": "SPL UNCLASSIFIED SECTION"}
    return (
        f'<component><section><code code="{code}" codeSystem="2.16.840.1.113883.6.1" '
        f'displayName="{names.get(code, "SECTION")}"/><title>{title}</title>'
        f"<text>{text}</text>{''.join(subsections)}</section></component>"
    )


LABEL = spl(
    section(
        "1 INDICATIONS AND USAGE",
        "<paragraph>Aspirin is indicated for <content styleCode='bold'>pain</content>"
        " relief.</paragraph>",
        section(
            "1.1 Pain",
            "<paragraph>Use for:</paragraph><list listType='unordered'>"
            "<item>headache</item><item>toothache<list><item>molar</item></list></item></list>",
        ),
        code="34067-9",
    ),
    section(
        "2 DOSAGE AND ADMINISTRATION",
        "<table><caption>Dosing</caption><thead><tr><th>Age</th><th>Dose</th></tr></thead>"
        "<tbody><tr><td>Adults</td><td>81 mg<br/>once daily</td></tr></tbody></table>",
    ),
    section("3 OVERDOSAGE", "<paragraph>" + "Call a poison center. " * 60 + "</paragraph>"),
)


def _params(mock_urlopen: MagicMock, call: int = -1) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args_list[call].args[0].full_url).query)


def _path(mock_urlopen: MagicMock, call: int = -1) -> str:
    return urlparse(mock_urlopen.call_args_list[call].args[0].full_url).path


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


class TestRxNormLists:
    @patch(HTTP_OPEN)
    def test_a_search_lists_25_then_reads_on_with_the_total(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = [respond(concepts(30)) for _ in range(2)]

        first = rxnorm_drug_search("aspirin")
        second = rxnorm_drug_search("aspirin", offset=25)

        assert isinstance(first, ToolResult)
        lines = first.value.splitlines()
        assert lines[0] == "RxNorm concepts for 'aspirin' (details: rxnorm_concept):"
        assert lines[1] == (
            "1. aspirin 1 MG Oral Tablet | RxCUI 1001 | TTY SCD (Semantic Clinical Drug)"
        )
        assert lines[-1] == "[results 1-25 of 30 | next: offset=25]"
        assert _text(second).splitlines()[1].startswith("26. aspirin 26 MG")
        assert _text(second).endswith("[results 26-30 of 30 | end]")
        assert _params(mock_urlopen) == {"name": ["aspirin"]}

    @patch(HTTP_OPEN)
    def test_no_match_says_so_with_the_name(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"drugGroup": {"name": None}})

        assert _text(rxnorm_drug_search("zzqq")) == "No RxNorm drug concepts match 'zzqq'."

    @patch(HTTP_OPEN)
    def test_related_without_a_term_type_asks_for_all_related(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(concepts(3, group="allRelatedGroup", tty="IN"))

        text = _text(rxnorm_related("1191"))

        assert _path(mock_urlopen) == "/REST/rxcui/1191/allrelated.json"
        assert "1. aspirin 1 MG Oral Tablet | RxCUI 1001 | TTY IN (Ingredient)" in text

    @pytest.mark.parametrize("tty", ["SBD+SBDF", "SBD SBDF", "sbd, sbdf"])
    @patch(HTTP_OPEN)
    def test_term_types_go_space_separated(self, mock_urlopen: MagicMock, tty: str):
        mock_urlopen.return_value = respond(concepts(2, group="relatedGroup", tty="SBD"))

        rxnorm_related("174742", tty=tty)

        assert _path(mock_urlopen) == "/REST/rxcui/174742/related.json"
        assert _params(mock_urlopen)["tty"] == ["SBD SBDF"]

    @patch(HTTP_OPEN)
    def test_related_reads_on_past_its_page(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(concepts(30, group="relatedGroup"))

        text = _text(rxnorm_related("1191", tty="SCD", max_results=10))

        assert text.endswith("[results 1-10 of 30 | next: offset=10]")

    @patch(HTTP_OPEN)
    def test_no_related_concept_says_so(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"relatedGroup": {"rxcui": ""}})

        assert _text(rxnorm_related("1191", tty="BN")) == (
            "RxNorm lists no related concepts of term types BN for RxCUI 1191."
        )

    @patch(HTTP_OPEN)
    def test_ndcs_read_on_and_point_to_the_labels(self, mock_urlopen: MagicMock):
        ndcs = [f"000694200{n:02d}" for n in range(30)]
        mock_urlopen.return_value = respond({"ndcGroup": {"rxcui": "", "ndcList": {"ndc": ndcs}}})

        text = _text(rxnorm_ndcs("213269"))

        assert text.splitlines()[0] == (
            "NDCs of RxCUI 213269, in the CMS 11-digit form (its labels: "
            'dailymed_label_search(rxcui="213269")):'
        )
        assert text.splitlines()[1] == "1. 00069420000"
        assert text.endswith("[results 1-25 of 30 | next: offset=25]")

    @patch(HTTP_OPEN)
    def test_a_concept_without_ndcs_says_so(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"ndcGroup": {"rxcui": ""}})

        assert _text(rxnorm_ndcs("1191")) == "RxNorm lists no active NDCs for RxCUI 1191."

    @pytest.mark.parametrize(("max_results", "kept"), [(25, True), (26, False), (0, False)])
    @patch(HTTP_OPEN)
    def test_max_results_is_refused_outside_its_limits(self, mock_urlopen, max_results, kept):
        mock_urlopen.return_value = respond(concepts(3, group="relatedGroup"))
        call = ToolCall(
            id="c1", name="rxnorm_related", input={"rxcui": "1191", "max_results": max_results}
        )

        result = ToolGroup(rxnorm_related).execute(call)

        assert result.ok is kept


class TestRxNormConcept:
    @patch(HTTP_OPEN)
    def test_a_concept_names_its_term_type(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            {"properties": {"rxcui": "1191", "name": "aspirin", "tty": "IN", "language": "ENG"}}
        )

        text = _text(rxnorm_concept("1191"))

        assert text.splitlines()[:2] == ["RxNorm concept 1191:", "aspirin"]
        assert "TTY: IN (Ingredient)" in text

    @patch(HTTP_OPEN)
    def test_an_unknown_rxcui_is_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({})

        failure = _failure(rxnorm_concept, "999999999")

        assert failure.error.type == "not_found"
        assert "rxnorm_drug_search" in failure.error.message

    @pytest.mark.parametrize("tool", [rxnorm_concept, rxnorm_related, rxnorm_ndcs])
    @patch(HTTP_OPEN)
    def test_a_404_for_a_concept_is_not_found(self, mock_urlopen: MagicMock, tool: Any):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(tool, "999999999")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "no RxNorm concept with RxCUI 999999999; search with rxnorm_drug_search."
        )

    @patch(HTTP_OPEN)
    def test_a_request_rxnav_cannot_process_is_a_validation_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(400, "Bad Request")

        failure = _failure(rxnorm_related, "1191", tty="XYZ")

        assert failure.error.type == "validation_error"
        assert "HTTP 400" in failure.error.message
        assert "term types such as IN, BN, SCD" in failure.error.message


class TestDailyMedSearch:
    @patch(HTTP_OPEN)
    def test_a_page_of_labels_with_the_total_and_the_next_page(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            {
                "metadata": {"total_elements": "26", "total_pages": "3", "current_page": "1"},
                "data": [
                    {
                        "title": "ASPIRIN TABLET",
                        "setid": SETID,
                        "spl_version": 5,
                        "published_date": "Jun 10, 2026",
                    }
                ],
            }
        )

        text = _text(dailymed_label_search(rxcui="243670", max_results=10))

        assert text.splitlines() == [
            "DailyMed labels for RxCUI 243670 (read one with dailymed_label):",
            "1. ASPIRIN TABLET",
            f"   setid: {SETID} | version 5 | published 2026-06-10",
            "[results 1-1 of 26 | next: page=2]",
        ]
        params = _params(mock_urlopen)
        assert params["rxcui"] == ["243670"]
        assert params["pagesize"] == ["10"]
        assert params["page"] == ["1"]

    @patch(HTTP_OPEN)
    def test_a_later_page_is_numbered_on(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            {
                "metadata": {"total_elements": "11", "total_pages": "2"},
                "data": [{"title": "B", "setid": SETID, "published_date": "Jan 02, 2025"}],
            }
        )

        text = _text(dailymed_label_search(drug_name="aspirin", max_results=10, page=2))

        assert "11. B" in text
        assert text.endswith("[results 11-11 of 11 | end]")

    @patch(HTTP_OPEN)
    def test_no_label_says_so_with_the_search(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"data": [], "metadata": {"total_elements": "0"}})

        assert _text(dailymed_label_search(drug_name="zzqq")) == (
            "No DailyMed labels match drug name 'zzqq'."
        )

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({}, "provide drug_name, ndc or rxcui"),
            ({"ndc": "abc"}, "invalid ndc"),
            ({"rxcui": "abc"}, "invalid rxcui"),
            ({"drug_name": "a<b"}, "invalid drug_name"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_searches_fail_before_asking(self, mock_urlopen, kwargs, words):
        failure = _failure(dailymed_label_search, **kwargs)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_415_is_upstream_with_the_status(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(415, "Unsupported Media Type")

        failure = _failure(dailymed_label_search, drug_name="aspirin")

        assert failure.error.type == "upstream"
        assert "415" in failure.error.message


class TestDailyMedLabel:
    @patch(HTTP_OPEN)
    def test_the_label_gives_its_facts_and_numbered_sections_with_sizes(self, mock_urlopen):
        mock_urlopen.return_value = respond(LABEL)

        text = _text(dailymed_label(SETID))

        lines = text.splitlines()
        assert lines[:5] == [
            f"DailyMed label {SETID}: ASPIRIN 81 MG tablet",
            "   version 5 | effective 2026-06-08 | Example Pharma",
            f"   DailyMed: https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={SETID}",
            f'Sections (read one with dailymed_label_text(setid="{SETID}", section=N)):',
            lines[4],
        ]
        assert lines[4].startswith(
            "1. 1 INDICATIONS AND USAGE [LOINC 34067-9: INDICATIONS & USAGE SECTION]: "
        )
        assert lines[5].startswith("  2. 1.1 Pain [LOINC 42229-5: SPL UNCLASSIFIED SECTION]: ")
        assert lines[6].startswith("3. 2 DOSAGE AND ADMINISTRATION")
        assert lines[7].startswith("4. 3 OVERDOSAGE")
        assert _path(mock_urlopen) == f"/dailymed/services/v2/spls/{SETID}.xml"

    @patch(HTTP_OPEN)
    def test_a_section_size_is_what_dailymed_label_text_returns(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(LABEL) for _ in range(2)]

        outline = _text(dailymed_label(SETID)).splitlines()
        result = dailymed_label_text(SETID, section=1, max_chars=20_000)

        size = int(outline[4].rsplit(": ", 1)[1].removesuffix(" chars"))
        assert result.metadata["window"]["total"] == size

    @patch(HTTP_OPEN)
    def test_a_long_outline_reads_on(self, mock_urlopen: MagicMock):
        many = spl(*(section(f"{n} PART") for n in range(1, 61)))
        mock_urlopen.return_value = respond(many)

        text = _text(dailymed_label(SETID))

        assert text.endswith("[results 1-50 of 60 | next: offset=50]")

    @patch(HTTP_OPEN)
    def test_a_label_with_no_sections_says_so(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(spl())

        assert _text(dailymed_label(SETID)).endswith("The label has no sections.")

    @patch(HTTP_OPEN)
    def test_an_unknown_label_is_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        for tool in (dailymed_label, dailymed_label_text):
            failure = _failure(tool, SETID)
            assert failure.error.type == "not_found"
            assert failure.error.message == (
                f"DailyMed has no label with set ID {SETID}; find labels with "
                "dailymed_label_search"
            )

    @patch(HTTP_OPEN)
    def test_an_unreadable_label_is_upstream(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond("<not xml")

        failure = _failure(dailymed_label, SETID)

        assert failure.error.type == "upstream"
        assert "could not parse" in failure.error.message


class TestDailyMedLabelText:
    @patch(HTTP_OPEN)
    def test_a_section_reads_with_its_subsections_lists_and_tables(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(LABEL) for _ in range(2)]

        indications = _text(dailymed_label_text(SETID, section=1))
        dosage = _text(dailymed_label_text(SETID, section=3))

        assert indications.splitlines() == [
            f"DailyMed label {SETID}, section 1 (1 INDICATIONS AND USAGE):",
            "## 1 INDICATIONS AND USAGE",
            "Aspirin is indicated for pain relief.",
            "### 1.1 Pain",
            "Use for:",
            "- headache",
            "- toothache",
            "  - molar",
        ]
        assert dosage.splitlines()[1:] == [
            "## 2 DOSAGE AND ADMINISTRATION",
            "Table: Dosing",
            "Age | Dose",
            "Adults | 81 mg / once daily",
        ]

    @patch(HTTP_OPEN)
    def test_a_long_section_reads_on_and_names_the_section(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = [respond(LABEL) for _ in range(2)]

        first = dailymed_label_text(SETID, section=4, max_chars=500)
        window = first.metadata["window"]
        second = dailymed_label_text(SETID, section=4, max_chars=500, offset=window["last"])

        assert window["next_call"] == {"offset": window["last"], "section": 4}
        assert second.metadata["window"]["first"] == window["last"]

    @patch(HTTP_OPEN)
    def test_find_searches_the_whole_label(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(LABEL)

        text = _text(dailymed_label_text(SETID, find="toothache"))

        assert "passages that mention 'toothache':" in text
        assert "- toothache" in text
        assert text.endswith('[matches 1-1 of 1 for "toothache" | end]')

    @patch(HTTP_OPEN)
    def test_a_section_the_label_does_not_have_is_a_validation_error(self, mock_urlopen):
        mock_urlopen.return_value = respond(LABEL)

        failure = _failure(dailymed_label_text, SETID, section=9)

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "the label has 4 sections, not 9; list them with dailymed_label"
        )


class TestSpl:
    def test_the_title_and_dates_read_from_the_document(self) -> None:
        label = spl_label(LABEL)

        assert (label.title, label.effective, label.version, label.organization) == (
            "ASPIRIN 81 MG tablet",
            "2026-06-08",
            "5",
            "Example Pharma",
        )
        assert [(s.number, s.depth, s.title) for s in label.sections] == [
            (1, 0, "1 INDICATIONS AND USAGE"),
            (2, 1, "1.1 Pain"),
            (3, 0, "2 DOSAGE AND ADMINISTRATION"),
            (4, 0, "3 OVERDOSAGE"),
        ]

    def test_a_section_without_a_title_is_named_by_its_code(self) -> None:
        label = spl_label(spl(section("", "<paragraph>Text.</paragraph>", code="34067-9")))

        assert label.sections[0].title == "INDICATIONS & USAGE SECTION"
        assert label.text.startswith("## INDICATIONS & USAGE SECTION\nText.")

    def test_the_sources_newlines_are_whitespace_and_paragraphs_in_an_item_stay_apart(
        self,
    ) -> None:
        text = (
            "<paragraph>Take one\n        tablet daily.</paragraph>"
            "<list><item><paragraph>First</paragraph><paragraph>second</paragraph></item></list>"
            "Loose <content>inline</content> words."
        )
        label = spl_label(spl(section("1 USE", text)))

        assert label.text.splitlines() == [
            "## 1 USE",
            "Take one tablet daily.",
            "- First second",
            "Loose inline words.",
        ]

    def test_a_table_in_a_cell_is_read_once(self) -> None:
        inner = "<table><tbody><tr><td>inner</td></tr></tbody></table>"
        outer = f"<table><tbody><tr><td>outer</td><td>{inner}</td></tr></tbody></table>"

        label = spl_label(spl(section("1 TABLE", outer)))

        assert label.text.splitlines()[1:] == ["outer | inner"]


class TestDailyMedEdges:
    @patch(HTTP_OPEN)
    def test_a_label_without_sections_has_no_text_and_says_so(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(spl())

        assert _text(dailymed_label_text(SETID)) == f"DailyMed label {SETID} has no text."

    @patch(HTTP_OPEN)
    def test_without_a_page_count_the_total_still_gives_the_next_page(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"metadata": {"total_elements": 3}, "data": [{"title": "A", "setid": SETID}]}
        )

        text = _text(dailymed_label_search(drug_name="aspirin", max_results=1))

        assert text.endswith("[results 1-1 of 3 | next: page=2]")
