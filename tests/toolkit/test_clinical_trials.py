"""Tests for toolkit/tools/_clinical_trials.py (T07a).

The answers are shaped as the API v2's OpenAPI spec gives them (https://clinicaltrials.gov/api/oas/v2):
a page of studies with ``totalCount`` (``countTotal=true``) and ``nextPageToken``, a study's record
with its ``markup`` fields in markdown, and a refused request as a ``text/plain`` 400.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._clinical_trials import (
    clinical_trial_study,
    clinical_trials_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_SUMMARY = "This study evaluates remdesivir. " + "It measures recovery in hospital. " * 40
_CRITERIA = (
    "Inclusion Criteria:\n\n"
    "* Admitted to a hospital with symptoms suggestive of COVID-19.\n"
    "* Aged 18 years or older.\n\n"
    "Exclusion Criteria:\n\n"
    "* Pregnancy or breast feeding.\n"
    "* Allergy to any study medication."
)


def study(nct_id: str = "NCT04280705", *, sites: int = 12, arms: int = 8) -> dict[str, Any]:
    """A study record as the API v2 sends it, longer than the old cuts in every list."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": "Adaptive COVID-19 Treatment Trial",
                "officialTitle": "A Randomized Trial of Remdesivir for COVID-19",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-02-21"},
                "primaryCompletionDateStruct": {"date": "2020-04-19"},
                "completionDateStruct": {"date": "2020-05-21"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "National Institute of Allergy and Infectious Diseases"}
            },
            "descriptionModule": {
                "briefSummary": _SUMMARY,
                "detailedDescription": "First paragraph.\n\nSecond paragraph.",
            },
            "conditionsModule": {"conditions": ["COVID-19"]},
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 1062, "type": "ACTUAL"},
            },
            "armsInterventionsModule": {
                "armGroups": [
                    {
                        "label": f"Arm {n}",
                        "type": "EXPERIMENTAL",
                        "description": f"Description of arm {n}.",
                        "interventionNames": ["Drug: Remdesivir"],
                    }
                    for n in range(1, arms + 1)
                ],
                "interventions": [
                    {"type": "DRUG", "name": "Remdesivir", "description": "200 mg on day 1."}
                ],
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": f"Primary outcome {n}",
                        "timeFrame": "Day 1 through Day 29",
                        "description": "First day recovered.",
                    }
                    for n in range(1, 9)
                ],
                "secondaryOutcomes": [{"measure": "Clinical status", "timeFrame": "Day 15"}],
            },
            "eligibilityModule": {
                "eligibilityCriteria": _CRITERIA,
                "healthyVolunteers": False,
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "99 Years",
            },
            "contactsLocationsModule": {
                "locations": [
                    {
                        "facility": f"Hospital {n}",
                        "city": "Lisbon" if n == sites else "Boston",
                        "country": "Portugal" if n == sites else "United States",
                        "status": "RECRUITING",
                    }
                    for n in range(1, sites + 1)
                ]
            },
            "referencesModule": {
                "references": [
                    {"pmid": str(32445440 + n), "citation": f"Paper {n}"} for n in range(1, 7)
                ],
                "seeAlsoLinks": [{"label": "Protocol", "url": "https://example.test/protocol"}],
            },
        },
        "hasResults": True,
    }


def _page(*ids: str, token: str = "", total: int | None = None) -> dict[str, Any]:
    page: dict[str, Any] = {"studies": [study(nct_id) for nct_id in ids]}
    if token:
        page["nextPageToken"] = token
    if total is not None:
        page["totalCount"] = total
    return page


def _params(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


class TestSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_call_for_the_next(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            _page("NCT04280705", "NCT04280706", token="NEXT", total=57)
        )

        result = clinical_trials_search("covid", max_results=2)

        assert isinstance(result, ToolResult)
        text = result.value
        assert text.splitlines()[0] == (
            "ClinicalTrials.gov studies for query 'covid' (read one with clinical_trial_study):"
        )
        assert "1. Adaptive COVID-19 Treatment Trial" in text
        assert "   NCT04280705 | status: COMPLETED | type: INTERVENTIONAL | phase: PHASE3" in text
        assert "2. Adaptive COVID-19 Treatment Trial" in text
        assert "   Enrollment: 1062 participants (actual)" in text
        assert text.endswith('[results 1-2 of 57 | next: page_token="NEXT", offset=2]')
        params = _params(mock_urlopen)
        assert params["countTotal"] == ["true"]
        assert params["pageSize"] == ["2"]
        assert params["query.term"] == ["covid"]

    @patch(HTTP_OPEN)
    def test_the_next_page_is_numbered_on_from_the_offset(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(_page("NCT04280707"))

        text = _text(clinical_trials_search("covid", page_token="NEXT", offset=2))

        assert "3. Adaptive COVID-19 Treatment Trial" in text
        assert text.endswith("[results 3-3 | end]")
        assert _params(mock_urlopen)["pageToken"] == ["NEXT"]

    @patch(HTTP_OPEN)
    def test_the_search_lists_no_summary_and_names_the_record_tool(self, mock_urlopen):
        mock_urlopen.return_value = respond(_page("NCT04280705", total=1))

        text = _text(clinical_trials_search("covid"))

        assert "It measures recovery" not in text  # the record has it, whole
        assert "[truncated]" not in text
        assert "clinical_trial_study" in text.splitlines()[0]

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_search(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"studies": [], "totalCount": 0})

        text = _text(clinical_trials_search(condition="zzqqxx", status="recruiting"))

        assert text == (
            "No ClinicalTrials.gov studies match condition 'zzqqxx', status RECRUITING."
        )

    @patch(HTTP_OPEN)
    def test_filters_go_as_the_api_takes_them(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"studies": []})

        clinical_trials_search(
            condition="diabetes",
            intervention="insulin",
            location="Portugal",
            status="active not recruiting",
            study_type="interventional",
            phase="phase 3",
            max_results=20,
        )

        params = _params(mock_urlopen)
        assert params["query.cond"] == ["diabetes"]
        assert params["query.intr"] == ["insulin"]
        assert params["query.locn"] == ["Portugal"]
        assert params["filter.overallStatus"] == ["ACTIVE_NOT_RECRUITING"]
        assert params["filter.advanced"] == ["AREA[StudyType]INTERVENTIONAL AND AREA[Phase]PHASE3"]
        assert params["pageSize"] == ["20"]

    @pytest.mark.parametrize(("max_results", "kept"), [(20, True), (21, False), (0, False)])
    @patch(HTTP_OPEN)
    def test_max_results_is_refused_outside_its_limits(self, mock_urlopen, max_results, kept):
        mock_urlopen.return_value = respond({"studies": []})
        call = ToolCall(
            id="c1",
            name="clinical_trials_search",
            input={"query": "asthma", "max_results": max_results},
        )

        result = ToolGroup(clinical_trials_search).execute(call)

        assert result.ok is kept
        if not kept:
            assert result.error is not None
            assert result.error.type == "validation_error"

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({}, "provide query, condition, intervention, or location"),
            ({"page_token": "NEXT"}, "provide query, condition, intervention, or location"),
            ({"query": "asthma", "offset": 5}, "offset numbers the page a page_token reads"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_search_without_terms_or_an_offset_without_a_token_is_refused(
        self, mock_urlopen: MagicMock, kwargs: dict[str, Any], words: str
    ):
        failure = _failure(clinical_trials_search, **kwargs)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_refused_search_is_a_validation_error_with_the_sources_words(
        self, mock_urlopen: MagicMock
    ):
        body = b"filter.overallStatus: unknown value 'RECRUITNG'"
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=body)

        failure = _failure(clinical_trials_search, "covid", status="recruitng")

        assert failure.error.type == "validation_error"
        assert "unknown value 'RECRUITNG'" in failure.error.message
        assert "check the terms' syntax and the filter values" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_timeout_and_a_404_are_upstream(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = TimeoutError()
        timed_out = _failure(clinical_trials_search, "test")
        assert (timed_out.error.type, timed_out.error.retryable) == ("upstream", True)

        mock_urlopen.side_effect = http_error(404, "Not Found")
        moved = _failure(clinical_trials_search, "test")
        assert moved.error.type == "upstream"
        assert "ClinicalTrials.gov: endpoint not found (HTTP 404)" in moved.error.message


class TestStudy:
    @patch(HTTP_OPEN)
    def test_the_record_comes_whole_with_its_sections_and_sizes(self, mock_urlopen):
        mock_urlopen.return_value = respond(study())

        result = clinical_trial_study("nct04280705", max_chars=20_000)

        assert isinstance(result, ToolResult)
        text = result.value
        lines = text.splitlines()
        assert (
            lines[0] == "ClinicalTrials.gov study NCT04280705: Adaptive COVID-19 Treatment Trial"
        )
        assert lines[1].startswith("Sections: overview (")
        assert "arms (8, " in lines[1]
        assert "locations (12, " in lines[1]
        assert "references (7, " in lines[1]
        assert _SUMMARY.strip() in text  # once cut at 900 characters
        assert "[truncated]" not in text
        assert "Official title: A Randomized Trial of Remdesivir for COVID-19" in text
        assert "PMID 32445441: Paper 1" in text
        assert "https://clinicaltrials.gov/study/NCT04280705" in text
        assert result.metadata["window"]["next_call"] is None

    @patch(HTTP_OPEN)
    def test_the_eligibility_criteria_keep_their_lines(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(study())

        text = _text(clinical_trial_study("NCT04280705", section="eligibility"))

        assert text.splitlines()[2:] == [
            "NCT04280705, section eligibility:",
            "## Eligibility",
            "Sex: ALL | ages: 18 Years to 99 Years | healthy volunteers: no",
            "Inclusion Criteria:",
            "* Admitted to a hospital with symptoms suggestive of COVID-19.",
            "* Aged 18 years or older.",
            "Exclusion Criteria:",
            "* Pregnancy or breast feeding.",
            "* Allergy to any study medication.",
        ]

    @pytest.mark.parametrize(
        ("section", "count", "line"),
        [
            ("arms", 8, "- EXPERIMENTAL: Arm 8: Description of arm 8. Interventions: Drug: "),
            ("outcomes", 8, "- Primary outcome 8 (Day 1 through Day 29): First day recovered."),
            ("locations", 12, "- Hospital 12, Lisbon, Portugal (RECRUITING)"),
            ("references", 7, "- Protocol: https://example.test/protocol"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_every_arm_outcome_site_and_reference_is_there(
        self, mock_urlopen: MagicMock, section: str, count: int, line: str
    ):
        mock_urlopen.return_value = respond(study())

        text = _text(clinical_trial_study("NCT04280705", section=section, max_chars=20_000))

        assert any(row.startswith(line) for row in text.splitlines()), text
        assert sum(row.startswith("- ") for row in text.splitlines()) >= count

    @patch(HTTP_OPEN)
    def test_a_long_section_reads_on_window_by_window_and_names_the_section(
        self, mock_urlopen: MagicMock
    ):
        mock_urlopen.side_effect = [respond(study(sites=200)) for _ in range(2)]

        first = clinical_trial_study("NCT04280705", section="locations", max_chars=500)
        window = first.metadata["window"]
        second = clinical_trial_study(
            "NCT04280705", section="locations", max_chars=500, offset=window["last"]
        )

        assert window["next_call"] == {"offset": window["last"], "section": "locations"}
        assert first.value.endswith(f'next: offset={window["last"]}, section="locations"]')
        assert second.metadata["window"]["first"] == window["last"]
        assert "- Hospital 1," not in second.value

    @patch(HTTP_OPEN)
    def test_find_gives_the_passages_that_mention_a_term(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(study(sites=200))

        text = _text(clinical_trial_study("NCT04280705", find="Lisbon"))

        assert "passages that mention 'Lisbon':" in text
        assert "- Hospital 200, Lisbon, Portugal (RECRUITING)" in text
        assert text.endswith('[matches 1-1 of 1 for "Lisbon" | end]')

    @patch(HTTP_OPEN)
    def test_an_empty_section_says_so(self, mock_urlopen: MagicMock):
        record = study()
        del record["protocolSection"]["descriptionModule"]["detailedDescription"]
        mock_urlopen.return_value = respond(record)

        text = _text(clinical_trial_study("NCT04280705", section="description"))

        assert text.endswith("NCT04280705 has no description.")

    @patch(HTTP_OPEN)
    def test_an_unknown_study_is_not_found_with_the_next_step(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(404, "Not Found", body=b"Study not found")

        failure = _failure(clinical_trial_study, "NCT00000000")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "no ClinicalTrials.gov study with NCT ID NCT00000000; search with "
            "clinical_trials_search."
        )

    @patch(HTTP_OPEN)
    def test_a_record_without_a_protocol_is_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"hasResults": False})

        assert _failure(clinical_trial_study, "NCT00000000").error.type == "not_found"

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"nct_id": "bad"}, "invalid NCT ID"),
            ({"nct_id": "NCT04280705", "section": "budget"}, "invalid section 'budget'"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_bad_id_or_section_fails_before_asking(self, mock_urlopen, kwargs, words):
        failure = _failure(clinical_trial_study, **kwargs)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_other_statuses_stay_upstream(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        failure = _failure(clinical_trial_study, "NCT04280705")

        assert (failure.error.type, failure.error.retryable) == ("upstream", True)
