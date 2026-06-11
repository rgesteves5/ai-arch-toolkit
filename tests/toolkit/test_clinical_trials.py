"""Tests for toolkit/tools/_clinical_trials.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._clinical_trials import (
    clinical_trial_study,
    clinical_trials_search,
)

_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT04280705",
            "briefTitle": "Adaptive COVID-19 Treatment Trial",
            "officialTitle": "A Randomized Trial of Remdesivir for COVID-19",
        },
        "statusModule": {
            "overallStatus": "COMPLETED",
            "startDateStruct": {"date": "2020-02-21"},
            "primaryCompletionDateStruct": {"date": "2020-05-21"},
            "completionDateStruct": {"date": "2020-05-21"},
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "National Institute of Allergy and Infectious Diseases"}
        },
        "descriptionModule": {"briefSummary": "This study evaluates remdesivir."},
        "conditionsModule": {"conditions": ["COVID-19"]},
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE3"],
            "enrollmentInfo": {"count": 1062, "type": "ACTUAL"},
        },
        "armsInterventionsModule": {
            "armGroups": [
                {
                    "label": "Remdesivir",
                    "type": "EXPERIMENTAL",
                    "description": "Remdesivir arm.",
                }
            ],
            "interventions": [{"type": "DRUG", "name": "Remdesivir"}],
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {
                    "measure": "Time to Recovery",
                    "timeFrame": "Day 1 through Day 29",
                    "description": "First day recovered.",
                }
            ],
            "secondaryOutcomes": [{"measure": "Clinical status", "timeFrame": "Day 15"}],
        },
        "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria: hospitalized adults."},
        "contactsLocationsModule": {
            "locations": [
                {
                    "facility": "University Hospital",
                    "city": "Lisbon",
                    "country": "Portugal",
                    "status": "RECRUITING",
                }
            ]
        },
        "referencesModule": {
            "references": [{"pmid": "32445440", "citation": "Remdesivir paper"}],
            "seeAlsoLinks": [{"label": "Protocol", "url": "https://example.test/protocol"}],
        },
    },
    "hasResults": True,
}


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    if isinstance(data, dict):
        resp.read.return_value = json.dumps(data).encode()
    else:
        resp.read.return_value = data.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestClinicalTrialsSearch:
    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_returns_results_and_next_page_token(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"studies": [_STUDY], "nextPageToken": "NEXT"})

        result = clinical_trials_search("covid", max_results=2)

        assert "ClinicalTrials.gov studies:" in result
        assert "Adaptive COVID-19 Treatment Trial" in result
        assert "NCT ID: NCT04280705 | status: COMPLETED" in result
        assert "type: INTERVENTIONAL | phase: PHASE3" in result
        assert "Conditions: COVID-19" in result
        assert "Interventions: DRUG: Remdesivir" in result
        assert "Has results: True" in result
        assert "Next page token: NEXT" in result

        params = _called_params(mock_urlopen)
        assert params["format"] == ["json"]
        assert params["pageSize"] == ["2"]
        assert params["query.term"] == ["covid"]

    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_filters_status_study_type_phase_and_page_token(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen({"studies": []})

        clinical_trials_search(
            condition="diabetes",
            intervention="insulin",
            location="Portugal",
            status="active not recruiting",
            study_type="interventional",
            phase="phase 3",
            max_results=99,
            page_token="TOKEN",
        )

        params = _called_params(mock_urlopen)
        assert params["query.cond"] == ["diabetes"]
        assert params["query.intr"] == ["insulin"]
        assert params["query.locn"] == ["Portugal"]
        assert params["filter.overallStatus"] == ["ACTIVE_NOT_RECRUITING"]
        assert params["filter.advanced"] == ["AREA[StudyType]INTERVENTIONAL AND AREA[Phase]PHASE3"]
        assert params["pageSize"] == ["20"]
        assert params["pageToken"] == ["TOKEN"]

    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_requires_query_or_page_token(self, mock_urlopen):
        result = clinical_trials_search()

        assert "provide query" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        result = clinical_trials_search("test")

        assert "timed out" in result.lower()

    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("not json")

        result = clinical_trials_search("test")

        assert "could not parse" in result


class TestClinicalTrialStudy:
    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_returns_study(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_STUDY)

        result = clinical_trial_study("nct04280705")

        assert result.startswith("ClinicalTrials.gov study NCT04280705:")
        assert "Official title: A Randomized Trial of Remdesivir for COVID-19" in result
        assert "Arms:" in result
        assert "Primary outcomes:" in result
        assert "Eligibility:" in result
        assert "PMID 32445440 - Remdesivir paper" in result
        assert "https://clinicaltrials.gov/study/NCT04280705" in result

    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_invalid_nct_id(self, mock_urlopen):
        result = clinical_trial_study("bad")

        assert "invalid NCT ID" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._clinical_trials.urllib.request.urlopen")
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://clinicaltrials.gov/api/v2/studies/NCT00000000",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        result = clinical_trial_study("NCT00000000")

        assert "not found" in result.lower()
