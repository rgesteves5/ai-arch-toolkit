"""Tests for toolkit/tools/_gbif.py (T07b).

The match service resolves scientific names only (``name``: "The scientific name to fuzzy match
against", https://github.com/gbif/matching-ws, ``MatchV1Controller``); the species search covers
"the scientific and vernacular names" (https://github.com/gbif/checklistbank,
``SpeciesResource.search``). A refused request answers 400 with its reason as plain text
(https://github.com/gbif/gbif-common-ws, ``IllegalArgumentExceptionMapper``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolFailure, ToolResult
from ai_arch_toolkit.toolkit.tools._gbif import (
    gbif_occurrence_search,
    gbif_species,
    gbif_species_match,
    gbif_species_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_PUMA = {
    "usageKey": 2435099,
    "scientificName": "Puma concolor (Linnaeus, 1771)",
    "canonicalName": "Puma concolor",
    "rank": "SPECIES",
    "status": "ACCEPTED",
    "confidence": 99,
    "matchType": "EXACT",
    "kingdom": "Animalia",
    "phylum": "Chordata",
    "class": "Mammalia",
    "order": "Carnivora",
    "family": "Felidae",
    "genus": "Puma",
    "species": "Puma concolor",
    "synonym": False,
}


def _failure(call: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _query(mock_urlopen: MagicMock) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


class TestMatch:
    @patch(HTTP_OPEN)
    def test_a_scientific_name_resolves_to_its_key(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(_PUMA)

        text = gbif_species_match("Puma concolor")

        assert text.splitlines() == [
            "GBIF match for 'Puma concolor':",
            "Puma concolor (Linnaeus, 1771) | key 2435099 | SPECIES | ACCEPTED",
            "Match: exact, confidence 99",
            "Classification: Animalia > Chordata > Mammalia > Carnivora > Felidae > Puma > "
            "Puma concolor",
        ]
        assert _query(mock_urlopen)["name"] == ["Puma concolor"]

    @patch(HTTP_OPEN)
    def test_a_match_only_at_a_higher_rank_says_so(self, mock_urlopen: MagicMock) -> None:
        genus = {**_PUMA, "usageKey": 2435098, "scientificName": "Puma Jardine, 1834"}
        genus |= {"rank": "GENUS", "matchType": "HIGHERRANK", "confidence": 90}
        mock_urlopen.return_value = respond(genus)

        assert gbif_species_match("Puma zzz").splitlines()[2] == (
            "Match: higher rank only ('Puma zzz' is not in the backbone; this is its genus), "
            "confidence 90"
        )

    @patch(HTTP_OPEN)
    def test_a_synonym_names_its_accepted_taxon(self, mock_urlopen: MagicMock) -> None:
        synonym = {**_PUMA, "status": "SYNONYM", "synonym": True, "acceptedUsageKey": 2435099}
        synonym |= {"usageKey": 7193927, "matchType": "FUZZY"}
        mock_urlopen.return_value = respond(synonym)

        lines = gbif_species_match("Felis concolr").splitlines()

        assert lines[2] == "Match: fuzzy (the spelling differs), confidence 99"
        assert lines[3] == "Synonym of the taxon with key 2435099 (read it with gbif_species)"

    @patch(HTTP_OPEN)
    def test_no_match_is_not_found_and_points_common_names_to_the_search(
        self, mock_urlopen: MagicMock
    ) -> None:
        # The docstring promised common names; the match service resolves scientific names.
        mock_urlopen.return_value = respond({"confidence": 100, "matchType": "NONE"})

        failure = _failure(lambda: gbif_species_match("mountain lion"))

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "GBIF's backbone has no scientific name matching 'mountain lion'; for a common "
            "name, search with gbif_species_search(query='mountain lion')"
        )

    @patch(HTTP_OPEN)
    def test_several_equal_matches_say_why(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            {"confidence": 100, "note": "Multiple equal matches for Apidae", "matchType": "NONE"}
        )

        failure = _failure(lambda: gbif_species_match("Apidae"))

        assert failure.error.type == "not_found"
        assert "(Multiple equal matches for Apidae)" in failure.error.message
        assert "give the kingdom or the authorship" in failure.error.message


class TestSpeciesSearch:
    @patch(HTTP_OPEN)
    def test_a_search_pages_and_shows_the_common_names_that_matched(
        self, mock_urlopen: MagicMock
    ) -> None:
        taxon = {
            "key": 2435099,
            "scientificName": "Puma concolor (Linnaeus, 1771)",
            "rank": "SPECIES",
            "taxonomicStatus": "ACCEPTED",
            "kingdom": "Animalia",
            "family": "Felidae",
            "vernacularNames": [
                {"vernacularName": "Cougar", "language": "eng"},
                {"vernacularName": "cougar", "language": "eng"},
                {"vernacularName": "Puma", "language": "spa"},
            ],
        }
        mock_urlopen.side_effect = [
            respond(
                {"offset": 0, "limit": 1, "endOfRecords": False, "count": 3, "results": [taxon]}
            ),
            respond(
                {"offset": 2, "limit": 1, "endOfRecords": True, "count": 3, "results": [taxon]}
            ),
        ]

        first = _text(gbif_species_search("cougar", max_results=1))
        last = _text(gbif_species_search("cougar", max_results=1, offset=2))

        assert first.splitlines() == [
            "GBIF taxa that match 'cougar' (scientific and common names):",
            "1. Puma concolor (Linnaeus, 1771) | key 2435099 | SPECIES | ACCEPTED | "
            "Animalia > Felidae | common names: Cougar",
            "[results 1-1 of 3 | next: offset=1]",
        ]
        assert last.endswith("[results 3-3 of 3 | end]")
        assert _query(mock_urlopen)["offset"] == ["2"]

    @patch(HTTP_OPEN)
    def test_the_search_pages_to_the_last_offset_gbif_takes(self, mock_urlopen: MagicMock) -> None:
        # GBIF refuses only an offset past 100,000 (checklistbank, SpeciesResource
        # .checkDeepPaging): the footer stopped at 99,990 while 100,000 was still served.
        rows = [{"key": n, "scientificName": "Puma"} for n in range(10)]
        mock_urlopen.side_effect = [
            respond({"offset": 99_990, "endOfRecords": False, "count": 200_000, "results": rows}),
            respond({"offset": 100_000, "endOfRecords": False, "count": 200_000, "results": rows}),
        ]

        before = _text(gbif_species_search("Puma", offset=99_990))
        last = _text(gbif_species_search("Puma", offset=100_000))

        assert before.endswith("[results 99991-100000 of 200000 | next: offset=100000]")
        assert last.endswith(
            "[results 100001-100010 of 200000 | GBIF's species search pages no further than "
            "offset 100000; narrow the query, the rank or the higher taxon for the rest]"
        )
        assert (
            "(GBIF's species search pages no further than offset 100000;" in (last.splitlines()[0])
        )

    @patch(HTTP_OPEN)
    def test_no_taxa_is_a_success_that_names_the_query(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond({"count": 0, "endOfRecords": True, "results": []})

        assert _text(gbif_species_search("zzqq")) == "No GBIF taxa match 'zzqq'."

    @patch(HTTP_OPEN)
    def test_a_taxon_of_another_checklist_names_its_backbone_key(
        self, mock_urlopen: MagicMock
    ) -> None:
        # Occurrence search takes backbone keys ("A taxon key from the GBIF backbone",
        # OccurrenceSearchResource); a hit from another checklist has its own key.
        taxon = {"key": 100_123, "nubKey": 2435099, "scientificName": "Puma concolor"}
        mock_urlopen.return_value = respond({"count": 1, "endOfRecords": True, "results": [taxon]})

        assert "| key 100123 (backbone key 2435099) |" in _text(gbif_species_search("Puma"))


class TestTaxon:
    @patch(HTTP_OPEN)
    def test_a_taxon_reads_with_its_parent_and_common_name(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            {
                "key": 2435099,
                "scientificName": "Puma concolor (Linnaeus, 1771)",
                "rank": "SPECIES",
                "taxonomicStatus": "ACCEPTED",
                "vernacularName": "Cougar",
                "kingdom": "Animalia",
                "genus": "Puma",
                "parentKey": 2435098,
                "parent": "Puma",
                "publishedIn": "Syst. Nat., 10th ed.",
            }
        )

        assert gbif_species("2435099").splitlines() == [
            "GBIF taxon 2435099:",
            "Puma concolor (Linnaeus, 1771) | SPECIES | ACCEPTED",
            "Common name: Cougar",
            "Classification: Animalia > Puma",
            "Parent: Puma (key 2435098)",
            "Published in: Syst. Nat., 10th ed.",
        ]

    @patch(HTTP_OPEN)
    def test_an_unknown_taxon_key_is_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(lambda: gbif_species("999999999"))

        assert failure.error.type == "not_found"
        assert not failure.error.retryable
        assert str(failure) == "GBIF has no taxon 999999999; find its key with gbif_species_match"

    @patch(HTTP_OPEN)
    def test_a_key_past_gbifs_integers_is_refused_before_any_request(
        self, mock_urlopen: MagicMock
    ) -> None:
        # GBIF reads the key as a 32-bit integer (``@PathVariable int usageKey``).
        failure = _failure(lambda: gbif_species("99999999999"))

        assert failure.error.type == "validation_error"
        mock_urlopen.assert_not_called()


class TestOccurrences:
    @patch(HTTP_OPEN)
    def test_occurrences_read_with_labels_dates_and_a_link(self, mock_urlopen: MagicMock) -> None:
        occurrence = {
            "key": 4011775102,
            "scientificName": "Puma concolor (Linnaeus, 1771)",
            "eventDate": "2024-01-01T10:00:00",
            "locality": "Serra",
            "stateProvince": "Lisboa",
            "country": "Portugal",
            "decimalLatitude": 38.7,
            "decimalLongitude": -0.00001,
            "basisOfRecord": "HUMAN_OBSERVATION",
            "datasetName": "iNaturalist research-grade observations",
        }
        mock_urlopen.return_value = respond(
            {"offset": 0, "limit": 10, "endOfRecords": False, "count": 25, "results": [occurrence]}
        )

        text = _text(gbif_occurrence_search(taxon_key="2435099", country="pt"))

        assert text.splitlines() == [
            "GBIF occurrences of taxon 2435099, country PT, with coordinates:",
            "1. Puma concolor (Linnaeus, 1771) | 2024-01-01T10:00:00 | Serra, Lisboa, Portugal | "
            "38.7, -0.00001 | human observation | iNaturalist research-grade observations | "
            "https://www.gbif.org/occurrence/4011775102",
            "[results 1-1 of 25 | next: offset=1]",
        ]
        query = _query(mock_urlopen)
        assert query["taxonKey"] == ["2435099"]
        assert query["country"] == ["PT"]
        assert query["hasCoordinate"] == ["true"]

    @patch(HTTP_OPEN)
    def test_past_gbifs_depth_the_footer_says_the_rest_is_not_here(
        self, mock_urlopen: MagicMock
    ) -> None:
        # GBIF serves offset + limit up to 100,000 by search (OccurrenceSearchResource).
        rows = [{"key": n, "scientificName": "Puma concolor"} for n in range(5)]
        mock_urlopen.return_value = respond(
            {"offset": 99_995, "endOfRecords": False, "count": 400_000, "results": rows}
        )

        text = _text(gbif_occurrence_search(country="PT", max_results=10, offset=99_995))

        assert _query(mock_urlopen)["limit"] == ["5"]
        assert "(GBIF's search reaches the first 100000;" in text.splitlines()[0]
        # The page asked for 5, not max_results' 10, and said nothing of it.
        assert text.endswith(
            "[results 99996-100000 of 400000 | GBIF's search reaches the first 100000 "
            "(max_results=10 stops there: 5 on this page); narrow the filters, or use GBIF's "
            "download service, for the rest]"
        )

    @patch(HTTP_OPEN)
    def test_a_page_that_ends_at_gbifs_depth_names_no_shortfall(
        self, mock_urlopen: MagicMock
    ) -> None:
        rows = [{"key": n, "scientificName": "Puma concolor"} for n in range(10)]
        mock_urlopen.return_value = respond(
            {"offset": 99_990, "endOfRecords": False, "count": 400_000, "results": rows}
        )

        text = _text(gbif_occurrence_search(country="PT", max_results=10, offset=99_990))

        assert _query(mock_urlopen)["limit"] == ["10"]
        assert text.endswith(
            "[results 99991-100000 of 400000 | GBIF's search reaches the first 100000; narrow "
            "the filters, or use GBIF's download service, for the rest]"
        )

    @patch(HTTP_OPEN)
    def test_no_occurrences_is_a_success_that_names_the_filters(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond({"count": 0, "endOfRecords": True, "results": []})

        assert _text(gbif_occurrence_search(country="PT", year="2020,2024")) == (
            "No GBIF occurrences of country PT, years 2020-2024, with coordinates."
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: gbif_species_match("Puma concolor"),
        lambda: gbif_species_search("Puma"),
        lambda: gbif_species("1"),
        lambda: gbif_occurrence_search(country="PT"),
    ],
)
@patch(HTTP_OPEN)
def test_a_request_gbif_refuses_is_the_callers_to_fix_in_its_words(
    mock_urlopen: MagicMock, call: Any
) -> None:
    mock_urlopen.side_effect = http_error(
        400, "Bad Request", body=b"Offset is limited for this operation to 100000"
    )

    failure = _failure(call)

    assert failure.error.type == "validation_error"
    assert failure.error.message == (
        "GBIF refused the request: Offset is limited for this operation to 100000; "
        "change what it names"
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: gbif_species_match("Puma concolor"),
        lambda: gbif_species_search("Puma"),
        lambda: gbif_occurrence_search(country="PT"),
    ],
)
@patch(HTTP_OPEN)
def test_a_404_on_a_search_is_an_endpoint_not_found(mock_urlopen: MagicMock, call: Any) -> None:
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(call)

    assert failure.error.type == "upstream"
    assert str(failure) == "GBIF: endpoint not found (HTTP 404); the API may have changed"


@patch(HTTP_OPEN)
def test_a_server_error_is_a_retryable_upstream_failure(mock_urlopen: MagicMock) -> None:
    mock_urlopen.side_effect = http_error(503, "Service Unavailable")

    failure = _failure(lambda: gbif_species("1"))

    assert failure.error.type == "upstream"
    assert failure.error.retryable


@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda: gbif_occurrence_search(), "provide taxon_key"),
        (lambda: gbif_occurrence_search(country="POR"), "invalid country code"),
        (lambda: gbif_occurrence_search(year="20"), "invalid year"),
        (lambda: gbif_species("abc"), "invalid taxon_key"),
        (lambda: gbif_species_match(""), "invalid name"),
        (lambda: gbif_species_search("Puma", rank="1"), "invalid rank"),
    ],
)
@patch(HTTP_OPEN)
def test_invalid_arguments_do_not_call_api(mock_urlopen: MagicMock, call: Any, words: str):
    failure = _failure(call)

    assert failure.error.type == "validation_error"
    assert words in str(failure)
    mock_urlopen.assert_not_called()
