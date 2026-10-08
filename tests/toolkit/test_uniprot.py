"""Tests for toolkit/tools/_uniprot.py (T07b).

Answers are shaped as UniProt documents them (https://www.uniprot.org/help/pagination,
https://www.uniprot.org/help/rest-api-headers): a search pages by the cursor in its ``Link``
header and counts in ``x-total-results``; an error says why in ``messages``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolFailure, ToolResult
from ai_arch_toolkit.toolkit.tools._uniprot import (
    uniprot_crossrefs,
    uniprot_entry,
    uniprot_features,
    uniprot_search,
    uniprot_sequence,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_ENTRY: dict[str, Any] = {
    "primaryAccession": "P01308",
    "uniProtkbId": "INS_HUMAN",
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Insulin"}},
        "alternativeNames": [{"fullName": {"value": "Proinsulin"}}],
    },
    "organism": {"scientificName": "Homo sapiens", "commonName": "Human", "taxonId": 9606},
    "proteinExistence": "1: Evidence at protein level",
    "entryAudit": {"firstPublicDate": "1986-07-21", "lastAnnotationUpdateDate": "2026-08-13"},
    "sequence": {"length": 110, "molWeight": 11981},
    "genes": [{"geneName": {"value": "INS"}}],
    "comments": [
        {"commentType": "FUNCTION", "texts": [{"value": "Insulin decreases blood glucose."}]},
        {
            "commentType": "SUBCELLULAR LOCATION",
            "subcellularLocations": [{"location": {"value": "Secreted"}}],
        },
        {
            "commentType": "DISEASE",
            "disease": {
                "diseaseId": "Hyperproinsulinemia",
                "acronym": "HPRI",
                "description": "An autosomal dominant condition.",
            },
        },
    ],
    "keywords": [{"name": "Diabetes mellitus"}, {"name": "Hormone"}],
    "features": [
        {
            "type": "Chain",
            "description": "Insulin B chain",
            "location": {"start": {"value": 25}, "end": {"value": 54}},
        },
        {
            "type": "Natural variant",
            "featureId": "VAR_003971",
            "description": "in MODY10",
            "location": {"start": {"value": 34}, "end": {"value": 34}},
            "alternativeSequence": {"originalSequence": "H", "alternativeSequences": ["D"]},
        },
    ],
    "uniProtKBCrossReferences": [
        {
            "database": "PDB",
            "id": "1TRZ",
            "properties": [
                {"key": "Method", "value": "X-ray"},
                {"key": "Resolution", "value": "1.60 A"},
                {"key": "Chains", "value": "A/C=90-110"},
                {"key": "Extra", "value": "kept"},
            ],
        },
        {"database": "ChEMBL", "id": "CHEMBL5881", "properties": [{"key": "x", "value": "-"}]},
    ],
}


def _result(item: dict[str, Any]) -> dict[str, Any]:
    """A search hit, as ``fields=accession,protein_name,...`` returns it."""
    keys = ("primaryAccession", "uniProtkbId", "entryType", "proteinDescription", "organism")
    return {key: item[key] for key in keys} | {"sequence": {"length": 110}, "genes": item["genes"]}


def _hits(*accessions: str) -> list[dict[str, Any]]:
    return [_result({**_ENTRY, "primaryAccession": accession}) for accession in accessions]


def _page(results: list[dict[str, Any]], total: int, cursor: str = "") -> Any:
    """A search answer: the results, the total in ``x-total-results``, the next page's cursor
    in ``Link`` (https://www.uniprot.org/help/pagination)."""
    response = respond({"results": results})
    response.headers["x-total-results"] = str(total)
    if cursor:
        response.headers["Link"] = (
            "<https://rest.uniprot.org/uniprotkb/search?query=insulin"
            f'&cursor={cursor}&size=2>; rel="next"'
        )
    return response


def _query(mock_urlopen: MagicMock, call: int = -1) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args_list[call].args[0].full_url).query)


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _failure(call: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


class TestSearch:
    @patch(HTTP_OPEN)
    def test_a_search_reads_on_by_the_cursor_the_link_header_gives(
        self, mock_urlopen: MagicMock
    ) -> None:
        # The offset it sent was not a UniProt parameter: every page was the first.
        mock_urlopen.side_effect = [
            _page(_hits("P01308", "P01315"), total=5, cursor="c2"),
            _page(_hits("P01317", "P01318"), total=5, cursor="c3"),
        ]

        first = uniprot_search("insulin", max_results=2)
        second = uniprot_search("insulin", max_results=2, cursor="c2", offset=2)

        assert _text(first).splitlines()[0] == "UniProtKB proteins that match 'insulin':"
        assert "1. Insulin | P01308 (INS_HUMAN) | Homo sapiens" in _text(first)
        assert _text(first).endswith('[results 1-2 of 5 | next: cursor="c2", offset=2]')
        assert "3. Insulin | P01317" in _text(second)
        assert _text(second).endswith('[results 3-4 of 5 | next: cursor="c3", offset=4]')
        sent = _query(mock_urlopen, 0)
        assert "offset" not in sent and "cursor" not in sent
        assert sent["size"] == ["2"]
        assert _query(mock_urlopen, 1)["cursor"] == ["c2"]
        assert isinstance(first, ToolResult)
        assert first.metadata["window"]["next_call"] == {"cursor": "c2", "offset": 2}

    @patch(HTTP_OPEN)
    def test_the_last_page_has_no_link_and_says_the_end(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _page(_hits("P01319"), total=5)

        text = _text(uniprot_search("insulin", max_results=2, cursor="c3", offset=4))

        assert text.endswith("[results 5-5 of 5 | end]")

    @patch(HTTP_OPEN)
    def test_an_offset_without_its_cursor_is_refused_before_any_request(
        self, mock_urlopen: MagicMock
    ) -> None:
        failure = _failure(lambda: uniprot_search("insulin", offset=20))

        assert failure.error.type == "validation_error"
        assert "cursor" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_results_is_a_success_that_names_the_query(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _page([], total=0)

        assert _text(uniprot_search("zzzz", reviewed="true")) == (
            "No UniProtKB proteins match 'zzzz' (reviewed: true)."
        )

    @patch(HTTP_OPEN)
    def test_filters_go_in_the_query_and_a_name_with_spaces_stays_one_value(
        self, mock_urlopen: MagicMock
    ) -> None:
        # "Homo sapiens" went unquoted, as organism_name:Homo AND sapiens; an OR in the query
        # took the filters with it.
        mock_urlopen.side_effect = [_page(_hits("P01308"), total=1) for _ in range(3)]

        uniprot_search("insulin", organism="Homo sapiens", reviewed="true")
        query = _query(mock_urlopen)["query"][0]
        uniprot_search("insulin OR glucagon", organism="9606")
        either = _query(mock_urlopen)["query"][0]
        uniprot_search("insulin")

        assert query == '(insulin) AND organism_name:"Homo sapiens" AND reviewed:true'
        assert either == "(insulin OR glucagon) AND organism_id:9606"
        assert _query(mock_urlopen)["query"][0] == "insulin"
        # The bare collection path answers with a redirect to plain HTTP on port 8080.
        assert urlparse(mock_urlopen.call_args.args[0].full_url).path == "/uniprotkb/search"

    @patch(HTTP_OPEN)
    def test_the_query_syntax_uniprot_takes_passes(self, mock_urlopen: MagicMock) -> None:
        # Phrases and ranges were refused here, though UniProt reads them.
        mock_urlopen.return_value = _page(_hits("P01308"), total=1)

        uniprot_search('"insulin receptor" AND length:[100 TO 200]')

        assert _query(mock_urlopen)["query"] == ['"insulin receptor" AND length:[100 TO 200]']

    @patch(HTTP_OPEN)
    def test_an_unreviewed_entry_is_named_by_its_submission_name(
        self, mock_urlopen: MagicMock
    ) -> None:
        trembl = {
            **_hits("A0A024R161")[0],
            "entryType": "UniProtKB unreviewed (TrEMBL)",
            "proteinDescription": {"submissionNames": [{"fullName": {"value": "Insulin-like"}}]},
        }
        mock_urlopen.return_value = _page([trembl], total=1)

        assert "1. Insulin-like | A0A024R161" in _text(uniprot_search("insulin"))

    @pytest.mark.parametrize(
        ("query", "reason"),
        [
            ("nosuchfield:abc", "'nosuchfield' is not a valid search field"),
            ("insulin AND (", "query parameter has an invalid syntax"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_search_uniprot_refuses_is_the_callers_to_fix_in_uniprots_words(
        self, mock_urlopen: MagicMock, query: str, reason: str
    ) -> None:
        # As answered live (2026-09-30); a 400 is a request the client must change
        # (https://www.uniprot.org/help/rest-api-headers). It read as an upstream failure.
        body = {"url": "http://rest.uniprot.org/uniprotkb/search", "messages": [reason]}
        mock_urlopen.side_effect = http_error(400, "Bad Request", body=json.dumps(body).encode())

        failure = _failure(lambda: uniprot_search(query))

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            f"UniProt refused the request: {reason}; fix what it names "
            "(query syntax: https://www.uniprot.org/help/text-search)"
        )

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_that_moved_not_no_results(
        self, mock_urlopen: MagicMock
    ) -> None:
        # The defect of uniprot_search before 2026-09-28 read as "no matching records found".
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(lambda: uniprot_search("insulin"))

        assert failure.error.type == "upstream"
        assert failure.error.message.startswith("UniProt: endpoint not found (HTTP 404)")

    @patch(HTTP_OPEN)
    def test_rate_limiting_is_typed(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        failure = _failure(lambda: uniprot_search("insulin"))

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable


class TestEntry:
    @patch(HTTP_OPEN)
    def test_the_entry_reads_every_annotation_whole(self, mock_urlopen: MagicMock) -> None:
        # The FUNCTION text was cut at 500 characters, and the other comments never shown.
        function = "Insulin decreases blood glucose. " * 40
        entry = {
            **_ENTRY,
            "comments": [
                {"commentType": "FUNCTION", "texts": [{"value": function}]},
                *_ENTRY["comments"][1:],
            ],
        }
        mock_urlopen.return_value = respond(entry)

        text = _text(uniprot_entry("p01308", max_chars=20_000))

        assert text.splitlines()[0] == "UniProtKB entry P01308 (INS_HUMAN):"
        assert "Protein: Insulin (also: Proinsulin)" in text
        assert "Organism: Homo sapiens (Human), taxon 9606" in text
        assert "Sequence: 110 aa, 11981 Da; read it with uniprot_sequence" in text
        assert "First public: 1986-07-21 | annotation updated: 2026-08-13" in text
        assert "Features: 2 (uniprot_features) | cross-references: 2 (uniprot_crossrefs)" in text
        assert f"## Function\n{function.strip()}" in text
        assert "## Subcellular location\nSecreted" in text
        assert "## Disease\nHyperproinsulinemia (HPRI): An autosomal dominant condition." in text
        assert "Keywords: Diabetes mellitus, Hormone" in text

    @patch(HTTP_OPEN)
    def test_every_kind_of_annotation_reads_as_text(self, mock_urlopen: MagicMock) -> None:
        # Shapes as UniProt's own website reads them
        # (https://github.com/ebi-uniprot/uniprot-website, src/uniprotkb/types/commentTypes.ts).
        comments = [
            {
                "commentType": "CATALYTIC ACTIVITY",
                "reaction": {"name": "ATP + H2O = ADP + phosphate", "ecNumber": "3.6.1.3"},
            },
            {"commentType": "COFACTOR", "cofactors": [{"name": "Mg(2+)"}, {"name": "Zn(2+)"}]},
            {
                "commentType": "BIOPHYSICOCHEMICAL PROPERTIES",
                "kineticParameters": {
                    "michaelisConstants": [{"constant": 1.5e-05, "unit": "mM", "substrate": "ATP"}]
                },
                "absorption": {"max": 280, "approximate": True},
            },
            {
                "commentType": "INTERACTION",
                "interactions": [
                    {
                        "interactantTwo": {"uniProtKBAccession": "P06213", "geneName": "INSR"},
                        "numberOfExperiments": 6,
                    }
                ],
            },
            {
                "commentType": "ALTERNATIVE PRODUCTS",
                "isoforms": [{"name": {"value": "2"}, "isoformIds": ["P01308-2"]}],
            },
            {"commentType": "PTM", "texts": [{"value": "Cleaved."}], "molecule": "Isoform 2"},
        ]
        mock_urlopen.return_value = respond({**_ENTRY, "comments": comments})

        text = _text(uniprot_entry("P01308", max_chars=20_000))

        assert "## Catalytic activity\nATP + H2O = ADP + phosphate (EC 3.6.1.3)" in text
        assert "## Cofactor\nMg(2+), Zn(2+)" in text
        assert (
            "## Biophysicochemical properties\nKM=0.000015 mM for ATP; absorption max ~280 nm"
        ) in text
        assert "## Interaction\nINSR (P06213), 6 experiments" in text
        assert "## Alternative products\n2 (P01308-2)" in text
        assert "## Post-translational modification\n[Isoform 2] Cleaved." in text

    @patch(HTTP_OPEN)
    def test_a_long_entry_reads_window_by_window(self, mock_urlopen: MagicMock) -> None:
        entry = {
            **_ENTRY,
            "comments": [
                {"commentType": "FUNCTION", "texts": [{"value": f"Sentence {n}."}]}
                for n in range(200)
            ],
        }
        mock_urlopen.side_effect = [respond(entry) for _ in range(2)]

        first = uniprot_entry("P01308", max_chars=500)
        assert isinstance(first, ToolResult)
        window = first.metadata["window"]
        second = uniprot_entry("P01308", max_chars=500, offset=window["next_call"]["offset"])

        assert window["first"] == 0 and window["total"] > 2000
        assert _text(first).endswith(f"| next: offset={window['last']}]")
        assert _text(second).splitlines()[0] == "UniProtKB entry P01308 (INS_HUMAN):"
        assert f"[chars {window['last']}-" in _text(second)

    @patch(HTTP_OPEN)
    def test_an_accession_merged_into_another_says_which_entry_answered(
        self, mock_urlopen: MagicMock
    ) -> None:
        # UniProt redirects a merged accession to its entry (303 See Other,
        # https://www.uniprot.org/help/rest-api-headers): the answer is another accession's.
        merged = {**_ENTRY, "primaryAccession": "P23141", "uniProtkbId": "EST1_HUMAN"}
        mock_urlopen.return_value = respond(merged)

        text = _text(uniprot_entry("Q00015"))

        assert text.splitlines()[0] == (
            "UniProtKB entry P23141 (EST1_HUMAN), which UniProt gives for Q00015:"
        )


class TestFeatures:
    @patch(HTTP_OPEN)
    def test_every_feature_can_be_read_page_by_page(self, mock_urlopen: MagicMock) -> None:
        # Only the first 25 were shown, with no count.
        features = [
            {
                "type": "Modified residue",
                "description": f"Phosphoserine {n}",
                "location": {"start": {"value": n}, "end": {"value": n}},
            }
            for n in range(1, 31)
        ]
        mock_urlopen.side_effect = [respond({**_ENTRY, "features": features}) for _ in range(2)]

        first = _text(uniprot_features("P01308", max_results=25))
        second = _text(uniprot_features("P01308", max_results=25, offset=25))

        assert first.splitlines()[:2] == [
            "Features of P01308 (INS_HUMAN):",
            "Types: Modified residue 30",
        ]
        assert "1. Modified residue 1: Phosphoserine 1" in first
        assert first.endswith("[results 1-25 of 30 | next: offset=25]")
        assert "30. Modified residue 30: Phosphoserine 30" in second
        assert second.endswith("[results 26-30 of 30 | end]")

    @patch(HTTP_OPEN)
    def test_a_feature_shows_its_range_its_change_and_its_id(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(_ENTRY)

        text = _text(uniprot_features("P01308"))

        assert "1. Chain 25-54: Insulin B chain" in text
        assert "2. Natural variant 34: H -> D, in MODY10 [VAR_003971]" in text

    @patch(HTTP_OPEN)
    def test_a_type_with_no_features_names_the_types_there_are(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(_ENTRY)

        assert _text(uniprot_features("P01308", feature_type="Domain")) == (
            "P01308 has no features of type 'Domain'; its types: Chain 1, Natural variant 1."
        )

    @patch(HTTP_OPEN)
    def test_the_type_filter_ignores_case(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(_ENTRY)

        text = _text(uniprot_features("P01308", feature_type="natural VARIANT"))

        assert text.splitlines()[0] == "Natural variant features of P01308 (INS_HUMAN):"
        assert "1. Natural variant 34" in text
        assert "Chain" not in text


class TestCrossrefs:
    @patch(HTTP_OPEN)
    def test_every_cross_reference_can_be_read_page_by_page(self, mock_urlopen: MagicMock) -> None:
        refs = [{"database": "PDB", "id": f"{n}ABC", "properties": []} for n in range(1, 31)]
        entry = {**_ENTRY, "uniProtKBCrossReferences": refs}
        mock_urlopen.side_effect = [respond(entry) for _ in range(2)]

        first = _text(uniprot_crossrefs("P01308"))
        second = _text(uniprot_crossrefs("P01308", offset=25))

        assert first.splitlines()[1] == "Databases: PDB 30"
        assert first.endswith("[results 1-25 of 30 | next: offset=25]")
        assert "30. PDB: 30ABC" in second

    @patch(HTTP_OPEN)
    def test_a_reference_keeps_all_its_properties(self, mock_urlopen: MagicMock) -> None:
        # The fourth property and on were dropped.
        mock_urlopen.return_value = respond(_ENTRY)

        text = _text(uniprot_crossrefs("P01308", database="pdb"))

        assert text.splitlines()[0] == "PDB cross-references of P01308 (INS_HUMAN):"
        assert (
            "1. PDB: 1TRZ | Method: X-ray, Resolution: 1.60 A, Chains: A/C=90-110, Extra: kept"
        ) in text

    @patch(HTTP_OPEN)
    def test_a_database_with_none_names_the_databases_there_are(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(_ENTRY)

        assert _text(uniprot_crossrefs("P01308", database="Reactome")) == (
            "P01308 has no cross-references to Reactome; its databases: ChEMBL 1, PDB 1."
        )


@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda: uniprot_entry("bad/id"), "invalid accession 'bad/id'"),
        (lambda: uniprot_sequence("x"), "invalid accession 'x'"),
        # Six characters, but not UniProt's format (https://www.uniprot.org/help/accession_numbers).
        (lambda: uniprot_entry("ABC123"), "invalid accession 'ABC123'"),
        (lambda: uniprot_search("insulin", reviewed="maybe"), "invalid reviewed 'maybe'"),
        (lambda: uniprot_search("  "), "query cannot be empty"),
        (lambda: uniprot_search("a\x00b"), "invalid query"),
        (lambda: uniprot_features("P01308", feature_type="a\nb"), "invalid feature_type"),
        (lambda: uniprot_crossrefs("P01308", database="a;b"), "invalid database"),
    ],
)
@patch(HTTP_OPEN)
def test_invalid_arguments_do_not_call_the_api(mock_urlopen: MagicMock, call: Any, words: str):
    failure = _failure(call)

    assert failure.error.type == "validation_error"
    assert words in failure.error.message
    mock_urlopen.assert_not_called()


@pytest.mark.parametrize("accession", ["P01308", "A0A023GPI8", "q00015", "P01308-2"])
@patch(HTTP_OPEN)
def test_accessions_in_uniprots_format_are_taken(mock_urlopen: MagicMock, accession: str):
    mock_urlopen.return_value = respond(">sp|X|Y\nMALW")

    assert uniprot_sequence(accession).startswith(">sp|")
    assert urlparse(mock_urlopen.call_args.args[0].full_url).path == (
        f"/uniprotkb/{accession.upper()}.fasta"
    )


@pytest.mark.parametrize(
    "fn", [uniprot_entry, uniprot_features, uniprot_crossrefs, uniprot_sequence]
)
@patch(HTTP_OPEN)
def test_a_404_on_an_accession_is_not_found(mock_urlopen: MagicMock, fn: Any) -> None:
    # UniProt answers an accession it does not have 404, {"messages": ["Resource not found"]}
    # (https://www.uniprot.org/help/rest-api-headers).
    body = {"url": "https://rest.uniprot.org/uniprotkb/Q00000", "messages": ["Resource not found"]}
    mock_urlopen.side_effect = http_error(404, "Not Found", body=json.dumps(body).encode())

    failure = _failure(lambda: fn("Q00000"))

    assert failure.error.type == "not_found"
    assert failure.error.message == "UniProt has no entry Q00000; find one with uniprot_search"


# Inactive accessions as UniProt answered them live, with HTTP 200 (2026-09-30).
_DEMERGED = {
    "entryType": "Inactive",
    "primaryAccession": "P00001",
    "uniProtkbId": "CYC_HUMAN",
    "annotationScore": 0.0,
    "inactiveReason": {"inactiveReasonType": "DEMERGED", "mergeDemergeTo": ["P99999", "P99998"]},
    "extraAttributes": {"uniParcId": "UPI0000128BBF"},
}
_DELETED = {
    "entryType": "Inactive",
    "primaryAccession": "A0A008APQ8",
    "uniProtkbId": "A0A008APQ8_STAAU",
    "annotationScore": 0.0,
    "inactiveReason": {
        "inactiveReasonType": "DELETED",
        "deletedReason": "Not part of a reference proteome",
    },
    "extraAttributes": {"uniParcId": "UPI0000054276"},
}


@pytest.mark.parametrize("fn", [uniprot_entry, uniprot_features, uniprot_crossrefs])
@patch(HTTP_OPEN)
def test_an_inactive_accession_says_where_it_went(mock_urlopen: MagicMock, fn: Any) -> None:
    # It read as a nameless entry, or as an entry with no features or cross-references.
    mock_urlopen.return_value = respond(_DEMERGED)

    failure = _failure(lambda: fn("P00001"))

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        "P00001 is inactive: demerged into P99999, P99998; look up one of them with uniprot_entry"
    )


@patch(HTTP_OPEN)
def test_a_merged_accession_names_the_one_to_look_up(mock_urlopen: MagicMock) -> None:
    merged = {
        **_DEMERGED,
        "inactiveReason": {"inactiveReasonType": "MERGED", "mergeDemergeTo": ["P99999"]},
    }
    mock_urlopen.return_value = respond(merged)

    failure = _failure(lambda: uniprot_entry("P00001"))

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        "P00001 is inactive: merged into P99999; look up P99999 with uniprot_entry"
    )


@patch(HTTP_OPEN)
def test_a_deleted_accession_says_why(mock_urlopen: MagicMock) -> None:
    mock_urlopen.return_value = respond(_DELETED)

    failure = _failure(lambda: uniprot_entry("A0A008APQ8"))

    assert failure.error.type == "not_found"
    assert failure.error.message == (
        "A0A008APQ8 is inactive: deleted (Not part of a reference proteome); "
        "search for the protein with uniprot_search"
    )


@patch(HTTP_OPEN)
def test_an_answer_that_is_not_fasta_is_not_passed_off_as_the_sequence(
    mock_urlopen: MagicMock,
) -> None:
    # A merged accession redirects (303) to a URL without the format asked for.
    mock_urlopen.return_value = respond(_ENTRY)

    failure = _failure(lambda: uniprot_sequence("Q00015"))

    assert failure.error.type == "upstream"
    assert failure.error.message == (
        "UniProt answered Q00015 with no FASTA; read the entry with uniprot_entry"
    )


@patch(HTTP_OPEN)
def test_an_empty_sequence_is_not_found(mock_urlopen: MagicMock) -> None:
    mock_urlopen.return_value = respond("", content_type="text/plain")

    failure = _failure(lambda: uniprot_sequence("P01308"))

    assert failure.error.type == "not_found"
    assert "uniprot_entry" in failure.error.message
