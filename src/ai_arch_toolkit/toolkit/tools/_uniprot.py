"""UniProt tools: search UniProtKB, and read an entry's annotation, features, cross-references
and sequence (T07b; D39, D41).

A search pages by cursor: UniProt takes ``size`` and ``cursor``, no offset, and answers the next
page's URL in a ``Link`` header and the total in ``x-total-results``
(https://www.uniprot.org/help/pagination, https://www.uniprot.org/help/api_queries,
https://www.uniprot.org/help/rest-api-headers). The window's footer gives the cursor, and the
offset that numbers the next page.

``uniprot_entry``, ``uniprot_features`` and ``uniprot_crossrefs`` read the same entry, each a
different part of it with a filter of its own: three jobs, so three tools (D41; the note in
T07b). Each part goes through the window whole: the annotation by characters, the lists by item.
"""

from __future__ import annotations

import re
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import list_window, page_window, text_window


def _uniprot_error(reply: Reply) -> ToolFailure | str | None:
    """The error a UniProt answer explains; ``None`` for a result.

    A request UniProt refuses says why in ``messages`` (https://www.uniprot.org/help/rest-api-headers):
    a 400 is one "the client request needs modification", so ``validation_error`` in UniProt's
    words; any other status keeps them as the source's text. An inactive accession is
    ``not_found`` (:func:`_inactive`).
    """
    answer = reply.body
    messages = answer.get("messages") if isinstance(answer, dict) else None
    said = "; ".join(filter(None, map(_string, messages))) if isinstance(messages, list) else ""
    if said and reply.status == 400:
        return ToolFailure(
            "validation_error",
            f"UniProt refused the request: {said.rstrip('.')}; fix what it names "
            "(query syntax: https://www.uniprot.org/help/text-search)",
        )
    return said or _inactive(answer)


def _inactive(answer: object) -> ToolFailure | None:
    """``not_found`` for an accession UniProt says is inactive, naming where it went; ``None``
    otherwise.

    An accession demerged or deleted (https://www.uniprot.org/help/deleted_accessions) still
    answers HTTP 200: ``{"entryType": "Inactive", "inactiveReason": {...}}`` (seen 2026-09-30),
    which read as an entry with no name, features or cross-references. It has no entry as asked:
    a merged or demerged one lives on under the accessions it names.
    """
    if not isinstance(answer, dict) or answer.get("entryType") != "Inactive":
        return None
    accession = _string(answer.get("primaryAccession")) or "the accession"
    reason = answer.get("inactiveReason")
    reason = reason if isinstance(reason, dict) else {}
    kind = _string(reason.get("inactiveReasonType")).lower()
    targets = reason.get("mergeDemergeTo")
    if kind and isinstance(targets, list) and targets:
        into = [_string(target) for target in targets]
        which = into[0] if len(into) == 1 else "one of them"
        msg = (
            f"{accession} is inactive: {kind} into {', '.join(into)}; "
            f"look up {which} with uniprot_entry"
        )
        return ToolFailure("not_found", msg)
    why = _string(reason.get("deletedReason"))
    detail = (f": {kind}" if kind else "") + (f" ({why})" if why else "")
    msg = f"{accession} is inactive{detail}; search for the protein with uniprot_search"
    return ToolFailure("not_found", msg)


_API = Api(
    base="https://rest.uniprot.org/uniprotkb",
    name="UniProt",
    timeout_s=20,
    error_reader=_uniprot_error,
)
# UniProt's accession format (https://www.uniprot.org/help/accession_numbers), and an isoform's
# "-N" (P05067-2).
_ACCESSION_RE = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?"
)
# A feature type or a database, as UniProt names them ("Modified residue", "PDB").
_NAME_RE = re.compile(r"[\w ()+,./'-]{1,80}")
_QUERY_CHARS = 1000
_FILTER_CHARS = 200
_MAX_CHARS = 20_000
_DEFAULT_CHARS = 6_000
_SEARCH_FIELDS = "accession,id,protein_name,gene_names,organism_name,reviewed,length"
# The next page's URL in a ``Link`` header (RFC 8288; https://www.uniprot.org/help/pagination).
_NEXT_RE = re.compile(r'<([^>]*)>\s*;\s*rel="?next"?')


@tool(capability="network")
def uniprot_search(
    query: str,
    organism: str = "",
    reviewed: str = "",
    max_results: Annotated[int, Range(1, 25)] = 10,
    cursor: str = "",
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search UniProtKB proteins, the most relevant first.

    The query takes UniProt's syntax: words, "a phrase", and fields such as gene:INS,
    organism_id:9606 or length:[100 TO 200], joined with AND, OR and NOT
    (https://www.uniprot.org/help/text-search).

    Args:
        query: What to look for.
        organism: An organism's name or NCBI taxonomy ID, e.g. "Homo sapiens" or "9606".
        reviewed: "true" for reviewed (Swiss-Prot) entries, "false" for unreviewed (TrEMBL),
            empty for both.
        max_results: How many proteins to list.
        cursor: The cursor of the next page, from the footer of the page before.
        offset: How many results came before that page, from the same footer.

    Raises:
        ToolFailure: validation_error when an argument is invalid, an offset comes without its
            cursor, or UniProt refuses the query (with its reason).
    """
    text = _free_text("query", query, _QUERY_CHARS)
    place = _free_text("organism", organism, _FILTER_CHARS) if organism.strip() else ""
    if reviewed and reviewed.lower() not in {"true", "false"}:
        raise ToolFailure(
            "validation_error",
            f"invalid reviewed {reviewed[:100]!r}; use 'true', 'false', or '' for both",
        )
    if offset and not cursor.strip():
        raise ToolFailure(
            "validation_error",
            "UniProt pages by cursor: an offset only numbers the page its cursor reaches; pass "
            "both from the previous page's footer, or neither for the first page",
        )
    params = {
        "query": _search_query(text, place, reviewed.lower()),
        "format": "json",
        "size": str(max_results),
        "fields": _SEARCH_FIELDS,
    }
    if cursor.strip():
        params["cursor"] = _free_text("cursor", cursor, _FILTER_CHARS)
    label = _search_label(text, place, reviewed.lower())
    return _API.get_json_reply(
        "search",
        params=params,
        parse=lambda reply: _search_answer(reply, label, offset, first_page=not cursor.strip()),
    )


@tool(capability="network")
def uniprot_entry(
    accession: str,
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Read a UniProtKB entry: names, organism, sequence size, dates, and every annotation
    (function, catalytic activity, subcellular location, disease, interactions …) in full.

    Args:
        accession: A UniProt accession, e.g. "P01308".
        offset: Where to start, in characters; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when the accession is malformed; not_found when UniProt has
            no entry for it, or the accession is inactive (merged, demerged or deleted).
    """
    asked = _accession(accession)

    def read(data: dict[str, Any]) -> ToolResult:
        window = text_window(_annotation(data), offset=offset, limit=max_chars)
        return window.result(heading=f"UniProtKB entry {_named(data, asked)}:")

    return _API.get_json(asked, params={"format": "json"}, parse=read, missing=_missing(asked))


@tool(capability="network")
def uniprot_features(
    accession: str,
    feature_type: str = "",
    max_results: Annotated[int, Range(1, 25)] = 20,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """List a UniProtKB entry's sequence features: domains, sites, modified residues, natural
    variants … with their positions.

    Args:
        accession: A UniProt accession, e.g. "P01308".
        feature_type: Only features of this type, e.g. "Domain", "Active site" or
            "Natural variant"; the first page lists the entry's types.
        max_results: How many features to list.
        offset: How many features to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the accession or the type is malformed; not_found
            when UniProt has no entry for it, or the accession is inactive.
    """
    asked = _accession(accession)
    wanted = _filter("feature_type", feature_type)

    def read(data: dict[str, Any]) -> ToolResult:
        return _FEATURES.answer(data, asked, wanted, offset=offset, limit=max_results)

    return _API.get_json(asked, params={"format": "json"}, parse=read, missing=_missing(asked))


@tool(capability="network")
def uniprot_crossrefs(
    accession: str,
    database: str = "",
    max_results: Annotated[int, Range(1, 25)] = 25,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """List a UniProtKB entry's cross-references to other databases, with what each says.

    A PDB ID reads on with pdb_entry, a ChEMBL ID with chembl_target.

    Args:
        accession: A UniProt accession, e.g. "P01308".
        database: Only references to this database, e.g. "PDB", "Reactome" or "ChEMBL"; the
            first page lists the entry's databases.
        max_results: How many references to list.
        offset: How many references to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the accession or the database is malformed;
            not_found when UniProt has no entry for it, or the accession is inactive.
    """
    asked = _accession(accession)
    wanted = _filter("database", database)

    def read(data: dict[str, Any]) -> ToolResult:
        return _CROSSREFS.answer(data, asked, wanted, offset=offset, limit=max_results)

    return _API.get_json(asked, params={"format": "json"}, parse=read, missing=_missing(asked))


@tool(capability="network")
def uniprot_sequence(accession: str) -> str:
    """Get a UniProtKB protein sequence in FASTA form.

    Args:
        accession: A UniProt accession, e.g. "P01308".

    Raises:
        ToolFailure: validation_error when the accession is malformed; not_found when UniProt
            has no entry or no sequence for it.
    """
    asked = _accession(accession)
    fasta = _API.get_text(f"{asked}.fasta", parse=str.strip, missing=_missing(asked))
    if not fasta:
        msg = f"UniProt has no sequence for {asked}; check the entry with uniprot_entry"
        raise ToolFailure("not_found", msg)
    if not fasta.startswith(">"):  # a merged accession's redirect may drop the format
        msg = f"UniProt answered {asked} with no FASTA; read the entry with uniprot_entry"
        raise ToolFailure("upstream", msg)
    return fasta


# --- Arguments ---------------------------------------------------------------------------------


def _accession(accession: str) -> str:
    """``accession`` upper-cased; a malformed one raises ``ToolFailure`` (validation_error)."""
    normalized = accession.strip().upper()
    if not _ACCESSION_RE.fullmatch(normalized):
        raise ToolFailure(
            "validation_error",
            f"invalid accession {accession[:100]!r}; a UniProt accession looks like P01308 or "
            "A0A023GPI8 (find one with uniprot_search)",
        )
    return normalized


def _free_text(name: str, value: str, limit: int) -> str:
    """``value`` stripped: what UniProt reads, refused only when empty, too long, or holding a
    control character."""
    text = value.strip()
    if not text:
        raise ToolFailure("validation_error", f"{name} cannot be empty; say what to look for")
    if len(text) > limit or any(ord(char) < 32 for char in text):
        raise ToolFailure(
            "validation_error",
            f"invalid {name} {value[:100]!r}; give up to {limit} characters on one line",
        )
    return text


def _filter(name: str, value: str) -> str:
    """A feature type or a database to keep, as given (matched ignoring case); empty for all."""
    wanted = value.strip()
    if wanted and not _NAME_RE.fullmatch(wanted):
        example = "'Domain'" if name == "feature_type" else "'PDB'"
        raise ToolFailure(
            "validation_error",
            f"invalid {name} {value[:100]!r}; name it as UniProt does, e.g. {example}",
        )
    return wanted


def _missing(accession: str) -> str:
    """The not_found message for an accession UniProt has no entry for (its 404)."""
    return f"UniProt has no entry {accession}; find one with uniprot_search"


def _quoted(value: str) -> str:
    """A field's value as one term: quoted when it has a space."""
    return f'"{value}"' if " " in value and not value.startswith('"') else value


def _search_query(query: str, organism: str, reviewed: str) -> str:
    """The query and its filters; the query in parentheses, so an OR in it stays inside."""
    filters = []
    if organism:
        field = "organism_id" if organism.isdigit() else "organism_name"
        filters.append(f"{field}:{_quoted(organism)}")
    if reviewed:
        filters.append(f"reviewed:{reviewed}")
    return " AND ".join([f"({query})", *filters]) if filters else query


def _search_label(query: str, organism: str, reviewed: str) -> str:
    """The query as answers name it, with its filters: ``'insulin' (reviewed: true)``."""
    filters = [
        f"{name}: {value}"
        for name, value in (("organism", organism), ("reviewed", reviewed))
        if value
    ]
    return f"{query!r}" + (f" ({', '.join(filters)})" if filters else "")


# --- Answers -----------------------------------------------------------------------------------


def _search_answer(reply: Reply, label: str, offset: int, *, first_page: bool) -> ToolResult:
    body = reply.body if isinstance(reply.body, dict) else {}
    results = [item for item in body.get("results") or [] if isinstance(item, dict)]
    if not results and first_page:
        return ToolResult.success(f"No UniProtKB proteins match {label}.")
    lines = [_hit(offset + number, item) for number, item in enumerate(results, start=1)]
    after = _next_cursor(reply.headers.get("link", ""))
    next_call = {"cursor": after, "offset": offset + len(lines)} if after and lines else None
    total = _integer(reply.headers.get("x-total-results"))
    window = list_window(lines, first=offset + 1, total=total, next_call=next_call)
    return window.result(heading=f"UniProtKB proteins that match {label}:")


def _next_cursor(link: str) -> str:
    """The cursor of the page a ``Link`` header names as next; empty on the last page."""
    match = _NEXT_RE.search(link)
    if match is None:
        return ""
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(match[1]).query)
    return (query.get("cursor") or [""])[0]


def _hit(number: int, item: dict[str, Any]) -> str:
    parts = [
        f"{number}. {_protein_name(item) or '(no protein name)'}",
        _identity(item),
        _nested(item, "organism", "scientificName") or "?",
        _status(item),
    ]
    length = _nested(item, "sequence", "length")
    if length:
        parts.append(f"{length} aa")
    if genes := _gene_names(item):
        parts.append(f"genes: {genes}")
    return " | ".join(parts)


def _identity(item: dict[str, Any]) -> str:
    """``P01308 (INS_HUMAN)``: the accession, and the entry name when there is one."""
    accession = _string(item.get("primaryAccession"))
    name = _string(item.get("uniProtkbId"))
    return f"{accession} ({name})" if name else accession


def _named(data: dict[str, Any], asked: str) -> str:
    """The entry as a heading names it, saying so when UniProt answered for another accession
    (a merged one redirects to its entry: https://www.uniprot.org/help/rest-api-headers)."""
    identity = _identity(data)
    answered = _string(data.get("primaryAccession"))
    return identity if answered == asked else f"{identity}, which UniProt gives for {asked}"


def _status(item: dict[str, Any]) -> str:
    """``reviewed (Swiss-Prot)`` or ``unreviewed (TrEMBL)``, as the entry type says."""
    return _string(item.get("entryType")).removeprefix("UniProtKB ") or "?"


def _annotation(data: dict[str, Any]) -> str:
    """The entry as text: a summary, then each annotation under its heading."""
    lines = [*_summary(data), *_comments(data.get("comments"))]
    return "\n".join(lines)


def _summary(data: dict[str, Any]) -> list[str]:
    """The entry's first lines: what it is, of what organism, how long, how recent, and the
    sizes of the lists the other tools read."""
    others = _alternative_names(data)
    organism = _nested(data, "organism", "scientificName") or "?"
    common = _nested(data, "organism", "commonName")
    taxon = _nested(data, "organism", "taxonId")
    existence = _string(data.get("proteinExistence")) or "?"
    length = _nested(data, "sequence", "length") or "?"
    mass = _nested(data, "sequence", "molWeight") or "?"
    public = _nested(data, "entryAudit", "firstPublicDate") or "?"
    updated = _nested(data, "entryAudit", "lastAnnotationUpdateDate") or "?"
    features = _count(data.get("features"))
    refs = _count(data.get("uniProtKBCrossReferences"))
    lines = [
        f"Status: {_status(data)} | protein existence: {existence}",
        f"Protein: {_protein_name(data) or '(no protein name)'}"
        + (f" (also: {others})" if others else ""),
        f"Genes: {_gene_names(data) or '?'}",
        f"Organism: {organism}"
        + (f" ({common})" if common else "")
        + (f", taxon {taxon}" if taxon else ""),
        f"Sequence: {length} aa, {mass} Da; read it with uniprot_sequence",
        f"First public: {public} | annotation updated: {updated}",
        f"Features: {features} (uniprot_features) | cross-references: {refs} (uniprot_crossrefs)",
    ]
    keywords = [_string(word.get("name")) for word in _dicts(data.get("keywords"))]
    if any(keywords):
        lines.append(f"Keywords: {', '.join(word for word in keywords if word)}")
    return lines


def _comments(comments: object) -> list[str]:
    """Each comment under the heading of its type, comments of a type together."""
    lines: list[str] = []
    previous = ""
    for comment in _dicts(comments):
        kind = _string(comment.get("commentType"))
        text = " ".join(part for read in _COMMENT_PARTS for part in read(comment) if part)
        if not text:
            continue
        if kind != previous:
            lines.extend(["", f"## {_LABELS.get(kind, kind.capitalize())}"])
            previous = kind
        molecule = _string(comment.get("molecule"))
        lines.append(f"[{molecule}] {text}" if molecule else text)
    return lines


_LABELS = {"PTM": "Post-translational modification", "RNA EDITING": "RNA editing"}


def _texts(value: object) -> list[str]:
    """The ``value`` of each text of ``{"texts": [...]}``; a plain string as it is."""
    if isinstance(value, str):
        return [_string(value)]
    texts = value.get("texts") if isinstance(value, dict) else None
    return [_string(text.get("value")) for text in _dicts(texts)]


def _reaction(comment: dict[str, Any]) -> list[str]:
    reaction = comment.get("reaction")
    if not isinstance(reaction, dict):
        return []
    ec = _string(reaction.get("ecNumber"))
    return [_string(reaction.get("name")) + (f" (EC {ec})" if ec else "")]


def _disease(comment: dict[str, Any]) -> list[str]:
    disease = comment.get("disease")
    if not isinstance(disease, dict):
        return []
    acronym = _string(disease.get("acronym"))
    name = _string(disease.get("diseaseId")) + (f" ({acronym})" if acronym else "")
    description = _string(disease.get("description"))
    return [f"{name}: {description}" if description else name]


def _locations(comment: dict[str, Any]) -> list[str]:
    places = []
    for place in _dicts(comment.get("subcellularLocations")):
        where = _nested(place, "location", "value")
        topology = _nested(place, "topology", "value")
        places.append(where + (f" ({topology})" if topology else ""))
    return ["; ".join(place for place in places if place)]


def _cofactors(comment: dict[str, Any]) -> list[str]:
    return [", ".join(_string(c.get("name")) for c in _dicts(comment.get("cofactors")))]


def _isoforms(comment: dict[str, Any]) -> list[str]:
    isoforms = []
    for isoform in _dicts(comment.get("isoforms")):
        ids = ", ".join(_string(i) for i in isoform.get("isoformIds") or [])
        isoforms.append(f"{_nested(isoform, 'name', 'value')}" + (f" ({ids})" if ids else ""))
    return ["; ".join(isoforms)]


def _interactions(comment: dict[str, Any]) -> list[str]:
    partners = []
    for interaction in _dicts(comment.get("interactions")):
        other = interaction.get("interactantTwo")
        other = other if isinstance(other, dict) else {}
        accession = _string(other.get("uniProtKBAccession"))
        gene = _string(other.get("geneName")) or accession or "?"
        experiments = _string(interaction.get("numberOfExperiments"))
        partners.append(
            gene
            + (f" ({accession})" if accession and accession != gene else "")
            + (f", {experiments} experiments" if experiments else "")
        )
    return ["; ".join(partners)]


def _kinetics(comment: dict[str, Any]) -> list[str]:
    """The biophysicochemical numbers: Michaelis constants and maximum velocities with their
    units, and the absorption maximum in nm."""
    kinetics = comment.get("kineticParameters")
    kinetics = kinetics if isinstance(kinetics, dict) else {}
    parts = [
        f"KM={_number(km.get('constant'))} {_string(km.get('unit'))} for "
        f"{_string(km.get('substrate'))}"
        for km in _dicts(kinetics.get("michaelisConstants"))
    ]
    parts += [
        f"Vmax={_number(v.get('velocity'))} {_string(v.get('unit'))} {_string(v.get('enzyme'))}"
        for v in _dicts(kinetics.get("maximumVelocities"))
    ]
    parts += _texts(kinetics.get("note"))
    absorption = comment.get("absorption")
    if isinstance(absorption, dict) and absorption.get("max") is not None:
        about = "~" if absorption.get("approximate") else ""
        parts.append(f"absorption max {about}{_number(absorption.get('max'))} nm")
        parts += _texts(absorption.get("note"))
    return ["; ".join(part for part in parts if part)]


def _others(comment: dict[str, Any]) -> list[str]:
    """The rest a comment may carry: a note, a sequence caution, a mass, a web resource, edited
    positions, and the biophysicochemical texts."""
    parts = _texts(comment.get("note"))
    if caution := _string(comment.get("sequenceCautionType")):
        parts.append(f"{caution}: {_string(comment.get('sequence'))}")
    if weight := _string(comment.get("molWeight")):
        parts.append(f"{_number(comment.get('molWeight')) or weight} Da")
    if resource := _string(comment.get("resourceName")):
        parts.append(f"{resource}: {_string(comment.get('resourceUrl'))}")
    if positions := [_string(p.get("position")) for p in _dicts(comment.get("positions"))]:
        parts.append(f"positions {', '.join(positions)}")
    for key in ("phDependence", "redoxPotential", "temperatureDependence"):
        parts.extend(_texts(comment.get(key)))
    return parts


_COMMENT_PARTS: tuple[Callable[[dict[str, Any]], list[str]], ...] = (
    _texts,
    _reaction,
    _disease,
    _locations,
    _cofactors,
    _isoforms,
    _interactions,
    _kinetics,
    _others,
)


# --- Lists of an entry -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _Part:
    """One list of an entry (its features, its cross-references), whole or only the items with
    one value of one field, paged through the window.

    Attributes:
        key: The entry's key that holds the list.
        field: The field the filter matches, ignoring case.
        noun: What the items are, for the answer's words.
        count_label: The word over the first page's count of items by ``field``.
        line: The item's line, numbered.
        unmatched: The answer when no item matches, from the entry's accession, the value asked
            for and the count by ``field``.
    """

    key: str
    field: str
    noun: str
    count_label: str
    line: Callable[[int, dict[str, Any]], str]
    unmatched: Callable[[str, str, str], str]

    def answer(
        self, data: dict[str, Any], asked: str, wanted: str, *, offset: int, limit: int
    ) -> ToolResult:
        items = _dicts(data.get(self.key))
        kept = [item for item in items if self._matches(item, wanted)]
        if not kept:
            accession = _string(data.get("primaryAccession")) or asked
            if not items:
                return ToolResult.success(f"{accession} has no {self.noun}.")
            tally = _tally(self._value(item) for item in items)
            return ToolResult.success(self.unmatched(accession, wanted, tally))
        lines = [self.line(number, item) for number, item in enumerate(kept, start=1)]
        window = page_window(lines, offset=offset, limit=limit)
        what = f"{self._value(kept[0])} {self.noun}" if wanted else self.noun.capitalize()
        heading = f"{what} of {_named(data, asked)}:"
        if not wanted and offset == 0:
            heading += f"\n{self.count_label}: {_tally(self._value(item) for item in items)}"
        return window.result(heading=heading)

    def _matches(self, item: dict[str, Any], wanted: str) -> bool:
        return not wanted or self._value(item).casefold() == wanted.casefold()

    def _value(self, item: dict[str, Any]) -> str:
        return _string(item.get(self.field)) or "?"


def _tally(values: Iterable[str]) -> str:
    """``Chain 1, Natural variant 3``: each value and how many times, alphabetically."""
    counts = Counter(values)
    return ", ".join(f"{value} {counts[value]}" for value in sorted(counts, key=str.casefold))


def _feature(number: int, feature: dict[str, Any]) -> str:
    details = [
        _change(feature.get("alternativeSequence")),
        _string(feature.get("description")),
        _nested(feature, "ligand", "name") and f"ligand: {_nested(feature, 'ligand', 'name')}",
    ]
    said = ", ".join(detail for detail in details if detail)
    identifier = _string(feature.get("featureId"))
    where = _location(feature.get("location"))
    return (
        f"{number}. {_string(feature.get('type')) or '?'} {where}"
        + (f": {said}" if said else "")
        + (f" [{identifier}]" if identifier else "")
    )


def _change(value: object) -> str:
    """A variant's or a conflict's change: ``H -> D``."""
    if not isinstance(value, dict):
        return ""
    original = _string(value.get("originalSequence"))
    after = "/".join(_string(item) for item in value.get("alternativeSequences") or [])
    return f"{original or '?'} -> {after or 'missing'}" if original or after else ""


def _location(location: object) -> str:
    """``25-54``, or ``34`` for one residue; ``?`` where UniProt does not know, ``<``/``>``
    where the feature runs past the sequence shown."""
    if not isinstance(location, dict):
        return "?"
    start = _position(location.get("start"), "<")
    end = _position(location.get("end"), ">")
    return start if start == end else f"{start}-{end}"


def _position(value: object, outside: str) -> str:
    if not isinstance(value, dict):
        return _string(value) or "?"
    modifier = _string(value.get("modifier"))
    position = _string(value.get("value"))
    if modifier == "UNKNOWN" or not position:
        return "?"
    return (outside if modifier == "OUTSIDE" else "") + position


def _crossref(number: int, ref: dict[str, Any]) -> str:
    properties = [
        f"{_string(prop.get('key'))}: {_string(prop.get('value'))}"
        for prop in _dicts(ref.get("properties"))
        if _string(prop.get("value")) not in {"", "-"}
    ]
    isoform = _string(ref.get("isoformId"))
    return (
        f"{number}. {_string(ref.get('database')) or '?'}: {_string(ref.get('id')) or '?'}"
        + (f" (isoform {isoform})" if isoform else "")
        + (f" | {', '.join(properties)}" if properties else "")
    )


_FEATURES = _Part(
    key="features",
    field="type",
    noun="features",
    count_label="Types",
    line=_feature,
    unmatched=lambda accession, wanted, tally: (
        f"{accession} has no features of type {wanted!r}; its types: {tally}."
    ),
)
_CROSSREFS = _Part(
    key="uniProtKBCrossReferences",
    field="database",
    noun="cross-references",
    count_label="Databases",
    line=_crossref,
    unmatched=lambda accession, wanted, tally: (
        f"{accession} has no cross-references to {wanted}; its databases: {tally}."
    ),
)


# --- Fields ------------------------------------------------------------------------------------


def _protein_name(item: dict[str, Any]) -> str:
    """The recommended name, or for an unreviewed entry its first submission name."""
    description = item.get("proteinDescription")
    description = description if isinstance(description, dict) else {}
    name = _nested(description, "recommendedName", "fullName", "value")
    if name:
        return name
    submitted = _dicts(description.get("submissionNames"))
    return _nested(submitted[0], "fullName", "value") if submitted else ""


def _alternative_names(item: dict[str, Any]) -> str:
    description = item.get("proteinDescription")
    names = description.get("alternativeNames") if isinstance(description, dict) else None
    return ", ".join(
        name for name in (_nested(other, "fullName", "value") for other in _dicts(names)) if name
    )


def _gene_names(item: dict[str, Any]) -> str:
    names = (_nested(gene, "geneName", "value") for gene in _dicts(item.get("genes")))
    return ", ".join(name for name in names if name)


def _dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _number(value: object) -> str:
    """A number as written, never in scientific notation (``1.5e-05`` is ``0.000015``)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _string(value)
    try:
        return format(Decimal(str(value)), "f")
    except InvalidOperation:  # inf, nan
        return _string(value)


def _integer(value: str | None) -> int | None:
    try:
        return int(value or "")
    except ValueError:
        return None


def _nested(data: dict[str, Any], *keys: str) -> str:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return _string(current)


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
