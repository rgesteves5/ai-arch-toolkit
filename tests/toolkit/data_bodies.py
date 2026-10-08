"""Answers of the data and news sources (Eurostat, the World Bank, WHO GHO, GDELT, Wikidata, Hacker
News), shaped as each one's documentation shows them, for the contract cases (T08a)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def hn_ids(count: int) -> list[int]:
    """Hacker News' top stories list (up to 500 ids)."""
    return list(range(1, count + 1))


def hn_story(number: int) -> dict[str, Any]:
    return {"id": number, "type": "story", "title": f"Story {number}", "score": 1, "by": "a"}


def gdelt_articles(count: int) -> dict[str, Any]:
    """A GDELT ``artlist`` answer with ``count`` articles."""
    return {
        "articles": [
            {"title": f"Story {number}", "url": f"https://news.example/{number}"}
            for number in range(1, count + 1)
        ]
    }


def gdelt_timeline(count: int) -> dict[str, Any]:
    """A GDELT ``timelinevol`` answer with ``count`` daily points."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    points = [
        {"date": (start + timedelta(days=day)).strftime("%Y%m%dT%H%M%SZ"), "value": day / 100}
        for day in range(count)
    ]
    return {"timeline": [{"series": "Volume Intensity", "data": points}]}


def who_values(count: int, start: int = 1) -> dict[str, Any]:
    """An OData answer of the WHO GHO API: indicators, or an indicator's rows."""
    return {
        "value": [
            {
                "IndicatorCode": f"CODE_{number}",
                "IndicatorName": f"Indicator {number}",
                "SpatialDimType": "COUNTRY",
                "SpatialDim": "PRT",
                "TimeDim": 2000 + number,
                "NumericValue": number,
            }
            for number in range(start, start + count)
        ]
    }


def wikidata_search(count: int, start: int, more: int) -> dict[str, Any]:
    """A ``wbsearchentities`` answer, with where the search goes on."""
    return {
        "search": [
            {"id": f"Q{number}", "label": f"Item {number}"}
            for number in range(start, start + count)
        ],
        "search-continue": more,
        "success": 1,
    }


def wikidata_entity(statements: int) -> dict[str, Any]:
    """A Special:EntityData answer with one string statement per property."""
    claims = {
        f"P{number}": [
            {
                "mainsnak": {
                    "snaktype": "value",
                    "property": f"P{number}",
                    "datavalue": {"type": "string", "value": f"value {number}"},
                },
                "rank": "normal",
            }
        ]
        for number in range(1, statements + 1)
    }
    return {"entities": {"Q1": {"id": "Q1", "claims": claims}}}


def wikidata_labels() -> dict[str, Any]:
    """A ``wbgetentities`` answer with no labels."""
    return {"entities": {}, "success": 1}


def sparql_rows(count: int) -> dict[str, Any]:
    """A SPARQL JSON results answer with ``count`` rows."""
    return {
        "head": {"vars": ["item"]},
        "results": {
            "bindings": [
                {"item": {"type": "uri", "value": f"http://www.wikidata.org/entity/Q{number}"}}
                for number in range(1, count + 1)
            ]
        },
    }


def eurostat_catalogue(count: int) -> dict[str, Any]:
    """The dataflow stubs (``detail=allstubs``)."""
    return {
        "link": {
            "item": [
                {"label": f"Population table {number}", "extension": {"id": f"POP{number:03d}"}}
                for number in range(1, count + 1)
            ]
        }
    }


def eurostat_dataset(geos: int, years: int) -> dict[str, Any]:
    """A JSON-stat 2.0 answer: ``geos`` places by ``years`` periods, every cell a value."""
    places = {f"G{number:02d}": f"Region {number}" for number in range(geos)}
    periods = [str(2000 + year) for year in range(years)]
    return {
        "label": "Population on 1 January",
        "id": ["geo", "time"],
        "size": [geos, years],
        "value": {str(cell): cell for cell in range(geos * years)},
        "dimension": {
            "geo": {
                "label": "Geopolitical entity",
                "category": {
                    "index": {code: index for index, code in enumerate(places)},
                    "label": places,
                },
            },
            "time": {
                "label": "Time",
                "category": {"index": {year: index for index, year in enumerate(periods)}},
            },
        },
    }


def world_bank_page(
    items: list[dict[str, Any]], *, page: int, pages: int, total: int
) -> list[Any]:
    """A World Bank answer: ``[metadata, items]``."""
    return [{"page": page, "pages": pages, "per_page": str(len(items)), "total": total}, items]


def world_bank_records(count: int, start: int) -> list[dict[str, Any]]:
    """Records any World Bank list reads: an ID, a name, a date and a value."""
    return [
        {
            "id": f"ID{number}",
            "value": f"Record {number}",
            "name": f"Record {number}",
            "date": str(2000 + number),
            "country": {"id": "PT", "value": "Portugal"},
            "countryiso3code": "PRT",
            "indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
        }
        for number in range(start, start + count)
    ]


# What the API answers, with HTTP 200, for a code it does not know (api.worldbank.org,
# 2026-09-29): error 120.
WORLD_BANK_INVALID_VALUE = [
    {
        "message": [
            {
                "id": "120",
                "key": "Invalid value",
                "value": "The provided parameter value is not valid",
            }
        ]
    }
]
