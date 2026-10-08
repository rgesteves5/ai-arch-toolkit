"""Tests for toolkit/tools/_open_food_facts.py (T07a).

Answers are shaped as the API v2's OpenAPI spec gives them
(https://github.com/openfoodfacts/openfoodfacts-server/blob/main/docs/api/ref/api.yaml): a product
under ``product`` with ``status`` 1, an unknown barcode as a 404 or ``status`` 0, and a search page
with ``count`` (all matches), ``page``, ``page_size``, ``page_count`` (products on this page) and
``skip``. A global rate limit is a 503
(https://openfoodfacts.github.io/openfoodfacts-server/api/#rate-limits).
"""

from __future__ import annotations

import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit import tools
from ai_arch_toolkit.toolkit.tools._open_food_facts import (
    open_food_facts_compare,
    open_food_facts_product,
    open_food_facts_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_ALLERGENS = [f"en:allergen-{n}" for n in range(1, 13)]
_PRODUCT = {
    "code": "3017620422003",
    "product_name": "Nutella",
    "generic_name": "Hazelnut cocoa spread",
    "brands": "Ferrero",
    "quantity": "400 g",
    "categories_tags": [f"en:category-{n}" for n in range(1, 11)],
    "labels_tags": ["en:vegetarian", "en:no-gluten"],
    "countries_tags": [f"en:country-{n}" for n in range(1, 11)],
    "ingredients_text": "Sugar, palm oil, hazelnuts, cocoa.",
    "allergens_tags": _ALLERGENS,
    "traces_tags": [f"en:trace-{n}" for n in range(1, 12)],
    "additives_tags": [f"en:e{n}" for n in range(300, 312)],
    "nutriscore_grade": "e",
    "nova_group": 4,
    "ecoscore_grade": "d",
    "nutriments": {
        "energy-kcal_100g": 539,
        "energy-kcal_unit": "kcal",
        "fat_100g": 30.9,
        "saturated-fat_100g": 10.6,
        "sugars_100g": 56.3,
        "salt_100g": 0.107,
        "sodium_100g": 0.00001,
    },
    "image_front_url": "https://images.openfoodfacts.org/front.jpg",
}


def _found(product: dict[str, Any] = _PRODUCT) -> dict[str, Any]:
    return {
        "code": product["code"],
        "status": 1,
        "status_verbose": "product found",
        "product": product,
    }


def _missing(barcode: str = "12345678") -> dict[str, Any]:
    return {"code": barcode, "status": 0, "status_verbose": "product not found"}


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


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


class TestProduct:
    @patch(HTTP_OPEN)
    def test_the_record_comes_whole_with_labels_next_to_the_scores(self, mock_urlopen):
        mock_urlopen.return_value = respond(_found())

        text = open_food_facts_product("3017 6204 22003")

        lines = text.splitlines()
        assert lines[:3] == [
            "Open Food Facts product 3017620422003: Nutella",
            "   brand: Ferrero | quantity: 400 g",
            "   Nutri-Score: E (lowest nutritional quality) | NOVA: 4 (ultra-processed foods) | "
            "Eco-Score: D",
        ]
        assert "   Generic name: Hazelnut cocoa spread" in lines
        assert (
            "   Nutrients per 100 g: energy 539 kcal | fat 30.9 g | saturated fat 10.6 g | "
            "sugars 56.3 g | salt 0.107 g | sodium 0.00001 g"
        ) in lines
        assert "   Ingredients: Sugar, palm oil, hazelnuts, cocoa." in lines
        assert "https://world.openfoodfacts.org/product/3017620422003" in text

    @pytest.mark.parametrize(
        ("label", "count"),
        [
            ("Allergens", 12),
            ("Traces", 11),
            ("Additives", 12),
            ("Categories", 10),
            ("Countries", 10),
        ],
    )
    @patch(HTTP_OPEN)
    def test_every_list_comes_whole(self, mock_urlopen: MagicMock, label: str, count: int):
        mock_urlopen.return_value = respond(_found())

        line = next(
            row
            for row in open_food_facts_product("3017620422003").splitlines()
            if row.strip().startswith(f"{label}:")
        )

        assert len(line.split(": ", 1)[1].split(", ")) == count

    @patch(HTTP_OPEN)
    def test_the_nutrition_tool_is_gone_into_the_product(self, mock_urlopen: MagicMock):
        assert not hasattr(tools, "open_food_facts_nutrition")
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        "answer",
        [
            respond(_missing()),
            http_error(404, "Not Found", body=b'{"code": "12345678", "status": 0}'),
        ],
    )
    @patch(HTTP_OPEN)
    def test_an_unknown_barcode_is_not_found(self, mock_urlopen: MagicMock, answer: Any):
        if isinstance(answer, urllib.error.HTTPError):
            mock_urlopen.side_effect = answer
        else:
            mock_urlopen.return_value = answer

        failure = _failure(open_food_facts_product, "12345678")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "Open Food Facts has no product with barcode 12345678; find products with "
            "open_food_facts_search"
        )

    @patch(HTTP_OPEN)
    def test_invalid_barcode_does_not_call_api(self, mock_urlopen: MagicMock):
        assert "invalid barcode 'abc'" in _invalid(open_food_facts_product, "abc")
        assert "invalid barcode '123'" in _invalid(open_food_facts_product, "123")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_503_is_the_global_rate_limit(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        failure = _failure(open_food_facts_product, "12345678")

        assert (failure.error.type, failure.error.retryable) == ("rate_limited", True)
        assert "HTTP 503" in failure.error.message


class TestCompare:
    @patch(HTTP_OPEN)
    def test_products_compare_on_one_line_each_with_every_allergen(self, mock_urlopen):
        second = {
            **_PRODUCT,
            "code": "3168930010265",
            "product_name": "Cereal",
            "brands": "Quaker",
            "nutriscore_grade": "b",
            "nova_group": 3,
            "nutriments": {"energy-kcal_100g": 462, "sugars_100g": 12, "salt_100g": 0},
            "allergens_tags": ["en:gluten"],
        }
        mock_urlopen.side_effect = [respond(_found()), respond(_found(second))]

        text = open_food_facts_compare("3017620422003,3168930010265")

        lines = text.splitlines()
        assert lines[0] == "Open Food Facts comparison (nutrients per 100 g):"
        assert lines[1].startswith(
            "1. Nutella | barcode 3017620422003 | brand: Ferrero | Nutri-Score: E | NOVA: 4 | "
            "energy 539 kcal | sugars 56.3 g | saturated fat 10.6 g | salt 0.107 g"
        )
        assert lines[1].endswith("allergens: " + ", ".join(f"allergen {n}" for n in range(1, 13)))
        assert lines[2] == (
            "2. Cereal | barcode 3168930010265 | brand: Quaker | Nutri-Score: B | NOVA: 3 | "
            "energy 462 kcal | sugars 12 g | salt 0 g | allergens: gluten"
        )
        assert mock_urlopen.call_count == 2

    @patch(HTTP_OPEN)
    def test_a_missing_product_is_named_beside_the_found_ones(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = [
            respond(_found()),
            http_error(404, "Not Found", body=b'{"status": 0}'),
        ]

        text = open_food_facts_compare("3017620422003,00000000")

        assert "1. Nutella | barcode 3017620422003" in text
        assert text.endswith("Missing products: 00000000")

    @patch(HTTP_OPEN)
    def test_none_found_is_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = [respond(_missing()), respond(_missing("87654321"))]

        failure = _failure(open_food_facts_compare, "12345678,87654321")

        assert failure.error.type == "not_found"
        assert "none of the products 12345678, 87654321" in failure.error.message

    @patch(HTTP_OPEN)
    def test_another_failure_stops_the_comparison(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = [respond(_found()), TimeoutError()]

        failure = _failure(open_food_facts_compare, "3017620422003,12345678")

        assert (failure.error.type, failure.error.retryable) == ("upstream", True)

    @patch(HTTP_OPEN)
    def test_invalid_compare_options_do_not_call_api(self, mock_urlopen: MagicMock):
        assert "invalid barcode" in _invalid(open_food_facts_compare, "abc")
        assert "at most 5" in _invalid(open_food_facts_compare, "1234,1235,1236,1237,1238,1239")
        mock_urlopen.assert_not_called()


class TestSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_count_and_the_next_page(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            {
                "count": 10,
                "page": 2,
                "page_count": 1,
                "page_size": 1,
                "skip": 1,
                "products": [_PRODUCT],
            }
        )

        result = open_food_facts_search(
            product_name="nutella",
            brand="ferrero",
            category="spreads",
            country="france",
            label="vegetarian",
            max_results=1,
            page=2,
        )

        assert isinstance(result, ToolResult)
        assert result.value.splitlines() == [
            "Open Food Facts products for product name 'nutella', brand 'ferrero', category "
            "'spreads', country 'france', label 'vegetarian' (read one with "
            "open_food_facts_product):",
            "2. Nutella | barcode 3017620422003 | brand: Ferrero | quantity: 400 g | "
            "Nutri-Score: E | NOVA: 4 | Eco-Score: D",
            "[results 2-2 of 10 | next: page=3]",
        ]
        params = _params(mock_urlopen)
        assert params["product_name"] == ["nutella"]
        assert params["brands_tags"] == ["ferrero"]
        assert params["categories_tags_en"] == ["spreads"]
        assert params["countries_tags_en"] == ["france"]
        assert params["labels_tags_en"] == ["vegetarian"]
        assert params["page_size"] == ["1"]
        assert params["page"] == ["2"]
        assert "ingredients_text" not in params["fields"][0]

    @patch(HTTP_OPEN)
    def test_the_last_page_ends(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond(
            {
                "count": 6,
                "page": 2,
                "page_count": 1,
                "page_size": 5,
                "skip": 5,
                "products": [_PRODUCT],
            }
        )

        text = _text(open_food_facts_search(brand="ferrero", max_results=5, page=2))

        assert text.endswith("[results 6-6 of 6 | end]")

    @patch(HTTP_OPEN)
    def test_no_products_say_so_with_the_search(self, mock_urlopen: MagicMock):
        mock_urlopen.return_value = respond({"count": 0, "products": []})

        assert _text(open_food_facts_search(product_name="zzqqxx")) == (
            "No Open Food Facts products match product name 'zzqqxx'."
        )

    @patch(HTTP_OPEN)
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen: MagicMock):
        assert "provide product_name" in _invalid(open_food_facts_search)
        assert "invalid filter value for product_name" in _invalid(
            open_food_facts_search, product_name="bad<>"
        )
        assert "invalid filter value for brand;" in _invalid(open_food_facts_search, brand="a<b")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_the_rate_limit_and_a_parse_failure(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")
        unavailable = _failure(open_food_facts_search, product_name="test")
        assert (unavailable.error.type, unavailable.error.retryable) == ("rate_limited", True)

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = respond("not json")
        not_json = _failure(open_food_facts_search, product_name="test")
        assert not_json.error.type == "upstream"
        assert "could not parse" in str(not_json)

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen: MagicMock):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(open_food_facts_search, product_name="test")

        assert failure.error.type == "upstream"
        assert "endpoint not found (HTTP 404)" in failure.error.message
