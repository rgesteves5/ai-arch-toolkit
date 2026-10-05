"""Tests for toolkit/tools/_open_food_facts.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._open_food_facts import (
    open_food_facts_compare,
    open_food_facts_nutrition,
    open_food_facts_product,
    open_food_facts_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_PRODUCT = {
    "code": "3017620422003",
    "product_name": "Nutella",
    "generic_name": "Hazelnut cocoa spread",
    "brands": "Ferrero",
    "quantity": "400 g",
    "categories_tags": ["en:spreads", "en:sweet-spreads"],
    "labels_tags": ["en:vegetarian", "en:no-gluten"],
    "countries_tags": ["en:france", "en:portugal"],
    "ingredients_text": "Sugar, palm oil, hazelnuts, cocoa.",
    "allergens_tags": ["en:milk", "en:nuts"],
    "traces_tags": ["en:soybeans"],
    "additives_tags": ["en:e322"],
    "nutriscore_grade": "e",
    "nova_group": 4,
    "ecoscore_grade": "d",
    "nutriments": {
        "energy-kcal_100g": 539,
        "fat_100g": 30.9,
        "saturated-fat_100g": 10.6,
        "sugars_100g": 56.3,
        "salt_100g": 0.107,
    },
    "image_front_url": "https://images.openfoodfacts.org/front.jpg",
}


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _invalid(fn, *args, **kwargs) -> str:
    failure = _failure(fn, *args, **kwargs)
    assert failure.error.type == "validation_error"
    return failure.error.message


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestOpenFoodFactsProduct:
    @patch(HTTP_OPEN)
    def test_returns_product_by_barcode(self, mock_urlopen):
        mock_urlopen.return_value = respond({"status": 1, "product": _PRODUCT})

        result = open_food_facts_product("3017 6204 22003")

        assert result.startswith("Open Food Facts product 3017620422003:")
        assert "Nutella" in result
        assert "barcode: 3017620422003 | brand: Ferrero | quantity: 400 g" in result
        assert "Nutri-Score: E | NOVA: 4 | Eco-Score: D" in result
        assert "Categories: spreads, sweet spreads" in result
        assert "Nutrients per 100g: energy kcal: 539" in result
        assert "Ingredients: Sugar, palm oil, hazelnuts, cocoa." in result
        assert "Allergens: milk, nuts" in result
        assert "https://world.openfoodfacts.org/product/3017620422003" in result

        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        assert urlparse(request.full_url).path == "/api/v2/product/3017620422003.json"
        assert "product_name" in _called_params(mock_urlopen)["fields"][0]

    @patch(HTTP_OPEN)
    def test_product_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({"status": 0})

        failure = _failure(open_food_facts_product, "12345678")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "Open Food Facts has no product with barcode 12345678; find products with "
            "open_food_facts_search"
        )

    @patch(HTTP_OPEN)
    def test_invalid_barcode_does_not_call_api(self, mock_urlopen):
        assert "invalid barcode 'abc'" in _invalid(open_food_facts_product, "abc")
        assert "invalid barcode '123'" in _invalid(open_food_facts_product, "123")
        mock_urlopen.assert_not_called()


class TestOpenFoodFactsNutrition:
    @patch(HTTP_OPEN)
    def test_returns_nutrition_summary(self, mock_urlopen):
        mock_urlopen.return_value = respond({"status": 1, "product": _PRODUCT})

        result = open_food_facts_nutrition("3017620422003")

        assert result.startswith("Open Food Facts nutrition 3017620422003:")
        assert "Nutri-Score: E (lowest nutritional quality)" in result
        assert "NOVA: 4 (ultra-processed foods)" in result
        assert "Nutrients per 100g: energy kcal: 539" in result
        assert "Allergens: milk, nuts" in result
        assert "Ingredients: Sugar, palm oil, hazelnuts, cocoa." in result

    @patch(HTTP_OPEN)
    def test_invalid_nutrition_barcode_does_not_call_api(self, mock_urlopen):
        assert "invalid barcode" in _invalid(open_food_facts_nutrition, "abc")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_nutrition_of_an_unknown_product_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({"status": 0})

        assert _failure(open_food_facts_nutrition, "12345678").error.type == "not_found"


class TestOpenFoodFactsCompare:
    @patch(HTTP_OPEN)
    def test_compares_products(self, mock_urlopen):
        second = {
            **_PRODUCT,
            "code": "3168930010265",
            "product_name": "Cereal",
            "brands": "Quaker",
            "nutriscore_grade": "b",
            "nutriments": {"energy-kcal_100g": 462, "sugars_100g": 12, "salt_100g": 0},
        }
        mock_urlopen.side_effect = [
            respond({"status": 1, "product": _PRODUCT}),
            respond({"status": 1, "product": second}),
        ]

        result = open_food_facts_compare("3017620422003,3168930010265")

        assert "Open Food Facts comparison:" in result
        assert "1. Nutella | barcode: 3017620422003" in result
        assert "sugars/100g: 56.3" in result
        assert "2. Cereal | barcode: 3168930010265" in result
        assert "Nutri-Score: B" in result
        assert mock_urlopen.call_count == 2

    @patch(HTTP_OPEN)
    def test_a_product_api_v2_answers_with_a_404_is_listed_as_missing(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond({"status": 1, "product": _PRODUCT}),
            http_error(404, "Not Found", body=b'{"status": 0}'),
        ]

        result = open_food_facts_compare("3017620422003,00000000")

        assert "1. Nutella | barcode: 3017620422003" in result
        assert "00000000" in result

    @patch(HTTP_OPEN)
    def test_a_404_for_one_product_is_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found", body=b'{"status": 0}')

        failure = _failure(open_food_facts_product, "12345678")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "Open Food Facts has no product with barcode 12345678; find products with "
            "open_food_facts_search"
        )

    @patch(HTTP_OPEN)
    def test_a_503_on_a_product_is_the_rate_limit(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        failure = _failure(open_food_facts_product, "12345678")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable

    @patch(HTTP_OPEN)
    def test_invalid_compare_options_do_not_call_api(self, mock_urlopen):
        assert "invalid barcode" in _invalid(open_food_facts_compare, "abc")
        assert "at most 5" in _invalid(open_food_facts_compare, "1234,1235,1236,1237,1238,1239")
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_missing_product_is_named_beside_the_found_ones(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond({"status": 1, "product": _PRODUCT}),
            respond({"status": 0}),
        ]

        result = open_food_facts_compare("3017620422003,12345678")

        assert "1. Nutella | barcode: 3017620422003" in result
        assert result.endswith("Missing products: 12345678")

    @patch(HTTP_OPEN)
    def test_none_found_is_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = [respond({"status": 0}), respond({"status": 0})]

        failure = _failure(open_food_facts_compare, "12345678,87654321")

        assert failure.error.type == "not_found"
        assert "none of the products 12345678, 87654321" in failure.error.message

    @patch(HTTP_OPEN)
    def test_another_failure_stops_the_comparison(self, mock_urlopen):
        mock_urlopen.side_effect = [respond({"status": 1, "product": _PRODUCT}), TimeoutError()]

        failure = _failure(open_food_facts_compare, "3017620422003,12345678")

        assert failure.error.type == "upstream"
        assert failure.error.retryable


class TestOpenFoodFactsSearch:
    @patch(HTTP_OPEN)
    def test_returns_search_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"count": 10, "page": 2, "page_count": 1, "page_size": 1, "products": [_PRODUCT]}
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

        assert "Open Food Facts products (page 2, returned 1, page_count 1, total 10)" in result
        assert "Nutella" in result
        assert "Ingredients:" not in result

        params = _called_params(mock_urlopen)
        assert params["product_name"] == ["nutella"]
        assert params["brands_tags"] == ["ferrero"]
        assert params["categories_tags_en"] == ["spreads"]
        assert params["countries_tags_en"] == ["france"]
        assert params["labels_tags_en"] == ["vegetarian"]
        assert params["page_size"] == ["1"]
        assert params["page"] == ["2"]

    @patch(HTTP_OPEN)
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "provide product_name" in _invalid(open_food_facts_search)
        assert "page must" in _invalid(open_food_facts_search, product_name="test", page=0)
        assert "invalid filter value for product_name" in _invalid(
            open_food_facts_search, product_name="bad<>"
        )
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_products_is_an_answer(self, mock_urlopen):
        mock_urlopen.return_value = respond({"count": 0, "products": []})

        assert (
            open_food_facts_search(product_name="zzqqxx") == "No Open Food Facts products found."
        )

    @patch(HTTP_OPEN)
    def test_rate_limit_and_parse_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://world.openfoodfacts.org/api/v2/search",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

        unavailable = _failure(open_food_facts_search, product_name="test")
        assert unavailable.error.type == "rate_limited"
        assert unavailable.error.retryable
        assert unavailable.error.message == (
            "Open Food Facts global rate limit reached (HTTP 503); try again in a minute."
        )

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = respond("not json")
        not_json = _failure(open_food_facts_search, product_name="test")
        assert not_json.error.type == "upstream"
        assert "could not parse" in str(not_json)

    @patch(HTTP_OPEN)
    def test_another_server_error_is_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(502, "Bad Gateway")

        failure = _failure(open_food_facts_search, product_name="test")

        assert failure.error.type == "upstream"
        assert failure.error.retryable

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(open_food_facts_search, product_name="test")

        assert failure.error.type == "upstream"
        assert "endpoint not found (HTTP 404)" in failure.error.message
