"""Tests for toolkit/tools/_open_food_facts.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._open_food_facts import (
    open_food_facts_compare,
    open_food_facts_nutrition,
    open_food_facts_product,
    open_food_facts_search,
)

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


class TestOpenFoodFactsProduct:
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_returns_product_by_barcode(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen({"status": 1, "product": _PRODUCT})

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

    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_product_not_found(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen({"status": 0})

        result = open_food_facts_product("12345678")

        assert "not found" in result.lower()

    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_invalid_barcode_does_not_call_api(self, mock_urlopen):
        assert "invalid barcode" in open_food_facts_product("abc")
        assert "invalid barcode" in open_food_facts_product("123")
        mock_urlopen.assert_not_called()


class TestOpenFoodFactsNutrition:
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_returns_nutrition_summary(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen({"status": 1, "product": _PRODUCT})

        result = open_food_facts_nutrition("3017620422003")

        assert result.startswith("Open Food Facts nutrition 3017620422003:")
        assert "Nutri-Score: E (lowest nutritional quality)" in result
        assert "NOVA: 4 (ultra-processed foods)" in result
        assert "Nutrients per 100g: energy kcal: 539" in result
        assert "Allergens: milk, nuts" in result
        assert "Ingredients: Sugar, palm oil, hazelnuts, cocoa." in result

    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_invalid_nutrition_barcode_does_not_call_api(self, mock_urlopen):
        assert "invalid barcode" in open_food_facts_nutrition("abc")
        mock_urlopen.assert_not_called()


class TestOpenFoodFactsCompare:
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_compares_products(self, mock_urlopen, _mock_throttle):
        second = {
            **_PRODUCT,
            "code": "3168930010265",
            "product_name": "Cereal",
            "brands": "Quaker",
            "nutriscore_grade": "b",
            "nutriments": {"energy-kcal_100g": 462, "sugars_100g": 12, "salt_100g": 0},
        }
        mock_urlopen.side_effect = [
            _mock_urlopen({"status": 1, "product": _PRODUCT}),
            _mock_urlopen({"status": 1, "product": second}),
        ]

        result = open_food_facts_compare("3017620422003,3168930010265")

        assert "Open Food Facts comparison:" in result
        assert "1. Nutella | barcode: 3017620422003" in result
        assert "sugars/100g: 56.3" in result
        assert "2. Cereal | barcode: 3168930010265" in result
        assert "Nutri-Score: B" in result
        assert mock_urlopen.call_count == 2

    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_invalid_compare_options_do_not_call_api(self, mock_urlopen):
        assert "invalid barcode" in open_food_facts_compare("abc")
        assert "at most 5" in open_food_facts_compare("1234,1235,1236,1237,1238,1239")
        mock_urlopen.assert_not_called()


class TestOpenFoodFactsSearch:
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_returns_search_results(self, mock_urlopen, _mock_throttle):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "provide product_name" in open_food_facts_search()
        assert "page must" in open_food_facts_search(product_name="test", page=0)
        assert "invalid filter value" in open_food_facts_search(product_name="bad<>")
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts._throttle")
    @patch("ai_arch_toolkit.toolkit.tools._open_food_facts.urllib.request.urlopen")
    def test_rate_limit_and_parse_failure(self, mock_urlopen, _mock_throttle):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://world.openfoodfacts.org/api/v2/search",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=None,
        )

        assert "global rate limit" in open_food_facts_search(product_name="test")

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _mock_urlopen("not json")
        assert "could not parse" in open_food_facts_search(product_name="test")
