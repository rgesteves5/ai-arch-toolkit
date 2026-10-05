"""Open Food Facts tools — public packaged food search and barcode lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply


def _rate_limit(reply: Reply) -> ToolFailure | None:
    """Open Food Facts answers a caller over its global rate limit with a 503, not a 429
    (rate limits: https://openfoodfacts.github.io/openfoodfacts-server/api/#rate-limits)."""
    if reply.status != 503:
        return None
    return ToolFailure(
        "rate_limited",
        "Open Food Facts global rate limit reached (HTTP 503); try again in a minute.",
        retryable=True,
    )


# Product reads and searches are spaced apart on separate clocks, each at its own interval.
_PRODUCTS = Api(
    base="https://world.openfoodfacts.org/api/v2",
    name="Open Food Facts",
    timeout_s=15,
    min_interval_s=4.1,
    error_reader=_rate_limit,
)
_SEARCH = Api(
    base="https://world.openfoodfacts.org/api/v2",
    name="Open Food Facts",
    timeout_s=15,
    min_interval_s=6.1,
    error_reader=_rate_limit,
)
_MAX_RESULTS_LIMIT = 20
_BARCODE_RE = re.compile(r"^\d{4,32}$")
_TEXT_FILTER_RE = re.compile(r"^[\w\s,.'&()/%+-]{1,120}$", re.UNICODE)
_FIELDS = (
    "code",
    "product_name",
    "generic_name",
    "brands",
    "quantity",
    "categories_tags",
    "labels_tags",
    "countries_tags",
    "ingredients_text",
    "allergens_tags",
    "traces_tags",
    "additives_tags",
    "nutriscore_grade",
    "nova_group",
    "ecoscore_grade",
    "nutriments",
    "image_front_url",
)
_NUTRIENTS = (
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _OpenFoodFactsProduct:
    """Normalized Open Food Facts product metadata."""

    code: str
    name: str
    generic_name: str
    brands: str
    quantity: str
    categories: tuple[str, ...]
    labels: tuple[str, ...]
    countries: tuple[str, ...]
    ingredients: str
    allergens: tuple[str, ...]
    traces: tuple[str, ...]
    additives: tuple[str, ...]
    nutriscore: str
    nova_group: int | None
    ecoscore: str
    nutrients: tuple[str, ...]
    image_url: str


@tool(capability="network")
def open_food_facts_product(barcode: str) -> str:
    """Fetch packaged food metadata by barcode from Open Food Facts.

    Args:
        barcode: Product barcode / GTIN.

    Raises:
        ToolFailure: validation_error when ``barcode`` is not a barcode; not_found when Open
            Food Facts has no product with it.
    """
    normalized = _barcode(barcode)
    product = _fetch_product(normalized)
    return f"Open Food Facts product {normalized}:\n" + _format_products(
        [product],
        include_index=False,
        include_details=True,
    )


@tool(capability="network")
def open_food_facts_search(
    product_name: str = "",
    brand: str = "",
    category: str = "",
    country: str = "",
    label: str = "",
    max_results: int = 5,
    page: int = 1,
) -> str:
    """Search packaged foods in Open Food Facts using structured filters.

    Args:
        product_name: Optional product name filter.
        brand: Optional brand filter.
        category: Optional category tag/name filter, e.g. "breakfast cereals".
        country: Optional country tag/name filter, e.g. "france".
        label: Optional label tag/name filter, e.g. "organic".
        max_results: Number of products to return (1-20). Defaults to 5.
        page: One-based result page. Defaults to 1.

    Raises:
        ToolFailure: validation_error when ``page`` is below 1, no filter is given, or a filter
            has characters it cannot take.
    """
    if page < 1:
        raise ToolFailure(
            "validation_error", f"page must be greater than or equal to 1, not {page}"
        )
    filters = {
        "product_name": product_name.strip(),
        "brands_tags": brand.strip(),
        "categories_tags_en": category.strip(),
        "countries_tags_en": country.strip(),
        "labels_tags_en": label.strip(),
    }
    if not any(filters.values()):
        raise ToolFailure(
            "validation_error",
            "no filter given; provide product_name, brand, category, country, or label",
        )
    invalid = [name for name, value in filters.items() if value and not _valid_filter(value)]
    if invalid:
        raise ToolFailure(
            "validation_error",
            f"invalid filter value for {', '.join(invalid)}; use 1-120 letters, digits, spaces "
            "and basic punctuation (,.'&()/%+-)",
        )

    params = {
        "fields": ",".join(_FIELDS),
        "page_size": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "page": str(page),
    }
    params.update({key: value for key, value in filters.items() if value})

    return _SEARCH.get_json("search", params=params, parse=lambda data: _search_text(data, page))


@tool(capability="network")
def open_food_facts_nutrition(barcode: str) -> str:
    """Fetch a nutrition-focused Open Food Facts summary by barcode.

    Args:
        barcode: Product barcode / GTIN.

    Raises:
        ToolFailure: validation_error when ``barcode`` is not a barcode; not_found when Open
            Food Facts has no product with it.
    """
    normalized = _barcode(barcode)
    product = _fetch_product(normalized)
    return f"Open Food Facts nutrition {normalized}:\n" + _format_nutrition(product)


@tool(capability="network")
def open_food_facts_compare(barcodes: str) -> str:
    """Compare nutrition signals for multiple Open Food Facts products.

    Args:
        barcodes: Comma-separated product barcodes / GTINs. At most 5 products.

    Raises:
        ToolFailure: validation_error when the list is empty, longer than 5, or holds something
            that is not a barcode; not_found when Open Food Facts has none of the products (a
            missing one among found ones is named in the answer).
    """
    parsed = _parse_barcode_list(barcodes)

    products: list[_OpenFoodFactsProduct] = []
    missing: list[str] = []
    for barcode in parsed:
        try:
            products.append(_fetch_product(barcode))
        except ToolFailure as e:  # a missing product is listed; any other failure stops
            if e.error.type != "not_found":
                raise
            missing.append(barcode)

    if not products:
        raise ToolFailure(
            "not_found",
            f"Open Food Facts has none of the products {', '.join(missing)}; check the barcodes "
            "or find products with open_food_facts_search",
        )
    lines = ["Open Food Facts comparison:"]
    for index, product in enumerate(products, start=1):
        lines.append(_format_compare_row(index, product))
    if missing:
        lines.append("Missing products: " + ", ".join(missing))
    return "\n".join(lines)


def _fetch_product(barcode: str) -> _OpenFoodFactsProduct:
    """The product with ``barcode``.

    Raises:
        ToolFailure: not_found when Open Food Facts has no product with ``barcode``.
    """
    missing = (
        f"Open Food Facts has no product with barcode {barcode}; find products with "
        "open_food_facts_search"
    )
    # API v2 answers an unknown barcode with a 404 as well as with status 0.
    product = _PRODUCTS.get_json(
        "product",
        f"{barcode}.json",
        params={"fields": ",".join(_FIELDS)},
        parse=_product,
        missing=missing,
    )
    if product is None:
        raise ToolFailure("not_found", missing)
    return product


def _product(data: dict[str, Any]) -> _OpenFoodFactsProduct | None:
    if data.get("status") == 0:
        return None
    product_data = data.get("product")
    return _parse_product(product_data) if isinstance(product_data, dict) else None


def _search_text(data: dict[str, Any], page: int) -> str:
    products = [
        product
        for item in data.get("products", [])
        if isinstance(item, dict)
        if (product := _parse_product(item))
    ]
    if not products:
        return "No Open Food Facts products found."
    total = _string(data.get("count")) or "?"
    page_count = _string(data.get("page_count")) or str(len(products))
    return (
        f"Open Food Facts products (page {page}, returned {len(products)}, "
        f"page_count {page_count}, total {total}):\n"
        + _format_products(products, include_details=False)
    )


def _parse_product(data: dict[str, Any]) -> _OpenFoodFactsProduct | None:
    code = _string(data.get("code"))
    name = _string(data.get("product_name"))
    if not code and not name:
        return None
    return _OpenFoodFactsProduct(
        code=code,
        name=name or "(unnamed)",
        generic_name=_string(data.get("generic_name")),
        brands=_string(data.get("brands")),
        quantity=_string(data.get("quantity")),
        categories=_tag_tuple(data.get("categories_tags")),
        labels=_tag_tuple(data.get("labels_tags")),
        countries=_tag_tuple(data.get("countries_tags")),
        ingredients=_string(data.get("ingredients_text")),
        allergens=_tag_tuple(data.get("allergens_tags")),
        traces=_tag_tuple(data.get("traces_tags")),
        additives=_tag_tuple(data.get("additives_tags")),
        nutriscore=_string(data.get("nutriscore_grade")).upper(),
        nova_group=_int_or_none(data.get("nova_group")),
        ecoscore=_string(data.get("ecoscore_grade")).upper(),
        nutrients=_nutrients(data.get("nutriments")),
        image_url=_string(data.get("image_front_url")),
    )


def _format_products(
    products: list[_OpenFoodFactsProduct],
    *,
    include_index: bool = True,
    include_details: bool = False,
) -> str:
    blocks: list[str] = []
    for index, product in enumerate(products, start=1):
        title = f"{index}. {product.name}" if include_index else product.name
        lines = [title]
        meta = []
        if product.code:
            meta.append(f"barcode: {product.code}")
        if product.brands:
            meta.append(f"brand: {product.brands}")
        if product.quantity:
            meta.append(f"quantity: {product.quantity}")
        if product.nutriscore:
            meta.append(f"Nutri-Score: {product.nutriscore}")
        if product.nova_group is not None:
            meta.append(f"NOVA: {product.nova_group}")
        if product.ecoscore:
            meta.append(f"Eco-Score: {product.ecoscore}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if product.generic_name:
            lines.append(f"   Generic name: {product.generic_name}")
        if product.categories:
            lines.append("   Categories: " + ", ".join(product.categories[:8]))
        if product.labels:
            lines.append("   Labels: " + ", ".join(product.labels[:8]))
        if product.countries:
            lines.append("   Countries: " + ", ".join(product.countries[:8]))
        if product.nutrients:
            lines.append("   Nutrients per 100g: " + " | ".join(product.nutrients))
        if include_details and product.ingredients:
            lines.append(f"   Ingredients: {product.ingredients}")
        if include_details and product.allergens:
            lines.append("   Allergens: " + ", ".join(product.allergens[:10]))
        if include_details and product.traces:
            lines.append("   Traces: " + ", ".join(product.traces[:10]))
        if include_details and product.additives:
            lines.append("   Additives: " + ", ".join(product.additives[:10]))
        if product.image_url:
            lines.append(f"   Image: {product.image_url}")
        if product.code:
            lines.append(
                f"   Open Food Facts: https://world.openfoodfacts.org/product/{product.code}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_nutrition(product: _OpenFoodFactsProduct) -> str:
    lines = [product.name]
    meta = []
    if product.brands:
        meta.append(f"brand: {product.brands}")
    if product.quantity:
        meta.append(f"quantity: {product.quantity}")
    if product.nutriscore:
        meta.append(f"Nutri-Score: {product.nutriscore} ({_nutriscore_label(product.nutriscore)})")
    if product.nova_group is not None:
        meta.append(f"NOVA: {product.nova_group} ({_nova_label(product.nova_group)})")
    if product.ecoscore:
        meta.append(f"Eco-Score: {product.ecoscore}")
    if meta:
        lines.append("   " + " | ".join(meta))
    if product.nutrients:
        lines.append("   Nutrients per 100g: " + " | ".join(product.nutrients))
    if product.allergens:
        lines.append("   Allergens: " + ", ".join(product.allergens[:10]))
    if product.traces:
        lines.append("   Traces: " + ", ".join(product.traces[:10]))
    if product.ingredients:
        lines.append(f"   Ingredients: {product.ingredients}")
    if product.code:
        lines.append(f"   Open Food Facts: https://world.openfoodfacts.org/product/{product.code}")
    return "\n".join(lines)


def _format_compare_row(index: int, product: _OpenFoodFactsProduct) -> str:
    nutrients = _nutrient_map(product)
    parts = [
        f"{index}. {product.name}",
        f"barcode: {product.code}",
    ]
    if product.brands:
        parts.append(f"brand: {product.brands}")
    if product.nutriscore:
        parts.append(f"Nutri-Score: {product.nutriscore}")
    if product.nova_group is not None:
        parts.append(f"NOVA: {product.nova_group}")
    for key in (
        "energy kcal",
        "sugars",
        "saturated fat",
        "salt",
        "proteins",
        "fiber",
    ):
        if key in nutrients:
            parts.append(f"{key}/100g: {nutrients[key]}")
    if product.allergens:
        parts.append("allergens: " + ", ".join(product.allergens[:5]))
    return " | ".join(parts)


def _nutrients(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict):
        return ()
    nutrients: list[str] = []
    for key in _NUTRIENTS:
        if key in value:
            label = key.removesuffix("_100g").replace("-", " ")
            nutrients.append(f"{label}: {_format_value(value[key])}")
    return tuple(nutrients)


def _nutrient_map(product: _OpenFoodFactsProduct) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in product.nutrients:
        if ": " in item:
            key, value = item.split(": ", 1)
            result[key] = value
    return result


def _parse_barcode_list(value: str) -> tuple[str, ...]:
    """The distinct barcodes of a comma-separated list.

    Raises:
        ToolFailure: validation_error for an invalid barcode, or more than 5 of them.
    """
    barcodes = list(dict.fromkeys(_barcode(raw) for raw in value.replace(";", ",").split(",")))
    if len(barcodes) > 5:
        raise ToolFailure(
            "validation_error",
            f"{len(barcodes)} barcodes given, at most 5 are allowed; compare them in groups",
        )
    return tuple(barcodes)


def _nutriscore_label(score: str) -> str:
    return {
        "A": "best nutritional quality",
        "B": "good nutritional quality",
        "C": "moderate nutritional quality",
        "D": "low nutritional quality",
        "E": "lowest nutritional quality",
    }.get(score.upper(), "unknown")


def _nova_label(group: int) -> str:
    return {
        1: "unprocessed or minimally processed foods",
        2: "processed culinary ingredients",
        3: "processed foods",
        4: "ultra-processed foods",
    }.get(group, "unknown")


def _tag_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_clean_tag(item) for item in value if _clean_tag(item))


def _clean_tag(value: Any) -> str:
    text = _string(value)
    if ":" in text:
        text = text.split(":", 1)[1]
    return text.replace("-", " ")


def _barcode(value: str) -> str:
    """``value``'s digits.

    Raises:
        ToolFailure: validation_error unless they are a barcode (4-32 digits).
    """
    barcode = re.sub(r"\D", "", value.strip())
    if not _BARCODE_RE.fullmatch(barcode):
        raise ToolFailure(
            "validation_error", f"invalid barcode {value.strip()!r}; a barcode has 4-32 digits"
        )
    return barcode


def _valid_filter(value: str) -> bool:
    return bool(_TEXT_FILTER_RE.fullmatch(value))


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        if value is not None and str(value).strip():
            return int(value)
    except ValueError:
        return None
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
