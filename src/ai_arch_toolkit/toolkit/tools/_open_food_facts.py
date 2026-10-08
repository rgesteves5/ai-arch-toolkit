"""Open Food Facts: search packaged foods, read one by barcode, compare several (T07; D39, D41).

A product is read whole: every allergen, trace, additive, category, label and country it lists,
its nutrients per 100 g with their units, and the Nutri-Score and NOVA group with what they mean
(one tool, where ``open_food_facts_product`` and ``open_food_facts_nutrition`` cut the same answer
two ways, D41). The search pages by ``page`` and ``page_size`` and counts every match (``count``;
``page_count`` is the products on the page), and an unknown barcode is a 404 or ``status`` 0
(https://github.com/openfoodfacts/openfoodfacts-server/blob/main/docs/api/ref/api.yaml).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._window import list_window


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


# Product reads (15 a minute) and searches (10 a minute) are spaced apart on separate clocks.
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
_PAGE_MAX = 20
_COMPARE_MAX = 5
_PRODUCT_URL = "https://world.openfoodfacts.org/product/"
_BARCODE_RE = re.compile(r"^\d{4,32}$")
_TEXT_FILTER_RE = re.compile(r"^[\w\s,.'&()/%+-]{1,120}$", re.UNICODE)
_LISTED = ("code", "product_name", "brands", "quantity", "nutriscore_grade", "nova_group")
_FIELDS = (
    *_LISTED,
    "ecoscore_grade",
    "generic_name",
    "categories_tags",
    "labels_tags",
    "countries_tags",
    "ingredients_text",
    "allergens_tags",
    "traces_tags",
    "additives_tags",
    "nutriments",
    "image_front_url",
)
_NUTRIENTS = (
    "energy-kcal",
    "fat",
    "saturated-fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "proteins",
    "salt",
    "sodium",
)
_COMPARED = ("energy-kcal", "sugars", "saturated-fat", "salt", "proteins", "fiber")
# What the scores mean (https://world.openfoodfacts.org/nutriscore, /nova).
_NUTRISCORE = {
    "A": "best nutritional quality",
    "B": "good nutritional quality",
    "C": "moderate nutritional quality",
    "D": "low nutritional quality",
    "E": "lowest nutritional quality",
}
_NOVA = {
    1: "unprocessed or minimally processed foods",
    2: "processed culinary ingredients",
    3: "processed foods",
    4: "ultra-processed foods",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class _Product:
    """A product as the tools read it."""

    code: str
    name: str
    generic_name: str
    brands: str
    quantity: str
    lists: tuple[tuple[str, tuple[str, ...]], ...]
    ingredients: str
    allergens: tuple[str, ...]
    nutriscore: str
    nova_group: int | None
    ecoscore: str
    nutrients: tuple[tuple[str, str], ...]
    image_url: str


@tool(capability="network")
def open_food_facts_product(barcode: str) -> str:
    """Read a packaged food by barcode: brand, scores, nutrients per 100 g, ingredients,
    allergens, traces, additives, categories, labels and countries.

    Args:
        barcode: The product's barcode (GTIN).

    Raises:
        ToolFailure: validation_error when ``barcode`` is not a barcode; not_found when Open
            Food Facts has no product with it.
    """
    product = _fetch_product(_barcode(barcode))
    return "\n   ".join(
        [f"Open Food Facts product {product.code}: {product.name}", *_details(product)]
    )


@tool(capability="network")
def open_food_facts_search(
    product_name: str = "",
    brand: str = "",
    category: str = "",
    country: str = "",
    label: str = "",
    max_results: Annotated[int, Range(1, _PAGE_MAX)] = 5,
    page: Annotated[int, Range(1)] = 1,
) -> ToolResult:
    """Search packaged foods in Open Food Facts by name, brand, category, country or label.

    Args:
        product_name: Words of the product's name.
        brand: A brand, e.g. "ferrero".
        category: A category, e.g. "breakfast cereals".
        country: A country where it is sold, e.g. "france".
        label: A label, e.g. "organic".
        max_results: How many products a page lists.
        page: Which page, from 1; the footer gives the next.

    Raises:
        ToolFailure: validation_error when no filter is given, or a filter has characters it
            cannot take.
    """
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
    arguments = ("product_name", "brand", "category", "country", "label")
    given = list(zip(arguments, filters.values(), strict=True))
    invalid = [name for name, value in given if value and not _valid_filter(value)]
    if invalid:
        raise ToolFailure(
            "validation_error",
            f"invalid filter value for {', '.join(invalid)}; use 1-120 letters, digits, spaces "
            "and basic punctuation (,.'&()/%+-)",
        )
    described = ", ".join(f"{name.replace('_', ' ')} {value!r}" for name, value in given if value)
    params = {"fields": ",".join((*_LISTED, "ecoscore_grade")), "page_size": str(max_results)}
    params |= {"page": str(page), **{key: value for key, value in filters.items() if value}}
    return _SEARCH.get_json(
        "search",
        params=params,
        parse=lambda data: _search_answer(data, described, page, max_results),
    )


@tool(capability="network")
def open_food_facts_compare(barcodes: str) -> str:
    """Compare packaged foods side by side, one line each: scores, the main nutrients per 100 g
    and allergens.

    Args:
        barcodes: Up to 5 barcodes (GTINs), separated by commas.

    Raises:
        ToolFailure: validation_error when the list is empty, too long, or holds something that
            is not a barcode; not_found when Open Food Facts has none of the products (a missing
            one among found ones is named in the answer).
    """
    products: list[_Product] = []
    missing: list[str] = []
    for barcode in _parse_barcode_list(barcodes):
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
    lines = ["Open Food Facts comparison (nutrients per 100 g):"]
    lines += [_compare_row(number, product) for number, product in enumerate(products, start=1)]
    if missing:
        lines.append("Missing products: " + ", ".join(missing))
    return "\n".join(lines)


# --- The product --------------------------------------------------------------------------------


def _fetch_product(barcode: str) -> _Product:
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


def _product(data: dict[str, Any]) -> _Product | None:
    if data.get("status") == 0:
        return None
    product_data = data.get("product")
    return _parse_product(product_data) if isinstance(product_data, dict) else None


def _parse_product(data: dict[str, Any]) -> _Product | None:
    code = _string(data.get("code"))
    name = _string(data.get("product_name"))
    if not code and not name:
        return None
    lists = (
        ("Traces", _tags(data.get("traces_tags"))),
        ("Additives", _tags(data.get("additives_tags"))),
        ("Categories", _tags(data.get("categories_tags"))),
        ("Labels", _tags(data.get("labels_tags"))),
        ("Countries", _tags(data.get("countries_tags"))),
    )
    return _Product(
        code=code,
        name=name or "(unnamed)",
        generic_name=_string(data.get("generic_name")),
        brands=_string(data.get("brands")),
        quantity=_string(data.get("quantity")),
        lists=lists,
        ingredients=_string(data.get("ingredients_text")),
        allergens=_tags(data.get("allergens_tags")),
        nutriscore=_string(data.get("nutriscore_grade")).upper(),
        nova_group=_int_or_none(data.get("nova_group")),
        ecoscore=_string(data.get("ecoscore_grade")).upper(),
        nutrients=_nutrients(data.get("nutriments")),
        image_url=_string(data.get("image_front_url")),
    )


def _details(product: _Product) -> list[str]:
    """The record's lines after its title, every list whole."""
    nutrients = " | ".join(f"{name} {amount}" for name, amount in product.nutrients)
    lines = [
        " | ".join(_present(("brand", product.brands), ("quantity", product.quantity))),
        " | ".join(_scores(product, labelled=True)),
        _labelled("Generic name", product.generic_name),
        _labelled("Nutrients per 100 g", nutrients),
        _labelled("Ingredients", product.ingredients),
        _labelled("Allergens", ", ".join(product.allergens)),
        *(_labelled(label, ", ".join(values)) for label, values in product.lists),
        _labelled("Image", product.image_url),
        _labelled("Open Food Facts", f"{_PRODUCT_URL}{product.code}" if product.code else ""),
    ]
    return [line for line in lines if line]


def _scores(product: _Product, *, labelled: bool = False) -> list[str]:
    """Nutri-Score, NOVA group and Eco-Score, with what the first two mean when ``labelled``."""
    scores: list[str] = []
    if product.nutriscore:
        meaning = _NUTRISCORE.get(product.nutriscore, "unknown")
        scores.append(f"Nutri-Score: {product.nutriscore}" + (f" ({meaning})" if labelled else ""))
    if product.nova_group is not None:
        meaning = _NOVA.get(product.nova_group, "unknown")
        scores.append(f"NOVA: {product.nova_group}" + (f" ({meaning})" if labelled else ""))
    if product.ecoscore:
        scores.append(f"Eco-Score: {product.ecoscore}")
    return scores


def _compare_row(number: int, product: _Product) -> str:
    amounts = dict(product.nutrients)
    compared = [
        f"{name} {amounts[name]}" for name in map(_nutrient_name, _COMPARED) if name in amounts
    ]
    parts = [
        f"{number}. {product.name}",
        f"barcode {product.code}",
        *_present(("brand", product.brands)),
        *(score for score in _scores(product) if not score.startswith("Eco-Score")),
        *compared,
        _labelled("allergens", ", ".join(product.allergens)),
    ]
    return " | ".join(part for part in parts if part)


def _nutrients(value: object) -> tuple[tuple[str, str], ...]:
    """Each nutrient per 100 g with its unit (the answer's ``<nutrient>_unit``, or kcal for
    energy and g for the rest)."""
    if not isinstance(value, dict):
        return ()
    amounts: list[tuple[str, str]] = []
    for key in _NUTRIENTS:
        amount = value.get(f"{key}_100g")
        if amount is None or isinstance(amount, dict | list):
            continue
        unit = _string(value.get(f"{key}_unit")) or ("kcal" if key == "energy-kcal" else "g")
        amounts.append((_nutrient_name(key), f"{_number(amount)} {unit}"))
    return tuple(amounts)


def _nutrient_name(key: str) -> str:
    return "energy" if key == "energy-kcal" else key.replace("-", " ")


def _number(value: object) -> str:
    """A number as a person writes it: no exponent, and a fraction to six significant digits."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _string(value)
    if isinstance(value, int):
        return str(value)
    try:
        return format(Decimal(f"{value:.6g}").normalize(), "f")
    except InvalidOperation:  # inf, nan
        return str(value)


# --- The search ---------------------------------------------------------------------------------


def _search_answer(data: dict[str, Any], described: str, page: int, size: int) -> ToolResult:
    found = data.get("products")
    products = [
        product
        for item in (found if isinstance(found, list) else [])
        if isinstance(item, dict)
        if (product := _parse_product(item))
    ]
    skip = (page - 1) * size
    if not products:
        later = f" on page {page}" if page > 1 else ""
        return ToolResult.success(f"No Open Food Facts products match {described}{later}.")
    count = data.get("count")
    total = count if isinstance(count, int) and not isinstance(count, bool) else None
    more = total is not None and skip + len(products) < total
    entries = [
        _search_entry(number, product) for number, product in enumerate(products, start=skip + 1)
    ]
    window = list_window(
        entries, first=skip + 1, total=total, next_call={"page": page + 1} if more else None
    )
    heading = f"Open Food Facts products for {described} (read one with open_food_facts_product):"
    return window.result(heading=heading)


def _search_entry(number: int, product: _Product) -> str:
    parts = [
        f"{number}. {product.name}",
        f"barcode {product.code}" if product.code else "",
        *_present(("brand", product.brands), ("quantity", product.quantity)),
        *_scores(product),
    ]
    return " | ".join(part for part in parts if part)


# --- Arguments and JSON -------------------------------------------------------------------------


def _parse_barcode_list(value: str) -> tuple[str, ...]:
    """The distinct barcodes of a comma-separated list.

    Raises:
        ToolFailure: validation_error for an invalid barcode, or more than 5 of them.
    """
    barcodes = list(dict.fromkeys(_barcode(raw) for raw in value.replace(";", ",").split(",")))
    if len(barcodes) > _COMPARE_MAX:
        raise ToolFailure(
            "validation_error",
            f"{len(barcodes)} barcodes given, at most {_COMPARE_MAX} are allowed; compare them "
            "in groups",
        )
    return tuple(barcodes)


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


def _tags(value: object) -> tuple[str, ...]:
    """Taxonomy tags without their language prefix: ``en:no-gluten`` is ``no gluten``."""
    if not isinstance(value, list):
        return ()
    cleaned = (_string(item).split(":", 1)[-1].replace("-", " ") for item in value)
    return tuple(tag for tag in cleaned if tag)


def _present(*pairs: tuple[str, str]) -> list[str]:
    return [f"{label}: {value}" for label, value in pairs if value]


def _labelled(label: str, value: str) -> str:
    return f"{label}: {value}" if value else ""


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _string(value)
    return int(text) if text.isdigit() else None
