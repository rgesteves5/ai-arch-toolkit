"""24 — Toolkit Tools Showcase.

The toolkit includes 25+ pre-built tools across 11 categories. All use
stdlib only (zero pip dependencies). Here's a quick demo of several.

Most tools require network access (weather, geo, wiki APIs).
Filesystem and shell tools work locally.
"""

from ai_arch_toolkit.toolkit.tools import (
    base64_encode,
    datetime_now,
    define_word,
    math_eval,
    text_stats,
    unit_convert,
)

# --- Math ---
print("=== Math ===")
print("  42 * 17 =", math_eval(expression="42 * 17"))
print("  100 km =", unit_convert(value=100, from_unit="km", to_unit="miles"))

# --- Text ---
print("\n=== Text ===")
print("  base64('hello') =", base64_encode(text="hello"))
print("  stats:", text_stats(text="The quick brown fox jumps over the lazy dog."))

# --- DateTime ---
print("\n=== DateTime ===")
print("  now:", datetime_now())

# --- Knowledge ---
print("\n=== Knowledge ===")
print("  'serendipity':", define_word(word="serendipity")[:200], "...")

print("\nAll tools available:")
print(
    "  datetime_now, timezone_convert, math_eval, unit_convert, "
    "base64_encode, base64_decode, regex_search, text_stats, "
    "list_directory, read_file, search_files, run_command, "
    "json_extract, csv_read, http_get, scrape_text, "
    "get_weather, get_forecast, geocode, ip_lookup, country_info, "
    "wikipedia_search, wikipedia_article, define_word, hacker_news"
)
