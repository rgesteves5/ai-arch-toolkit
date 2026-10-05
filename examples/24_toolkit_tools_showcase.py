"""24 — Toolkit Tools Showcase.

The default toolkit namespace exposes safe-by-default tools. Tools that can
inspect local files, execute commands or Python-like code, or fetch arbitrary
URLs are available only through ai_arch_toolkit.toolkit.tools.dangerous.

Most safe lookup tools require network access (weather, geo, wiki APIs).
"""

from ai_arch_toolkit import ToolFailure
from ai_arch_toolkit.toolkit.tools import (
    base64_encode,
    datetime_now,
    math_eval,
    text_stats,
    unit_convert,
    wiktionary_entry,
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
try:  # a tool that cannot answer raises ToolFailure; an agent gets it as a failed result
    # A long answer is a window: its text ends with the call that reads on.
    print(wiktionary_entry(term="serendipity", max_chars=600).value)
except ToolFailure as failure:
    print(f"  wiktionary_entry failed [{failure.error.type}]: {failure.error.message}")

print("\nAll tools available:")
print(
    "  datetime_now, timezone_convert, math_eval, unit_convert, "
    "base64_encode, base64_decode, regex_search, text_stats, "
    "json_extract, csv_read, "
    "get_weather, get_forecast, geocode, ip_lookup, country_info, "
    "wiki_search, wiki_outline, wiki_read, wiktionary_entry, hacker_news"
)
print("\nDangerous opt-in tools:")
print(
    "  from ai_arch_toolkit.toolkit.tools.dangerous import "
    "list_directory, read_file, search_files, run_command, "
    "python_repl, http_get, scrape_text"
)
