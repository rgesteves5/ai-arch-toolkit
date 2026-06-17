"""Dangerous toolkit tools that require explicit opt-in.

These tools can execute commands, inspect local files, run Python-like code, or
fetch arbitrary URLs. Expose them to agents only with sandboxing, permission
checks, and human approval appropriate for your application.
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files
from ai_arch_toolkit.toolkit.tools._python import python_repl
from ai_arch_toolkit.toolkit.tools._shell import run_command
from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text

__all__ = [
    "http_get",
    "list_directory",
    "python_repl",
    "read_file",
    "run_command",
    "scrape_text",
    "search_files",
]
