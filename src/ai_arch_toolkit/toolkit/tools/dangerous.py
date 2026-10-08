"""Dangerous toolkit tools that require explicit opt-in.

These tools can execute commands, inspect local files, run Python-like code, or
fetch arbitrary URLs. Expose them to agents only with sandboxing, permission
checks, and human approval appropriate for your application.

The file tools an agent should get come from ``filesystem_tools(policy)``: the reads and the
writes bound to a ``FilesystemPolicy``'s folders, each checking its paths right before it acts.
A ``PathScopeGate`` refuses a call outside those folders before anyone is asked to approve it.
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._filesystem import list_directory, read_file, search_files
from ai_arch_toolkit.toolkit.tools._filesystem_policy import (
    FilesystemAction,
    FilesystemPolicy,
    FilesystemPolicyError,
    PathScopeGate,
)
from ai_arch_toolkit.toolkit.tools._filesystem_write import filesystem_tools
from ai_arch_toolkit.toolkit.tools._json import csv_read
from ai_arch_toolkit.toolkit.tools._python import python_repl
from ai_arch_toolkit.toolkit.tools._shell import run_command
from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text

__all__ = [
    "FilesystemAction",
    "FilesystemPolicy",
    "FilesystemPolicyError",
    "PathScopeGate",
    "csv_read",
    "filesystem_tools",
    "http_get",
    "list_directory",
    "python_repl",
    "read_file",
    "run_command",
    "scrape_text",
    "search_files",
]
