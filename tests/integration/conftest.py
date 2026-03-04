from __future__ import annotations

import os

import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
skip_no_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
