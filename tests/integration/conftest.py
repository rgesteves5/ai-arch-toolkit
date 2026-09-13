from __future__ import annotations

import os

import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
XAI_KEY = os.environ.get("XAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
META_KEY = os.environ.get("MODEL_API_KEY")
skip_no_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
skip_no_xai = pytest.mark.skipif(not XAI_KEY, reason="XAI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(
    not GEMINI_KEY, reason="GOOGLE_API_KEY / GEMINI_API_KEY not set"
)
skip_no_meta = pytest.mark.skipif(not META_KEY, reason="MODEL_API_KEY not set")
