"""Provider for xAI Responses API (xAI-specific server tools)."""

from __future__ import annotations

from ai_arch_toolkit._legacy.llm._http import RetryConfig
from ai_arch_toolkit._legacy.llm._providers._openai_responses import OpenAIResponsesProvider


class XAIResponsesProvider(OpenAIResponsesProvider):
    """xAI Responses API — extends OpenAI Responses with xAI-specific server tools.

    Supported xAI server tools (via ``server_tools`` kwarg):
    - ``ServerTool(type="web_search", config={"allowed_domains": [...]})``
    - ``ServerTool(type="x_search", config={"date_range": {"start": ..., "end": ...}})``
    - ``ServerTool(type="code_execution", config={"pip_packages": [...]})``
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        retry: RetryConfig | None = None,
    ) -> None:
        super().__init__(
            model,
            api_key,
            retry=retry,
            base_url="https://api.x.ai",
        )
