"""No endpoint is read from the environment (D48, the ai-network's G-16).

Without ``base_url``, the SDKs would read OPENAI_BASE_URL, ANTHROPIC_BASE_URL or
GOOGLE_GEMINI_BASE_URL, and the toolkit would send the environment's key there.
``tests/integration/test_provider_endpoints.py`` proves it with a request.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from ai_arch_toolkit.core._providers import _OWN_HOSTS, OWN_BASE_URLS, create_provider
from ai_arch_toolkit.toolkit.moderation import OpenAIModerator

ELSEWHERE = "https://elsewhere.example/v1"


@pytest.fixture
def endpoints_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
    ):
        monkeypatch.setenv(name, ELSEWHERE)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")


def _endpoint(client: object) -> str:
    api_client = getattr(client, "_api_client", None)  # google-genai keeps it in its options
    if api_client is not None:
        return api_client._http_options.base_url
    return str(client.base_url)


@pytest.mark.usefixtures("endpoints_in_the_environment")
@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gpt-5-nano", "openai"),
        ("claude-haiku-4-5", "anthropic"),
        ("gemini-3.8-flash", "gemini"),
        ("muse-spark-1.3", "meta"),
    ],
)
def test_without_base_url_every_sdk_gets_its_providers_own_endpoint(model, provider) -> None:
    adapter = create_provider(model, api_key="test-key")
    assert _endpoint(adapter._client).rstrip("/") == OWN_BASE_URLS[provider].rstrip("/")


@pytest.mark.usefixtures("endpoints_in_the_environment")
def test_gemini_stays_on_the_developer_api() -> None:
    adapter = create_provider("gemini-3.8-flash", api_key="test-key")
    assert adapter._client._api_client.vertexai is False


@pytest.mark.usefixtures("endpoints_in_the_environment")
def test_the_openai_moderator_gets_openais_endpoint() -> None:
    moderator = OpenAIModerator(api_key="test-key")
    assert str(moderator._client.base_url).rstrip("/") == OWN_BASE_URLS["openai"]


@pytest.mark.usefixtures("endpoints_in_the_environment")
def test_a_base_url_given_is_still_honored() -> None:
    adapter = create_provider(
        "claude-haiku-4-5", api_key="test-key", base_url="http://127.0.0.1:9/anthropic"
    )
    assert str(adapter._client.base_url).rstrip("/") == "http://127.0.0.1:9/anthropic"


def test_the_key_guard_knows_the_same_hosts() -> None:
    assert {name: urlsplit(url).hostname for name, url in OWN_BASE_URLS.items()} == _OWN_HOSTS


def test_openai_account_headers_never_reach_an_openai_compatible_server(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-secret")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-secret")
    adapter = create_provider("llama-3.3-70b", api_key="k", base_url="https://api.together.xyz/v1")
    client = adapter._client
    sent = {name for name, value in client.default_headers.items() if isinstance(value, str)}
    assert not {"OpenAI-Organization", "OpenAI-Project"} & sent
    assert client.auth_headers == {"Authorization": "Bearer k"}
