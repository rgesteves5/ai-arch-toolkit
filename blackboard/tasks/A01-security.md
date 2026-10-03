# A01 · Segurança: nenhum endereço do ambiente (G-16) e mais chaves no `Redactor` (G-19)

- **Dono:** Claude (2026-10-03) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 1 (G-16, G-19) · **Decisões:** D48 ·
  **Regras:** `R00-rules.md`

## Problema

- **G-16.** Sem `base_url`, os SDKs da OpenAI, da Anthropic e do Gemini lêem o endereço do
  ambiente (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`), e o toolkit
  manda-lhes a chave do ambiente como se fosse o servidor do fornecedor. O `google-genai` também
  muda para o Vertex com `GOOGLE_GENAI_USE_VERTEXAI`, e o `OpenAIModerator` deixa o SDK escolher.
- **G-19.** O `Redactor` só reconhece `sk-…`: as chaves do xAI (`xai-…`), da Groq (`gsk_…`) e da
  Google (`AIza…`) passam inteiras.

## Nota de desenho (antes do código)

- **G-16:**
  - os endereços oficiais vivem no `OWN_BASE_URLS` do registo, e o `_OWN_HOSTS` sai dele;
  - OpenAI e Anthropic passam ao SDK `base_url or OWN_BASE_URLS[...]`;
  - o Gemini põe o endereço no `HttpOptions` e cria o cliente com `vertexai=False`;
  - a Meta já o fazia, e o `DEFAULT_BASE_URL` dela passa a vir do registo;
  - o `OpenAIModerator` passa o endereço da OpenAI.
- **G-19:** um padrão por formato no `_redact_text`, com o fim da chave por lookahead (uma chave da
  Google pode acabar em `-`, que o `\b` não apanha).
- **Provas:**
  - com as variáveis a apontar para um servidor falso, um pedido real pelo SDK vai para o
    endereço oficial (no teste, outro servidor local que o substitui), e o falso não recebe nada;
  - nos quatro adaptadores, o cliente do SDK fica com o endereço oficial;
  - um teste por formato de chave no `Redactor`.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/__init__.py`, `_openai.py`, `_anthropic.py`, `_gemini.py`,
  `_meta.py`; `src/ai_arch_toolkit/toolkit/moderation/_openai.py`;
  `src/ai_arch_toolkit/core/_redaction.py`
- `tests/test_provider_endpoints.py` (novo), `tests/integration/test_provider_endpoints.py`
  (novo), `tests/test_redaction.py`
- `docs/safety.md`, `docs/llm.md` (ou onde o `base_url` se explicar), `AGENTS.md`, `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-03), pela nota de desenho.
- **Reprodução antes da correcção** (o `HEAD` `104af8f`, com `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL` e `GOOGLE_GENAI_USE_VERTEXAI` definidos):
  - OpenAI, Anthropic e Gemini ficavam com o endereço do ambiente, e o Gemini em modo Vertex;
  - `redact_text` deixava passar uma chave `xai-…` inteira.
- **G-16:**
  - `OWN_BASE_URLS` no registo, de onde sai o `_OWN_HOSTS`;
  - OpenAI, Anthropic, Gemini (com `vertexai=False`), Meta (o `DEFAULT_BASE_URL` dela saiu: lê o
    registo) e o `OpenAIModerator` passam o endereço oficial ao SDK.
- **Do mesmo género, encontrado ao rever todos os clientes de SDK:** o adaptador para servidores
  compatíveis mandava os cabeçalhos `OpenAI-Organization` e `OpenAI-Project` (de `OPENAI_ORG_ID` e
  `OPENAI_PROJECT_ID`) a hosts que não são da OpenAI. Agora tira-os, como a Meta (D48).
- **G-19:** padrões `xai-`, `gsk_` e `AIza` no `_redact_text`, com o fim por lookahead.
- **Testes que afirmavam o bug, corrigidos:**
  - `tests/test_openai_provider.py::TestClient::test_disables_hidden_sdk_retries` afirmava
    `"base_url" not in kwargs`;
  - `tests/test_anthropic_provider.py::TestAnthropicProviderComplete::test_complete` esperava o
    cliente sem `base_url`.
- **Testes novos:**
  - `tests/test_provider_endpoints.py` (9: os quatro adaptadores, o Gemini fora do Vertex, o
    moderador, um `base_url` dado ainda conta, a guarda das chaves com os mesmos hosts, os
    cabeçalhos da conta OpenAI);
  - `tests/integration/test_provider_endpoints.py` (4: um pedido real pelo SDK, com o ambiente a
    apontar para um servidor local, chega ao outro servidor que substitui o oficial, e o do
    ambiente não recebe nada);
  - `tests/test_redaction.py` (2).
- **Docs:** `docs/getting-started.md`, `docs/safety.md`, `AGENTS.md`; `CHANGELOG` (Fixed, com a
  migração).
- Gate: 6232 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a; daqui não se edita):
  "o grupo 1 fechou: com `base_url=None`, os adaptadores mandam sempre para o endereço oficial e
  ignoram `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL`/`GOOGLE_GEMINI_BASE_URL`; o `Redactor` apaga
  `xai-…`, `gsk_…` e `AIza…`. Na app, o endereço oficial que o `make_llm` passa e o apagar da
  chave exacta na sondagem deixam de ser precisos."
