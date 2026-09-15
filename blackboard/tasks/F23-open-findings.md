# F23 · Achados que ficaram em aberto

- **Dono:** coordenador · **Estado:** done · **Depende de:** `main` @ `1334fa8`
- **Origem:** pedido do dono (2026-09-15): "corrige os achados que ficaram no FINDINGS.md, testa e
  depois commit e push".

## Achados e resolução

| Achado (`FINDINGS.md`) | Resolução |
|---|---|
| `X \| None` sem default não pode ser omitido | `validate_arguments` passa `None` a um parâmetro que o schema não exige e a assinatura não tem default. O schema não muda (a decisão do F12 fica). |
| `functools.partial` em `ToolGroup` | `infer_schema` usa a função embrulhada para nome, hints e docstring; `inspect.signature` já tira os argumentos fixos. `_resolve_fn` (lista em `run_tools`) usa o mesmo nome. |
| `approve(modified_args={})` ignorado | `ApprovalGate._outcome` só mantém os argumentos do modelo quando `modified_args is None`. |
| Gemini com declarações mistas numa `Tool` | Verificado ao vivo (`gemini-2.5-flash-lite`): aceita `parameters` e `parameters_json_schema` juntas e chama as duas tools. Sem alteração de código; teste `live_api` novo. |
| Chave da OpenAI enviada a um `base_url` remoto | `_resolve_key` só usa a chave do ambiente para o host do próprio fornecedor (`api.openai.com`, `api.anthropic.com`, `api.meta.ai`); outro host remoto exige `api_key=`. Breaking. |
| `output_schema` Pydantic com `strict: true` no OpenAI | Confirmado ao vivo (400 em `gpt-4.1-nano`). O adaptador normaliza schemas strict como o `parse()` do SDK: objectos fechados, todas as propriedades em `required`, `default: null` removido, `$ref` com irmãos embutido (com cópia e guarda de ciclos). Teste `live_api` novo passou. |
| `openai.APIConnectionError` sem retry nem fallback | Adaptadores OpenAI, Meta, Anthropic e Gemini (httpx e aiohttp) convertem falhas de rede em `ConnectionError`/`TimeoutError`; `_is_retryable` repete-as. xAI já mapeava gRPC `UNAVAILABLE`/`DEADLINE_EXCEEDED` para 503/504. |
| `@pytest.mark.timeout` não fazia nada | `pytest-timeout` no extra `dev`; o registo manual do marcador saiu. Confirmado com um teste descartável (sync e async cortados a 1 s). |

## Provas

- Testes que falham antes e passam depois: `tests/test_tools_validation.py` (+2),
  `tests/test_tools_schema.py` (+1), `tests/test_tools_group.py` (+2), `tests/test_runner.py` (+1),
  `tests/test_openai_provider.py` (+6), `tests/test_anthropic_provider.py` (+2),
  `tests/test_meta_provider.py` (+2), `tests/test_gemini_provider.py` (+3), `tests/test_retry.py`
  (+2), `tests/test_provider_registry.py` (+4; três testes de rota passam `api_key=`).
- Ao vivo: Gemini com declarações mistas e OpenAI com `output_schema` Pydantic
  (`tests/integration/test_provider_contracts_live.py`, 2 passed).
- Suite: 3045 passed, 22 skipped; `ruff check`/`format` limpos; pyright 0 erros; `uv lock --check`.

## Linhas do CHANGELOG

Changed: chaves do ambiente só para o host do fornecedor (Breaking); retry de falhas de rede;
`pytest-timeout`. Fixed: structured output OpenAI com Pydantic; `X | None` omitido; `partial`;
`approve(modified_args={})`.
