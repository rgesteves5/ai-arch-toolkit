# F10 · `provider` nos pedidos de metering

- **Dono:** worker-E · **Estado:** done · **Depende de:** —
- **Plano:** F10

## Problema

`OperationRequest.provider` e `UsageEvent.provider` existem, mas o `LLM` nunca os preenche (medido:
`provider=None`). Com fallbacks entre fornecedores, a telemetria não diz quem respondeu.

## Mudança

- `src/ai_arch_toolkit/core/_providers/__init__.py`: extrair o encaminhamento de `create_provider`
  para `resolve_provider_name(model, *, provider=None, base_url=None) -> str`, usado por
  `create_provider` (mesmas mensagens de erro).
- `src/ai_arch_toolkit/core/_llm.py`: guardar `self._provider_name` em `__init__`; `_meter_request`
  passa `provider=self._provider_name`. **Só** `__init__` e `_meter_request` — o coordenador reescreve
  a parte de streaming deste ficheiro a seguir.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/__init__.py`
- `src/ai_arch_toolkit/core/_llm.py`
- `tests/test_llm_metering.py`, `tests/test_provider_registry.py`

## Prova

- `MeterScope(RunConfig(retain_meter_events=True))` + `LLM("claude-sonnet-4-6", api_key="x")` com
  provider falso → `scope.events()[0].provider == "anthropic"`, em `complete` e em `stream`.
- Modelo desconhecido com `base_url="http://localhost:11434/v1"` → `"openai"`.
- Primário Anthropic a falhar com fallback `LLM("gpt-…", api_key="x")` → o evento da tentativa do
  fallback tem `"openai"`.
- `resolve_provider_name` levanta o mesmo `ValueError` que `create_provider` para um modelo
  desconhecido sem `base_url`.

## Registo do dono (transcrito pelo coordenador; o worker não conseguiu escrever fora da worktree)

- Estado: done — feito pelo worker-E na worktree `.claude/worktrees/agent-a7aa2b94367a2bed8`.
- Ficheiros tocados: `core/_providers/__init__.py` (`resolve_provider_name` extraído e usado por `create_provider`); `core/_llm.py` (import, `__init__`, `_meter_request`); `tests/test_provider_registry.py`, `tests/test_llm_metering.py`.
- Testes novos: `TestResolveProviderName` (9 casos) e 4 de metering (`complete`, `stream`, servidor local OpenAI-compatível, fallback entre fornecedores). Antes: `provider=None` e `ImportError`.
- CHANGELOG (Fixed): `OperationRequest.provider` e `UsageEvent.provider` passam a indicar o adaptador que respondeu, incluindo tentativas de fallback.
- Desvios: o nome é resolvido depois de `create_provider`, dentro de `try/except ValueError` (vira `None`), porque cerca de 100 testes fazem patch de `create_provider` com modelos sem rota. O `logger.debug` do encaminhamento aparece duas vezes (inofensivo).
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
