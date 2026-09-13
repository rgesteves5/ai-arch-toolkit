# F05 · Fundir prompts de sistema nos adaptadores

- **Dono:** worker-E · **Estado:** done · **Depende de:** —
- **Plano:** S, R, N6, F5 · **Decisões:** D3 (revisto)

## Problema

Todos os adaptadores escolhem **um** prompt de sistema: `system if system is not None else msg_system`
(Anthropic, Gemini, xAI) e, no OpenAI, `_messages_to_sdk(system=…)` descarta as mensagens `system()`
quando há `system=`. Medido: `system="B"` + `system("A")` nas mensagens → só `"B"` chega;
`MemoryMiddleware` (escreve `request.system`) apaga a mensagem `system()`; o `batch_submit` da
Anthropic ignora sempre as mensagens `system()` (`_, wire = _messages_to_sdk(messages)`).

## Decisão (D3 revisto)

Nunca escolher; juntar. Ordem: o argumento `system=` primeiro, depois as mensagens `system()` pela
ordem em que aparecem, separadas por uma linha em branco (`"\n\n"`).

- Anthropic, Gemini, xAI: juntar tudo no parâmetro nativo.
- OpenAI: `system=` à cabeça; as mensagens `system()` **ficam na sua posição** (hoje já ficam quando
  não há `system=`; mensagens de sistema a meio da conversa são válidas na Chat Completions e em
  Ollama/vLLM — não as mover).
- Caminhos de batch incluídos.
- Conteúdo de sistema que não seja string: manter o comportamento actual de cada adaptador e registar
  aqui se houver perda.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/_base.py` — helper, p.ex. `merge_system_prompts(*parts)` (ignora
  `None` e vazios).
- `src/ai_arch_toolkit/core/_providers/_anthropic.py` — `complete`, `stream`, `stream_events`,
  `count_tokens`, `batch_submit`. **Atenção:** o worker-A mexe no mesmo ficheiro em `_extract_usage` e
  no tratamento de `message_start`/`message_delta`; não toques nessas zonas.
- `src/ai_arch_toolkit/core/_providers/_openai.py` — `_messages_to_sdk` (serve também o batch)
- `src/ai_arch_toolkit/core/_providers/_gemini.py` — `complete`, `stream`, `count_tokens`, batch se existir
- `src/ai_arch_toolkit/core/_providers/_xai.py` — `complete`, `stream`, batch se existir
- Testes que hoje fixam a precedência e vão mudar: `tests/test_openai_provider.py:102,415`,
  `tests/test_gemini_provider.py:415`, `tests/test_anthropic_provider.py:435`; `tests/test_xai_provider.py`
  e `tests/test_batch.py` se aplicável.
- `docs/llm.md` — uma frase onde se fala de prompts de sistema (procura o sítio certo).

## Prova

- Por adaptador: `messages=[system("A"), user("x")]` e `system="B"` → o provider recebe os dois
  (`"B\n\nA"`; no OpenAI, `"B"` primeiro e `"A"` antes da mensagem do utilizador).
- OpenAI: `[user("x"), system("A"), user("y")]` mantém `"A"` no mesmo sítio, com e sem `system=`.
- Anthropic `batch_submit([{"messages": [system("A"), user("x")]}])` → `params["system"] == "A"`; com
  `"system": "B"` no pedido → `"B\n\nA"`.
- `LLM("claude-sonnet-4-6", api_key="x")` com o `_client` mockado e um middleware de teste cujo
  `abefore` devolve o pedido com `system="MEM"`, com `system("A")` nas mensagens → `messages.create`
  recebe `system="MEM\n\nA"`.

## Registo do dono (transcrito pelo coordenador; o worker não conseguiu escrever fora da worktree)

- Estado: done — feito pelo worker-E na worktree `.claude/worktrees/agent-a7aa2b94367a2bed8`.
- Ficheiros tocados: `_base.py` (novo `merge_system_prompts`); `_anthropic.py` (import, `effective_system` em `count_tokens`/`complete`/`stream`/`stream_events`, `batch_submit`); `_openai.py` (`_messages_to_sdk`); `_gemini.py` (import, `count_tokens`/`complete`/`stream`); `_xai.py` (import, `complete`/`stream`); `docs/llm.md`; testes em `test_anthropic_provider.py`, `test_openai_provider.py`, `test_gemini_provider.py`, `test_xai_provider.py`, `test_batch.py`.
- Testes novos / alterados: os 4 que fixavam a precedência foram renomeados e passam a exigir a fusão; novos em `TestAnthropicSystemPrompts` (7), OpenAI (3), Gemini (2), xAI (2), `test_batch.py` (4). Antes da correcção, 18 falhavam pela razão certa.
- Verificações na worktree: 2639 passed, 7 skipped; ruff e pyright limpos.
- CHANGELOG (Changed): os prompts de sistema passam a ser fundidos em vez de substituídos; `system=` primeiro; o OpenAI mantém as mensagens `system()` na posição; batch incluído; `system=""` deixa de suprimir mensagens `system()`.
- CHANGELOG (Fixed): `batch_submit` da Anthropic deixa de perder mensagens `system()`; middleware que escreve `request.system` (p.ex. `MemoryMiddleware`) deixa de apagar uma mensagem `system()`.
- Desvios e achados: conteúdo não-string mantém o comportamento anterior (lista numa `system()` dá `TypeError` em Anthropic/Gemini/xAI; um `system=` com blocos nativos passa inalterado e, junto com mensagens `system()`, o texto delas perde-se). Gemini e xAI não têm batch. Achado registado em `FINDINGS.md` (OpenAI não converte conteúdo não-string de `system()`).
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
