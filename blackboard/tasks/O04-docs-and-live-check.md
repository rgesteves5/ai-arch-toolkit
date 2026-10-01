# O04 · Documentação, quebras visíveis e verificação ao vivo final

- **Dono:** por atribuir (os docs) · dono do repositório (a verificação ao vivo) · **Estado:** todo ·
  **Depende de:** O03
- **Origem:** D43 · **Decisões:** D43 · **Regras:** `R00-rules.md`

## Problema

A O03 muda o que o OpenAI aceita e devolve. Os docs, o `AGENTS.md` e o inventário de probes dizem
hoje "Chat Completions only" e "Astra calls no tools here".

## Objectivo

- Docs e `AGENTS.md` a dizer o que o código faz depois da O03.
- Linhas propostas para o `CHANGELOG`: a quebra, com "**Breaking:**", e o que passa a funcionar.
- Inventário de probes com `tools_loop` no Astra e no 6.1 Sol.
- Matriz ao vivo dos modelos OpenAI, corrida pelo dono.

## Ficheiros

- `AGENTS.md` (as linhas do OpenAI e da Meta), `docs/llm.md`, `docs/model-compatibility.md`,
  `docs/framework-overview.md`, `README.md` (se tiver a tabela de fornecedores)
- `scripts/model_probe_models.toml`
- `tests/integration/test_provider_hardening_live.py` (se as regras verificadas ao vivo mudarem)

## Verificação ao vivo (dono)

```bash
set -a && source .env && set +a
uv run python scripts/probe_models.py --suite full --timeout-seconds 120 \
  --model gpt-6-astra --model gpt-6.1-sol --model gpt-6-sol --model gpt-6-luna --model gpt-5.5
uv run pytest -m live_api -k openai -q
```

O `probe_models.py` filtra por modelo (`--model`, repetível), não por fornecedor: junta os outros
ids OpenAI do inventário que quiseres cobrir.

## Registo do dono

- Estado: todo.
