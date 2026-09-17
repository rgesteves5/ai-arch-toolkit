# F25 · Correcções locais prontas a pegar

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (cada item é independente)
- **Origem:** achados de 2026-09-15 e 2026-09-17 em `FINDINGS.md` que são bugs locais, com correcção
  local. Não dependem dos refactors do `docs/internal/hardening-plan.md` nem os atrapalham.
- **Regras:** `blackboard/README.md`. Um item de cada vez: primeiro o teste que falha, depois a
  correcção, depois `uv run pytest -q`, `ruff check`, `ruff format --check`, `pyright src`. Sem
  chamadas a fornecedores. Não alargar o âmbito: o que aparecer a mais vai para `FINDINGS.md`.

| # | Problema (título em `FINDINGS.md`) | Correcção | Ficheiros | Prova |
|---|---|---|---|---|
| 1 | "`csv_read` lê qualquer ficheiro sem aprovação" | Passa para `toolkit.tools.dangerous` com os metadados de D4 (`capability="filesystem"`, `risk_level="high"`, `requires_approval=True`); sai de `toolkit.tools`. Entrada `Changed` (quebra: import e aprovação). | `toolkit/tools/_json.py`, `tools/__init__.py`, `tools/dangerous.py`, `tests/toolkit/test_tools_exports.py`, `docs/tools-catalog.md`, `docs/safety.md` | `ToolGroup(csv_read)` sem handler → `approval_denied`; já não se importa de `toolkit.tools`. |
| 2 | "`ip_lookup` usa `http://` e revela o IP da máquina" | Recusar `ip` vazio e validar com `ipaddress.ip_address()` antes de montar o URL (erro em string). O `https` fica em aberto: o endpoint gratuito do ip-api só serve `http`; registar em `FINDINGS.md` o que se decidir. | `toolkit/tools/_geo.py`, teste em `tests/toolkit/` | `ip=""` e `ip="a b?c"` → string de erro, `urlopen` nunca chamado. |
| 3 | "`step_end` em falta quando `max_wall_s` é excedido" | Emitir `step_end` antes do check `_over_budget()` nos dois sítios (`toolkit/flow/_executor.py`, modo sequencial e vaga de um step). Não refazer `_run_dag`: isso é do plano. | `toolkit/flow/_executor.py`, `tests/flow/test_engine.py` | Flow com step de 0,1 s e `BudgetPolicy(max_wall_s=0.05)` em `iter()` → a sequência de eventos tem `step_end` do step antes de `budget_exceeded`, nos dois modos. |
| 4 | "O 529 da Anthropic não é repetido" | Juntar 529 ao valor por omissão de `RetryConfig.retry_on_status`. | `core/_retry.py`, `tests/test_retry.py`, `docs/` onde os estados estejam listados | `APIError(529, …)` é repetido com a configuração por omissão. |
| 5 | "Bloco `document` da Anthropic leva `name`; o SDK só tem `title`" | Enviar `title`. Corrigir o teste que afirma `name`. | `core/_providers/_anthropic.py:112-113`, `tests/test_anthropic_provider.py:1117` | O bloco construído tem `title` e não tem `name`. |
| 6 | "O xAI ignora `timeout`, mas o SDK aceita-o" | Passar `timeout=` a `xai_sdk.AsyncClient(...)` e tirar o aviso. | `core/_providers/_xai.py:312-325`, `tests/test_xai_provider.py` | Com o SDK em mock, o cliente é criado com o `timeout` dado. |
| 7 | "`mediawiki_*` deixam o modelo escolher o host" | `api_url` só aceita hosts Wikimedia (`wikipedia.org`, `wikimedia.org`, `wiktionary.org`, `wikidata.org`, `wikibooks.org`, `wikiquote.org`, `wikisource.org`, `wikiversity.org`, `wikivoyage.org`, `wikinews.org`, `mediawiki.org` e subdomínios), sem credenciais nem porta no URL. Outros wikis ficam para o refactor das tools. Entrada `Changed`. | `toolkit/tools/_mediawiki.py`, `tests/toolkit/test_mediawiki.py`, `docs/tools-catalog.md` | `https://169.254.169.254/api.php`, `https://localhost:8443/x/api.php`, `https://user:pw@en.wikipedia.org/w/api.php` e `https://evil.example/api.php` → string de erro sem pedido; `https://pt.wikipedia.org/w/api.php` passa. |
| 8 | "Deriva do `AGENTS.md`" | O structured output da Anthropic só cai para o prompt com `structured_output_mode="prompt"`; juntar o prefixo `chat-` à lista de routing. | `AGENTS.md` | Leitura. |
| 9 | "Preço do `claude-fable-5-1` herdado por prefixo" e "gemini-3.8-flash sem entrada" | Acrescentar as entradas em falta, cada uma verificada na página oficial de preços do fornecedor, com o URL e a data em comentário. Não mexer na regra de correspondência (é do plano, D16). | `core/_default_pricing.toml`, `tests/test_pricing.py` | `pricing.get("claude-fable-5-1").cache_read == 0.25`; cada id novo resolve para a sua entrada exacta. |

## Registo do dono

- Estado: todo
- Itens feitos:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
