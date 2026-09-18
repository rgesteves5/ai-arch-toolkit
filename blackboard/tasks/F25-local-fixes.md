# F25 · Correcções locais prontas a pegar

- **Dono:** Codex (coordenador da fase) · **Estado:** done · **Depende de:** nada (cada item é independente)
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

- Estado: done
- Itens feitos: os nove, incluindo a migração nanope e as tarifas exactas após autorização explícita do dono. O histórico abaixo conserva os bloqueios iniciais já resolvidos.
- Testes novos: 37 casos (31 nas correcções 1–8 e 6 de custo calculado no item 9). Corrigidos os testes antigos do nome do documento, do erro remoto de IP e dos argumentos do cliente xAI.
- Verificações: R00 completa no fim de cada item; após desbloqueio do import 3101 passed, 22 deselected, e após preços 3107 passed, 22 deselected; lint/formato/pyright/lock limpos. Tabela histórica e resultado actual na R01.
- CHANGELOG: entradas em [Unreleased] para CSV, IP, eventos, 529, documento, timeout, hosts e preços; quebra de CSV assinalada.
- Desvios: itens independentes 2–8 avançaram durante o bloqueio do import, ao abrigo da R00; nenhum instrumento começou antes de resolver ambos os bloqueios e ter o gate completo verde. Item 8 só prosa, com prova por leitura pedida na ficha.

### Item 1 · Nota de desenho

`csv_read(path, max_rows=100) -> str` mantém a implementação e passa a declarar a aprovação e o risco no decorador. Só o namespace `dangerous` a exporta; desaparece a exposição segura. Prova: partição de exports e execução governada sem handler negada antes da leitura.

### Item 1 · Resultado e bloqueio

Vermelho: quatro falhas de exports/governança; a reprodução comportamental final devolveu conteúdo privado em vez de `approval_denied`. Correcção aplicada. Verificação R00: 3055 passed, 17 failed, 22 deselected; lint, formato, pyright e lock limpos. As 17 falhas são imports de `csv_read` no nanope; a R00 proíbe tocar nessa árvore. Pedida autorização para migrar um único import, sem shim nem enfraquecimento de testes. Avanço apenas em itens independentes, como manda a secção Bloqueios da R00.

### Item 2 · Nota de desenho

`ip_lookup(ip="") -> str` valida e normaliza com `ipaddress.ip_address()` antes de construir o URL; argumentos vazios ou inválidos devolvem erro sem I/O. Desaparece a consulta implícita do IP da máquina. Prova: `urlopen` nunca chamado nos argumentos hostis; a falha remota continua testada com um IP válido.

### Item 2 · Resultado

Vermelho: 3 casos devolviam sucesso e tentavam o pedido. Correcção e teste remoto com IP válido aplicados. R00: 3058 passed, 17 falhas do bloqueio F25.1, 22 deselected; ruff/formato/pyright/lock limpos. HTTPS registado em FINDINGS, conforme a ficha.

### Item 3 · Nota de desenho

A sequência do motor passa a emitir `_step_end()` logo depois de registar e fundir o resultado, antes de `_over_budget()`, nos dois caminhos indicados. Não muda o escalonamento. Prova: teste parametrizado em sequential/dag exige `step_end` antes de `budget_exceeded`.

### Item 3 · Resultado

Vermelho: faltava `step_end` nos dois modos. Correcção aplicada; R00: 3060 passed, as mesmas 17 falhas do import nanope, 22 deselected; lint/formato/pyright/lock limpos.

### Item 4 · Nota de desenho

`RetryConfig.retry_on_status` é a casa única dos estados repetíveis. Acrescenta-se 529 à tabela; a decisão `_is_retryable` mantém-se. Prova: `with_retry()` repete uma falha 529 e devolve o sucesso seguinte com a configuração por omissão.

### Item 4 · Resultado

Vermelho: 529 propagava à primeira tentativa. R00: 3061 passed, 17 falhas do import nanope, 22 deselected; restantes controlos limpos.

### Item 5 · Nota de desenho

A montagem de conteúdo Anthropic usa `title` para `DocumentPart.name`; elimina-se o campo inválido `name`. Confirmação local: `anthropic.types.DocumentBlockParam.__annotations__` contém `title` e não `name`; referência oficial indicada no achado: https://platform.claude.com/docs/en/api/messages (não consultada por restrição de rede). Corrige-se o teste que afirmava o bug e exige-se a ausência de `name`.

### Item 5 · Resultado

Vermelho: `KeyError: title`. Teste antigo corrigido sem o enfraquecer. R00: 3061 passed, 17 falhas do import nanope, 22 deselected; restantes controlos limpos.

### Item 6 · Nota de desenho

`XAIProvider(model, api_key, timeout=None)` passa o timeout directamente ao construtor SDK e elimina o aviso falso. Confirmado com `inspect.signature(xai_sdk.AsyncClient.__init__)`, que aceita `timeout: Optional[float]`; referência https://github.com/xai-org/xai-sdk-python (não consultada por restrição de rede). Prova: construtor mockado recebe o valor sem aviso; teste dos retries gRPC preservado e actualizado para o valor `None`.

### Item 6 · Resultado

Vermelho: construtor sem `timeout` e aviso falso. R00: 3062 passed, 17 falhas do import nanope, 22 deselected; restantes controlos limpos.

### Item 7 · Nota de desenho

`_valid_api_url(value) -> bool` continua a ser a casa única da validação das três tools: HTTPS, host exacto ou subdomínio de um domínio Wikimedia enumerado, sem credenciais nem porta. Prova parametrizada em três tools × URLs hostis, incluindo hosts com sufixo enganador; caso positivo português. Desaparece a escolha arbitrária de host.

### Item 7 · Resultado

Vermelho: 21 casos passavam a validação e tentavam rede mockada; caso português já passava. R00: 3084 passed, 17 falhas do import nanope, 22 deselected; restantes controlos limpos.

### Item 8 · Nota de desenho

Só documentação: `AGENTS.md` passa a descrever os contratos já existentes. Confirmação por leitura em `_providers/__init__.py` (prefixo `chat-`) e `_anthropic.py` (`structured_output_mode="prompt"`). Não há comportamento a corrigir nem teste novo de linhas de prosa: a ficha pede prova por leitura.

### Item 8 · Resultado

Leitura dos contratos confirmada. R00: 3084 passed, 17 falhas do import nanope, 22 deselected; restantes controlos limpos.

### Item 9 · Bloqueio

A ficha exige verificar as tarifas nas páginas oficiais e escrever URL/data. A R00 e o pedido limitam qualquer rede a `127.0.0.1`. Não há uma cópia verificada das páginas nos protótipos, e o achado só fornece a tarifa de cache do Fable 5.1, não a tabela completa do Gemini 3.8 Flash. Não se inventaram preços nem comentários de verificação. `_default_pricing.toml` e os testes de preços ficam intactos. A verificação R00 do item 8 é também a do estado final deste bloqueio (sem alteração de código no item 9).

### Retoma autorizada · 2026-09-18

O dono autorizou explicitamente o único import nanope e as páginas oficiais de preços. Migrado o import; os testes vermelhos da execução anterior são a prova antes da correcção. Restantes proibições em vigor.

### Item 9 · Nota de desenho e fontes (2026-09-18)

Só tabelas: ids exactos Fable/Mythos 5.1 e Gemini 3.8 Flash; o casador mantém-se para a R02. https://platform.claude.com/docs/en/about-claude/pricing confirma entrada/saída 10/50, escrita de cache 12,50, leitura 0,25 e desconto batch 50% cumulativo com cache, para ambos os Claude. https://ai.google.dev/gemini-api/docs/pricing confirma Gemini 3.8 Flash 0,75/3,75, cache 0,075, batch 0,375/1,875 e cache batch 0,0375 (promoção até 2026-12-31). Prova comportamental: preços calculados com tokens de entrada, saída e cache, nos dois modos.

Migração nanope verificada: 3101 passed, 22 deselected; todos os restantes comandos R00 limpos. Bloqueio do item 1 resolvido.

### Item 9 · Resultado

Seis falhas vermelhas observadas: Claude cobrava 61/30,5 em vez de 60,25/30,125; Gemini devolvia None. Entradas exactas acrescentadas, sem mudar o casador. R00: 3107 passed, 22 deselected; lint, formato, pyright e lock limpos. Os nove itens estão concluídos. Registo anterior de bloqueio preservado como histórico; ambos os bloqueios foram resolvidos por autorização explícita do dono.
