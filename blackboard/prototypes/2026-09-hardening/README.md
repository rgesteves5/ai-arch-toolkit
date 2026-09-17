# Protótipos · plano de robustez (2026-09-17)

Scripts descartáveis que sustentam `docs/internal/hardening-plan.md`. São evidência e ponto de
partida, não código a importar: têm caminhos absolutos do scratchpad e `sys.path.insert`. Quem pegar
numa tarefa transforma-os em testes do repositório e apaga o que já não for preciso.

Correr com `uv run python <script>` a partir da raiz do repositório. Nenhum faz chamadas a
fornecedores: usam providers falsos, servidores em `127.0.0.1` ou sockets bloqueados. O script que
chegou a tocar num endpoint real (`exp_sdk_local.py`) não foi guardado.

| Pasta | Conteúdo |
|---|---|
| `meter/` | `retry_under_cap.py`, `failure_matrix.py`, `step_cap.py`: a chamada falhada fica com custo desconhecido e, sob `max_cost`, nega retry, fallback e o resto do run (achado de 2026-09-17). |
| `wire-contract/` | Validação dos pedidos dos adaptadores contra os tipos dos próprios SDKs. `strict.py` reescreve o core schema do pydantic (extras proibidos, `Iterable` validado, modelos de resposta só por instância); `contract.py` é o registo por adaptador; `wire_contract_plugin.py` é a fixture autouse; `exp5_strict_probes.py` são os canários (9 maus têm de falhar, 2 bons têm de passar); `capture.py` gera os 23 cenários por adaptador. |
| `adapter-phases/` | `fakeserver.py` e experiências por SDK: onde o pedido sai, que excepções escapam de cada fase, causa de transporte (`httpx`), reenvio escondido do `google-genai` em `aiohttp`, hook `event_hooks` que marca "despachado". |
| `tool-results/` | Forma que cada adaptador dá a um histórico com duas tool calls paralelas e dois resultados; `_raw` do Gemini depois de um stream. |
| `tool-invariants/` | `ast_survey.py` (capacidades por tool pelo grafo de chamadas), `net_static.py`, `caps_static.py`, `harness*.py` (argumentos e corpos hostis, sockets bloqueados), `bounds_local.py`, `inflate_plugin.py`, `hosts_by_tool.json` (semente da lista de hosts por módulo). |
