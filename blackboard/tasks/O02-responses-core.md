# O02 · Núcleo Responses partilhado, extraído do `_meta.py`

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (pode correr em paralelo com a
  O01)
- **Origem:** D43 · **Decisões:** D11 a D14 (Meta), D43 · **Regras:** `R00-rules.md`

## Problema

- Tudo o que é da Responses API vive no `_meta.py` (674 linhas): itens de entrada, function tools,
  `text.format`, montagem da `Response`, eventos de stream, reenvio do `_raw`, usage, falhas
  dentro da resposta e do stream, `input_tokens.count`. A O03 precisa do mesmo para o OpenAI;
  copiá-lo daria duas casas ao mesmo assunto.
- O reenvio só verifica o tipo do `_raw` (`_replayable_output`: `isinstance(raw, SDKResponse)`).
  Hoje só a Meta devolve esse tipo. Com a O03 o OpenAI também o devolve, e um `fallback=` entre os
  dois mandaria a um fornecedor o raciocínio cifrado do outro. A OpenAI também não reaproveita
  raciocínio de outra família de modelos ("Persisted reasoning can be reused only within the same
  model family", https://developers.openai.com/api/docs/guides/reasoning).

## Objectivo

- `core/_providers/_responses.py` com o que é da Responses API. O `_meta.py` fica só com o que é
  da Meta: host, chave, códigos de erro, esforços, `tool_choice` só `auto`, `strict: false`, a
  `web_search` alojada.
- O comportamento da Meta não muda: os testes dela passam sem alteração, salvo os novos do
  reenvio.

## Desenho (confirma-o na nota de desenho, antes do código)

- O núcleo recebe um perfil, uma dataclass congelada com o que varia entre fornecedores: as regras
  de esforço, o `tool_choice`, o `strict`, as tools alojadas aceites, o mapa de códigos de erro e
  a família do modelo.
- **Reenvio:** um `_raw` só se reenvia quando o fornecedor e a família do modelo coincidem com os
  do pedido. A família é uma regra do perfil, resolvida por `core/_model_id.py` (nada de
  `startswith` sobre ids). Fora disso a mensagem reconstrói-se dos campos, sem raciocínio, como
  hoje quando o texto não coincide.
- Sem caminhos duplos nem shims: o `_meta.py` usa o núcleo e perde o código que se mudou.
- Confirma as formas no SDK instalado (`openai` 3.19.2) e regista o que confirmaste.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/_responses.py` (novo), `src/ai_arch_toolkit/core/_providers/_meta.py`
- `src/ai_arch_toolkit/core/_model_id.py` (só se a família precisar de uma regra nova)
- `tests/test_meta_provider.py` (só testes novos), `tests/test_responses_core.py` (novo),
  `tests/test_architecture.py` (se a casa única precisar de teste)

## Provas

- Os testes da Meta passam sem alteração.
- Teste novo, que falha antes: um `_raw` de outro fornecedor, ou de outra família de modelos, não
  é reenviado; a mensagem reconstrói-se sem raciocínio.
- Saldo de linhas registado: o par `_meta.py` + `_responses.py` contra o `_meta.py` de hoje. A
  descida a sério vem na O03, quando o OpenAI usar o núcleo.

## Registo do dono

- Estado: todo.
