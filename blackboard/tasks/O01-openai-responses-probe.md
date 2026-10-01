# O01 · Sonda ao vivo: a Responses do OpenAI contra a Chat Completions

- **Dono:** por atribuir (o script) · dono do repositório (a execução) · **Estado:** todo ·
  **Depende de:** nada
- **Origem:** D43; LOG de 2026-09-28 (os custos relatados da porta) · **Decisões:** D43 ·
  **Regras:** `R00-rules.md`. A proibição de chamadas a fornecedores mantém-se para o agente: o
  script chama a OpenAI, mas só o dono o corre.

## Problema

A D43 assenta em factos que só se vêem ao vivo, e alguns dos números conhecidos não são nossos:

1. **Latência.** Um relato de terceiros (medição de 2025 no Azure) dá a Responses 2 a 3× mais
   lenta. A OpenAI diz o contrário sobre a cache: "40% to 80% improvement when compared to Chat
   Completions in internal tests" (https://developers.openai.com/api/docs/guides/migrate-to-responses).
2. **Reenvio sem estado.** Com `store: false`, o guia de migração diz que cada item de raciocínio
   traz `encrypted_content` por omissão; o guia de raciocínio diz que a API "still accepts the
   legacy `reasoning.encrypted_content` value in `include`"
   (https://developers.openai.com/api/docs/guides/reasoning). A Meta pede-o com `include` (D12).
3. **Itens sem par.** A Meta responde 400 a um item de raciocínio sem o par ou sem
   `encrypted_content` (M01). Falta saber o que faz a OpenAI.
4. **Outra família.** "Persisted reasoning can be reused only within the same model family" (guia
   de raciocínio). Um `fallback=` de `gpt-6.1-sol` para `gpt-5.6-terra` reenviaria raciocínio de
   outra família: 400 ou ignorado?
5. **`strict` omitido.** Na Responses, "omitting `strict` attempts strict mode" (guia de migração).
   Uma function tool com um schema que o modo strict não aceita é recusada, alterada ou aceite?
6. **Os parâmetros que a Responses não tem:** `stop`, `seed`, `frequency_penalty`,
   `presence_penalty` (confirmado nos tipos do SDK 3.19.2). O SDK nem os tipa; a API dá 400 ou
   ignora-os se chegarem no corpo?
7. **Sampling.** O `gpt-6.1-sol` aceita `temperature` em algum effort? Os GPT-6 com `none` aceitam
   `temperature` e `top_p` na Responses, como na Chat Completions?

## Desenho

- `scripts/probe_openai_responses.py`, com o SDK `openai` cru (o adaptador ainda não existe), no
  estilo de `scripts/probe_models.py`: chave do ambiente, modelos por argumento (por omissão
  `gpt-6-luna`, o mais barato, e `gpt-6.1-sol`), N repetições por cenário, tecto de custo por
  argumento, custo estimado no início pelo `pricing` do pacote.
- **Latência** (os dois endpoints, mesmo prompt, mesmo limite de saída): texto simples; um loop de
  tools de duas voltas, com raciocínio na Responses e com `none` na Chat Completions (o único modo
  que ela aceita); a mesma conversa com um prefixo longo e fixo, para os tokens em cache. Em
  stream e sem stream: tempo até ao primeiro token e tempo total.
- **Verificações 2 a 7:** um pedido cada, com o resultado (aceite, 400 com a mensagem, ou
  ignorado) registado tal como vem.
- **Saída:** tabela Markdown com p50 e p95, tokens (entrada, cache, raciocínio, saída) e custo, em
  `scripts/output/` (ignorado pelo git).

## Ficheiros

- `scripts/probe_openai_responses.py` (novo)
- `tests/test_probe_openai_responses.py` (novo; só a lógica sem rede, com os sockets fechados)
- `scripts/model_probe_notes.md` (os resultados, pelo dono)

## Critério

- O dono corre o script e regista os números aqui e em `scripts/model_probe_notes.md`.
- A O03 só começa com as verificações 2 a 7 respondidas: são elas que fixam o pedido.
- A latência decide se a D43 avança como está. Se a Responses for claramente mais lenta, decide o
  dono, com os números à frente.

## Registo do dono

- Estado: todo.
