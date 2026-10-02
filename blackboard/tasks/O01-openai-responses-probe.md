# O01 · Sonda ao vivo: a Responses do OpenAI contra a Chat Completions

- **Dono:** Claude (o script e a execução: o dono pediu-o em 2026-10-02) · **Estado:** done ·
  **Depende de:** nada
- **Origem:** D43; LOG de 2026-09-28 (os custos relatados da porta) · **Decisões:** D43 ·
  **Regras:** `R00-rules.md`, com uma excepção dada pelo dono em 2026-10-02: o Claude corre o
  script, que faz chamadas pagas à OpenAI com um tecto de custo.

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
- `tests/test_probe_openai_responses.py` (novo; só a lógica sem rede)
- `scripts/model_probe_notes.md` (os resultados)

## Critério

- O script corre com a autorização do dono, e os números ficam aqui e em
  `scripts/model_probe_notes.md`.
- A O03 só começa com as verificações 2 a 7 respondidas: são elas que fixam o pedido.
- A latência decide se a D43 avança como está. Se a Responses for claramente mais lenta, decide o
  dono, com os números à frente.

## Registo do dono

- Estado: done (2026-10-02). Corrida final `openai-responses-20261002T022338Z`, em
  `scripts/output/model-probes/` (ignorado pelo git), com 6 repetições. Custou $0.03, mais
  quatro pedidos avulsos de `max`, de custo desprezável.
- Ficheiros: `scripts/probe_openai_responses.py` (novo), `tests/test_probe_openai_responses.py`
  (novo, 8 testes sem rede), `scripts/model_probe_notes.md` (as notas em inglês).
- Duas corridas exploratórias corrigiram o script antes da final:
  - um modelo que vê qual tool chamar não devolve item de raciocínio, por isso a pergunta das
    verificações 3 e 4 obriga-o a pensar primeiro;
  - a mensagem do utilizador não tem `type`.

### Resultados

1. **Latência: a Responses não fica atrás.** Valores p50.
   - Luna a `none`: 1,11 s contra 1,35 s; primeiro token em stream 0,50 s contra 0,65 s.
   - 6.1 Sol a `low`: 2,49 s contra 2,36 s; primeiro token 1,10 s contra 1,27 s.
   - Loop de tools da Luna a `none`: 1,02 s e 0,93 s por volta, contra 0,91 s e 0,87 s.
   - A cache rendeu o mesmo nos dois endpoints (2527 de 3049 tokens em média).
   - Os 2 a 3× do relato de terceiros não se viram, nesta janela de minutos e com N=6.
   - A Responses contou menos tokens de entrada para a mesma tool: 69 contra 151.
2. **Raciocínio sem estado.** Com `store: false`, o `encrypted_content` vem por omissão. O
   `include` legado ainda é aceite.
3. **Itens sem par e reenvio por id.** A OpenAI aceita o que a Meta recusa:
   - o raciocínio só por id, sem `encrypted_content`;
   - o raciocínio seguido de uma mensagem do utilizador, em vez da sua call;
   - a volta reconstruída sem raciocínio (o caminho de reconstrução funciona).
4. **Outra família.** O raciocínio da Luna reenviado ao 6.1 Sol e ao `gpt-5.5` é aceite; o
   servidor ignora-o, ou usa-o, sem erro.
5. **`strict` omitido.** A OpenAI reescreve o schema em modo strict: devolve `strict: true`,
   todas as propriedades obrigatórias e `additionalProperties: false`. Um parâmetro opcional
   passa a obrigatório sem aviso. Com `strict: false`, o schema fica como foi enviado.
6. **Parâmetros que a Responses não tem.**
   - `stop` e `seed` dão 400 `unknown_parameter`.
   - `frequency_penalty` e `presence_penalty` dão 500 ao fim de cerca de 90 s, em três corridas
     de três. Com o retry do `LLM`, a espera repetir-se-ia.
7. **Sampling.** As regras são as da Chat Completions:
   - a Luna só aceita `temperature` e `top_p` a `none`;
   - o 6.1 Sol recusa `temperature` nos dois endpoints;
   - os logprobs a `none` vêm com `top_logprobs` e `include: ["message.output_text.logprobs"]`.
8. **Capacidades.**
   - Tools com raciocínio funcionam na Responses: no 6.1 Sol a `low`, e na Luna no loop.
   - Os resumos com `summary: "auto"` chegam (Luna).
   - A Chat Completions recusa tools com raciocínio no 6.1 Sol (ao effort por omissão e a
     `none`) e na Luna a `low`.
   - O `max` é recusado pela Chat Completions em todos os GPT-6, e aceite pela Responses no
     6.1 Sol.
   - As mensagens trazem `phase: "final_answer"`, e as function calls não trazem outros campos.

### O que isto fixa

- **A D43 avança:** a latência medida não é pior.
- **O03, pedido:**
  - sem `include`;
  - `strict: false` explícito, porque sem ele as tools mudam de semântica;
  - `RequestError` antes de enviar para as quatro kwargs;
  - a tabela de sampling de hoje mantém-se;
  - logprobs com `top_logprobs` e o seu `include`;
  - o perfil da Responses aceita `max` nos GPT-6;
  - o reenvio mantém o `phase`.
- **O02:** a guarda de fornecedor e família continua certa. Na OpenAI evita carga inútil, não
  erros. O risco é a Meta receber raciocínio da OpenAI: já recusou com 400 um item de raciocínio
  que não conseguia resolver (M01).
  O reenvio deve usar `by_alias=True`: o SDK tem campos com alias, como o `async`, que a OpenAI
  ainda não manda.
- **Corrigido fora do plano:** o adaptador mandava `max` aos GPT-6 pela Chat Completions, que
  responde 400. O teste falhou antes da correcção, e os quatro modelos foram verificados ao vivo.
  Fica no `CHANGELOG` (Fixed), no `AGENTS.md` e no `docs/model-compatibility.md`.
