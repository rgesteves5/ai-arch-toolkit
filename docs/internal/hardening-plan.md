# Plano de robustez: causas, correcções estruturais e garantias

**Data:** 17 de setembro de 2026 (revisto no mesmo dia). **Código:** `main` em `3a4d1ec`.
**Ponto de partida:** os achados de 2026-09-15 e 2026-09-17 em `blackboard/FINDINGS.md`.
**Método:** cada afirmação foi lida no código ou reproduzida sem rede; o que depende de terceiros foi
lido na documentação oficial a 2026-09-17. Os scripts estão em
`blackboard/prototypes/2026-09-hardening/`. O que não foi confirmado ao vivo está marcado.

## 0. Resumo

Os cerca de quarenta achados em aberto são sintomas de seis causas:

1. O meter só conhece "liquidou" ou "falhou", e trata um custo desconhecido como infinito.
2. Os adaptadores não têm fases nem um mapeador de erros único, e montam a mesma resposta por dois caminhos.
3. As regras por modelo estão espalhadas, com o ramo antigo por omissão; os preços casam por prefixo.
4. Os testes validam os pedidos contra os nossos pressupostos, não contra os SDKs nem contra as regras dos fornecedores.
5. A segurança das tools vive de convenções, sem infra-estrutura comum.
6. O motor emite eventos à mão em várias saídas.

A primeira versão deste plano corrigia as causas, mas cinco dos seus mecanismos eram remendos (secção
1). O código onde eles assentariam já é o mais complexo do repositório: em `core/` há 35 funções com
complexidade ciclomática acima de 10 e 34 com mais de 60 linhas, quase todas em `_llm.py` e nos
adaptadores. Por isso o plano passa a redesenhar quatro costuras, com o desenho aprovado antes do
código e orçamentos de qualidade verificados no CI.

A ordem: actualizar dependências (duas têm saltos de versão maior), instrumentos, desenho das quatro
costuras, meter, um adaptador de cada vez, tools em paralelo, e por fim preços, motor e documentação.
A frente C retoma depois; cinco sub-tarefas dela passam para aqui (C01a, C02a, C02b, C08c e a validade
do fio do C05a).

## 1. Regras de desenho

**O critério.** Uma correcção é estrutural quando reduz o número de sítios que conhecem um assunto; é
um remendo quando acrescenta um caso especial a um sítio que já existe. Cada tarefa desta frente
responde à pergunta: que código desaparece?

**O que a primeira versão tinha de remendo, e o que o substitui:**

| Remendo | Porque é remendo | Em vez disso |
|---|---|---|
| Carimbo `delivery` lido com `getattr` na excepção | Campo solto, sem tipo, que qualquer caminho pode esquecer | Hierarquia tipada de erros no core; a classificação é um campo obrigatório do tipo (costura A) |
| `LLM` a usar `prepare()` "quando existe" e a cair para `complete()` | Dois caminhos para sempre | Um só caminho; `complete`/`stream`/`stream_events` implementados uma vez em `BaseProvider`; os 27 providers falsos dos testes passam a um único duplo de teste (costura B) |
| Disposições metidas em `_try_with_tracking` e outra vez em `_StreamRun` | O ciclo de vida de uma tentativa já está implementado duas vezes (cerca de 190 e 630 linhas) | Uma só pipeline de tentativa para `complete` e streams (costura C) |
| Validador que reescreve o core schema do pydantic como guarda principal do fio | Depende de nomes internos do pydantic | Os adaptadores constroem pedidos com os `TypedDict` do próprio SDK e o pyright, que já corre no CI, verifica o fio; o validador fica como rede de testes, com canários |
| Mais um casador de prefixos para os perfis | Seria o quinto | Uma gramática única de ids de modelo (costura D) |
| Listas `xfail`, de legado e de desvios tolerados sem fim à vista | Andaimes que ficam | Cada lista tem dono e critério de saída: fica vazia e é apagada na vaga que a criou |

**Regras para o código novo e tocado:**

- Um assunto, uma casa. Testes de arquitectura por AST garantem-no: excepções de SDK só no mapeador de
  cada adaptador; `urllib` só em `toolkit/tools/_http.py`; verificações de prefixo de modelo só em
  `core/_model_id.py`; `core` nunca importa `toolkit`; nada de `getattr`/`hasattr` sobre providers.
- Sem caminhos duplos que sobrevivam a uma vaga. Uma migração acaba na vaga em que começa.
- Tipos em vez de strings e flags: `Literal`, dataclasses congeladas, `TypedDict` do SDK. Nenhum
  `Any` novo em assinaturas públicas, nenhum `# type: ignore` novo.
- Orçamento de complexidade no CI: `C901`, `PLR0912` e `PLR0915` do ruff em `core/`, com a linha de
  base de hoje (35 funções acima de 10) a só poder descer; código novo ou tocado passa sem `noqa`.
  Nenhuma função acima de 60 linhas nos módulos redesenhados.
- Saldo de linhas por vaga nos módulos tocados. Em `core/_providers/` e `_llm.py` o saldo esperado é
  negativo (secção 2, costuras B e C).
- Desenho primeiro: para as quatro costuras escreve-se a interface (assinaturas, docstrings, estados,
  mapa de módulos antes e depois) e o dono aprova antes de haver código. Cada vaga fecha com uma
  revisão adversarial de desenho, com uma pergunta: isto acrescentou um caso especial ou tirou um?

## 2. As quatro costuras a redesenhar

**A · Modelo de erros** (`core/_exceptions.py`). Uma base `ProviderError` com o campo `delivery:
Literal["not_sent", "unbilled", "indeterminate"]`, definido na construção. Por baixo: `RequestError`
(erro ao preparar; herda também de `ValueError`), `APIError` e `RateLimitError` (o fornecedor
respondeu com erro), `TransportError` e `ProviderTimeout` (sem resposta; herdam também de
`ConnectionError` e `TimeoutError`, para quem já apanha os builtins) e `ResponseError` (HTTP 200
inutilizável ou falha dentro do stream). `PROVIDER_ERRORS` passa a ser a base; retry, fallback e meter
lêem campos tipados. Desaparecem: adivinhar pela `status_code`, os `getattr(exc, "status_code")` e os
estados inventados (`APIError(200)`, 5xx sintéticos).

**B · Contrato do provider** (`core/_providers/_base.py`). Três fases obrigatórias: `prepare()`
síncrono e puro, `send()`/`open_stream()` com toda a E/S dentro de um único mapeador de erros por
adaptador, e uma só montagem da `Response`. `complete`, `stream` e `stream_events` existem uma vez, na
base. Em quatro dos cinco SDKs o próprio SDK já acumula o objecto final do stream (Anthropic, OpenAI,
xAI, Meta); só o Gemini precisa de uma função de junção. Hipótese a confirmar SDK a SDK no desenho,
já sobre as versões novas. Desaparecem: os 44 blocos `except` que apanham excepções de SDK (165
linhas) espalhados pelos cinco adaptadores, que passam a cinco mapeadores; os ciclos de stream
duplicados da Anthropic; e o estado de acumulação manual.

**C · Pipeline de tentativa** (`core/_attempts.py`, novo). Uma tentativa física é sempre: admitir →
vaga de inferência → `mark_started()` → enviar → liquidar ou falhar com disposição, mais o registo
`Attempt`. A regra de retry e fallback é uma só: permitido enquanto nada foi entregue a quem chamou
(`complete` nunca entrega antes do fim; um stream deixa de poder depois do primeiro item). `LLM` fica
como fachada. Desaparecem: `_try_with_tracking`, `_StreamRun`, `_FallbackStreamRun`, `_single_stream`
e `_stream_with_fallbacks` como implementações paralelas (complexidade 20 e 23; 142 e 168 linhas), e
os ganchos de ciclo de vida que hoje viajam num `Callable` e são descobertos por `getattr`
(`_stream_abandon`, `_stream_refresh`, `_stream_release`, `_stream_attempts`, `_meter_op`; nove
sítios em `_llm.py` e `_response.py`). A pipeline também passa a ser dona da cadeia de fallbacks,
sem alterar os objectos `LLM` que o utilizador passou. O `on_event` do C01 liga-se a esta pipeline em
vez de criar um terceiro caminho.

**D · Gramática de ids de modelo** (`core/_model_id.py`, novo). Uma função resolve um id: exacto, id
com sufixo de snapshot (datas e carimbos por fornecedor), família, ou desconhecido. Usam-na os
preços, os perfis dos adaptadores, a escolha do tokenizer, o encaminhamento por fornecedor e, depois,
o catálogo do C06. Desaparecem: nove verificações `startswith` espalhadas pelo `core` e as listas
abertas de modelos novos.

Consolidações sem desenho novo: `toolkit/tools/_http.py` (35 helpers privados, 262 linhas, mais 11
`urlopen` inline → um módulo), limites de saída e de tempo no executor governado, e um único ponto de
emissão de `step_end` no motor.

## 3. As causas

### Causa 1 · O meter não distingue falhas e trata "desconhecido" como infinito

**Raiz.** `MeterStore.fail()` não recebe informação nenhuma e atribui `Cost.unknown` a toda a operação
`llm` que não liquida (`core/_metering/_store.py:156-160`). O plano original fixou isso sem taxonomia
de falhas (`metering-plan.md:135-137`). Os dois consumidores de `unknown_cost_count` fecham a porta ao
primeiro desconhecido (`toolkit/budget/_controller.py:102-108`, `core/_step_engine.py:212`). E
`mark_started()` corre antes de o adaptador montar o pedido e antes do `inference_slot`
(`core/_llm.py:677-683`), ao contrário do executor de tools, que valida, passa os gates e só então
abre a operação (`core/_tools/_executor.py:322-356`).

**Explica.** Retry e fallback mortos sob `max_cost`; um timeout de step, um cancelamento ou um stream
abandonado envenenam o resto do scope (o padrão documentado "timeout → fallback" nunca funciona com
tecto); o tecto por step falha depois de um retry com sucesso; um `ValueError` do adaptador conta como
chamada iniciada; cancelar na fila do `inference_limit` conta como chamada; uma server tool é negada
antes de correr; `reserve="strict"` não reserva tools com preço (C08c).

**O que os fornecedores facturam (documentação oficial, 2026-09-17).** Anthropic: pedidos falhados não
são cobrados; um timeout ou corte do cliente num pedido que ia ter sucesso **é** cobrado. Gemini: 400
e 500 não são cobrados; o resto não é dito. OpenAI, xAI e Meta: quase nada documentado; a xAI cobra
pedidos recusados por violação das regras de uso. Logo, "erro HTTP = custo zero" não pode ser regra
global, e "enviado sem resposta" é mesmo incerto.

**Correcção estrutural.**

- **Disposição da falha no core** (costura A): `op.fail(disposition)`. `not_sent` e `unbilled` dão
  custo zero conhecido; `indeterminate` dá custo incerto. A contagem mantém-se nos três, para
  `max_llm_calls` continuar a limitar tempestades de retries. O `UsageEvent` leva a disposição.
- **Desconhecido com tecto.** `Cost.unknown(reason, at_most=Money)`. O store soma os tectos num
  contador próprio, que entra na conta de `max_cost` e do tecto por step, mas nunca no `cost`
  reportado. Só um desconhecido **sem** tecto (modelo sem preço, server tool sem tarifa) acciona
  `unpriced="fail_closed"`. Com `reserve="strict"` o tecto é a reserva da própria operação, que passa a
  ficar retida em vez de libertada; com `reserve="none"` o controller calcula-o na altura da falha com
  o mesmo estimador do modo estrito. O relatório ganha `cost_at_most`; `cost` continua a ser o gasto
  conhecido.
- **Pipeline de tentativa única** (costura C), com a ordem do executor de tools. Um erro de
  construção nunca abre operação.
- **C08c entra aqui:** o estimador reserva o preço de tools com `Pricer`.

**Prova.** Matriz de interacções gerada por parametrização (protótipo `meter/failure_matrix.py`):
classe de falha × modo do scope (sem scope, só medir, tecto suave, tecto estrito) × caminho
(`complete`, `stream`, `stream_events`) × recuperação (retry, fallback, chamada seguinte, retry e
fallback de step). A tabela esperada é a especificação e vai para `docs/safety.md`. Mais um teste de
conservação do meter com sequências aleatórias de sementes fixas. Os testes que hoje fixam o erro são
corrigidos (`tests/test_llm_metering.py:279-319`).

### Causa 2 · Adaptadores sem fases nem mapeador único; duas montagens da mesma resposta

**Raiz.** Cada adaptador repete o bloco `except` por ponto de entrada e só embrulha o `await` inicial.
A construção já é pura nos cinco, mas nada o garante, e o tipo da excepção não identifica a fase: um
`ValueError` cru tanto sai da validação local do SDK (nada foi enviado) como de um HTTP 200 com corpo
ilegível (resposta cobrada). O caminho de stream remonta a `Response` à parte do `complete()`.

**Explica.** Erros de transporte a meio de um stream escapam crus em Anthropic, OpenAI e Meta, sem
retry nem fallback; o erro dentro do stream do OpenAI escapa cru; 529 não é repetido; `count_tokens` e
batch da Anthropic sem ramo 429; o xAI ignora `timeout` (27 minutos por omissão do SDK); o Gemini muda
de pilha HTTP conforme o `aiohttp` esteja instalado e, nessa pilha, reenvia pedidos às escondidas;
`stream_events()` perde `parsed`, `response_id` e `citations`; o `raw` do Gemini em stream é só o
último chunk; stream sem usage fica com custo zero conhecido (C01a).

**Correcção estrutural.** Costuras A e B. Mais: transporte determinista (Gemini fixo em `httpx` por
`HttpOptions.httpx_async_client`; o xAI passa o `timeout` ao SDK) e 529 nos estados repetíveis.

**Prova.** Testes com servidor falso em loopback por adaptador (protótipo
`adapter-phases/fakeserver.py`): ligação recusada, servidor mudo, fecho sem resposta, RST, 429, 500,
529, erro dentro do stream, corte a meio. Cada caso sai como o tipo de erro certo, é repetido ou passa
ao fallback conforme a política, e bate com a matriz da causa 1. Teste de paridade por adaptador: a
mesma resposta lógica por `complete()` e por stream em chunks dá `Response` igual, incluindo o que o
histórico reenvia.

### Causa 3 · Regras por modelo espalhadas, com o ramo antigo por omissão; preços por prefixo

**Raiz.** Nos três adaptadores principais, os modelos novos entram por lista explícita e tudo o resto
recebe a forma antiga (`_anthropic.py:54-62,238-252`, `_openai.py:60-61`, `_gemini.py:256`). A lista
aberta é a dos modelos futuros: cada geração nova parte até alguém a editar. O registo de preços tem o
mesmo defeito: o prefixo mais longo foi pensado para snapshots datados e também casa sucessores e
variantes.

**Explica.** `thinking=True` recusado nos Claude actuais (e inválido nos antigos com os valores por
omissão); `tool_choice="required"` no Fable 5.1; `thinking_effort="xhigh"` no Gemini 3 só avisa;
`o3-pro` cobrado como `o3` (um décimo do preço real), `gpt-4o-audio-preview` como `gpt-4o`, Fable 5.1
com a cache do Fable 5. Um custo "conhecido" errado passa por baixo do `fail_closed`.

**Correcção estrutural.**

- **Uma tabela declarativa de perfis por adaptador** (família → forma do pedido), resolvida pela
  gramática única (costura D). O perfil por omissão é o da geração mais recente; as famílias antigas
  são uma lista fechada. O que o modelo não suporta levanta `RequestError` em `prepare()`, como a Meta
  já faz (D13).
- **Anthropic, segundo a documentação:** 4.6 e seguintes usam `{"type": "adaptive"}` com
  `output_config.effort` (fundido com `output_config.format`) e `display: "summarized"` quando se pede
  thinking; nos modelos em que o thinking vem ligado, `thinking=False` omite o parâmetro; só os antigos
  levam `budget_tokens`, com `max_tokens` acima do budget.
- **Preços:** id exacto ou sufixo de snapshot; o resto fica sem preço e falha fechado com mensagem
  accionável. Entradas em falta acrescentadas. `register(..., match="prefix")` fica como opção
  explícita para famílias locais.
- **Auditoria de deriva:** `scripts/audit_models.py`, local e grátis (APIs de modelos), lista ids sem
  preço exacto ou sem perfil. O catálogo do C06 passa a ler estas tabelas (C06c).

**Prova.** Testes de contrato por família sobre a saída de `prepare()`, gramática de sufixos com a
lista de casos perigosos, e uma sonda ao vivo local por família, registada com data em
`docs/model-compatibility.md`.

### Causa 4 · Os testes validam contra os nossos pressupostos

**Raiz.** Os testes injectam um cliente falso que aceita tudo, por isso a forma do pedido só é
comparada com o que nós achamos que o SDK quer. Os cenários cobrem uma tool call de cada vez. E as
combinações entre mecanismos nunca foram cruzadas (falha × tecto × retry). Prova recente: com todas as
dependências actualizadas, incluindo dois saltos de versão maior de SDK, a suite passa inteira (3045),
porque nenhum teste toca no SDK real.

**Explica.** Server tool da Anthropic sem `name`; `{"type": "web_search"}` no Chat Completions;
`document` com `name` em vez de `title` (o teste afirma o erro); `tool_choice` com nome no xAI rebenta
no SDK real; o Gemini nunca junta resultados de tools nem devolve o `id`; a Anthropic recebe um `user`
por resultado, forma que a documentação chama "Wrong"; só a Meta tem teste de chamadas paralelas com
reenvio; um teste afirma `unknown_cost_count == 1` depois de um 500.

**Correcção estrutural.**

- **Fio verificado por tipos:** os construtores de pedidos passam a devolver os `TypedDict` do próprio
  SDK (modelos pydantic no Gemini, protobuf no xAI), e o pyright verifica o fio em cada PR, sem custo
  em runtime. Verificado num ficheiro de ensaio: o pyright apanha os três erros de dict conhecidos.
- **Rede de testes por cima:** a fixture que valida cada pedido construído contra os tipos do SDK fica
  confinada a `tests/`, com canários (nove pedidos maus têm de falhar, dois bons têm de passar), para o
  que a tipagem estática não vê. O protótipo funciona nos cinco adaptadores sem dependências novas.
- **Contrato de conversa:** históricos neutros canónicos (uma chamada, chamadas paralelas, texto com
  chamadas, thinking com chamadas, histórico depois de stream) × adaptadores, com as regras de
  sequência documentadas por fornecedor como asserções.
- **Testes de transporte** com servidor falso (causa 2): são os únicos que exercitam o SDK real sem
  custo, e por isso os que apanham uma versão maior de SDK.
- **Verificação ao vivo local:** uma selecção `live_api` de contratos, corrida pelo dono, com custo de
  cêntimos e resultados datados. Nunca no CI.

### Causa 5 · A segurança das tools vive de convenções

**Raiz.** Cada um dos 44 módulos reimplementa pedido, leitura e formatação. "Seguro por omissão",
"devolve strings de erro" e "só stdlib" estão em prosa; o único teste de namespace fixa sete nomes e
nunca olha para o que uma tool segura faz. As 125 tools seguras têm todas `capability=None`. O
executor não limita saída nem tempo.

**Explica.** `csv_read` lê qualquer ficheiro sem aprovação; `mediawiki_*` aceitam o host do modelo
(pedido tentado a `169.254.169.254`); `ip_lookup` em `http://` revela o IP da máquina; `..` chega ao
caminho em três tools; zero de 49 leituras com limite; redirects seguidos para outro host; 5 MB
chegam ao modelo; `max_chars=-1` contorna tectos; `math_eval("9**9**9")` e um regex com backtracking
prendem uma thread; 12 tools levantam com argumentos hostis e 90 com corpos inesperados; grupo de
tools vazio trocado por outro (C02a); `@tool(schema=)` corrompe a tool (C02b).

**Correcção estrutural.**

- **Um único ponto de saída HTTP**, `toolkit/tools/_http.py`: só `https` (excepções explícitas), host
  numa lista dada pelo módulo chamador, sem redirects para outro host, leitura com `max_bytes`, prazo
  total, segmentos de caminho com `quote(safe="")`, erros de charset e de forma do JSON devolvidos
  como string. Os 47 sítios migram na mesma vaga.
- **Garantias no executor governado** (core), válidas para qualquer tool, incluindo as dinâmicas (C02)
  e as de MCP (C03): `ToolRuntimePolicy.max_output_chars` e `timeout_s`; o corte fica marcado no texto
  e em `metadata`. As tools de cálculo ganham guardas de complexidade à entrada, porque uma thread não
  se mata. A validação de argumentos passa a recusar `None` em parâmetros não anuláveis e a
  validar os elementos de listas de escalares (achado de 2026-09-17; alinha com a D15 proposta no C02).
- **Metadados verdadeiros:** toda a tool declara `capability`; um teste por AST confirma que o
  declarado bate com o que o código alcança. `csv_read` passa para `dangerous`; `mediawiki_*` ficam
  com lista de hosts Wikimedia e uma factory governada para outros wikis; `ip_lookup` passa a `https`
  e deixa de aceitar `ip` vazio; `youtube_*` ficam como excepção declarada (extra próprio) mas cumprem
  os mesmos invariantes.
- **Invariantes executáveis** em `tests/toolkit/test_tool_invariants.py`, com descoberta automática:
  partição de exports, capacidades por AST, host fixo, nunca levanta, saída limitada.
- **Grupo vazio e `schema=`:** `is None` em vez de `or` no `Agent` e nas cinco estratégias;
  `@tool(schema=<schema completo>)` levanta `TypeError` (C02a e C02b).

### Causa 6 · O motor emite eventos à mão

`step_end` é emitido em vários sítios, depois do check de budget (`toolkit/flow/_executor.py:356-359`,
`:434-437`): quando `max_wall_s` é excedido, o step fica no trace e no estado mas o evento nunca sai.
A raiz está na forma de `_run_dag`, a função mais complexa do motor (radon 48): tem um caminho rápido
para vagas de um step e outro para vagas paralelas, com os checks de budget, prazo e erro de
orquestração repetidos em cada um. Correcção: uma vaga de um step é uma vaga; fica um só caminho,
um só ponto de emissão e um verificador de gramática de eventos reutilizado pelos testes do motor.
`_run_attempts` (radon 33) tem o mesmo bloco de fallback escrito várias vezes e entra na mesma tarefa.

## 4. Dependências

Ensaio de 2026-09-17 numa cópia descartável, com `uv lock --upgrade`: 95 pacotes mudam. Dois SDKs têm
salto de versão maior, e ambos trocam o transporte para `httpx2`: `anthropic` 0.116.0 → 1.6.0 e
`openai` 2.45.0 → 3.14.1. Também `google-genai` 2.11.0 → 2.24.0, `xai-sdk` 1.17.0 → 1.19.0, `ruff`
0.15.21 → 0.16.8 e `pyright` 1.1.411 → 1.1.414. Resultado com tudo actualizado: 3045 testes passam,
pyright sem erros, um aviso novo do ruff (`RUF036`, com correcção automática). O ensaio com servidor
falso confirmou o `httpx2` no fio (a fuga a meio do stream passa a ser `httpx2.ReadTimeout`).

Consequências:

- **A actualização vem primeiro.** O mapeador de erros e os pedidos tipados dependem da versão maior
  do SDK; desenhá-los sobre `httpx` seria trabalho perdido.
- **A suite verde não prova nada sobre o fio** (causa 4): os dois saltos maiores só ficam verificados
  com os testes de transporte e com a verificação ao vivo local.
- **Os mínimos declarados não são testados** e quase de certeza são falsos (`anthropic>=0.40`,
  `openai>=1.50`; o código usa `output_config` e a Responses API). Passam a ser a versão mais baixa
  que o CI testa, com um job `--resolution lowest-direct`, e tecto na versão maior seguinte. As apps
  resolvem as dependências por estes intervalos, não pelo `uv.lock`.
- Actualizam-se também os hooks do pre-commit (`ruff-pre-commit` v0.15.13) e as actions do CI
  (`checkout@v4`, `setup-uv@v4`, `setup-python@v5`).

## 5. Como se garante

1. **Vermelho primeiro.** Os instrumentos entram antes das correcções, sem mudar comportamento. Cada
   falha conhecida fica enumerada por um `xfail` estrito; cada correcção vira os seus.
2. **Andaimes com saída:** listas de legado e de `xfail` só encolhem e são apagadas na vaga que as criou.
3. **Canários** em cada validador, para um no-op nunca passar por verde.
4. **Orçamentos de qualidade no CI** (secção 1): complexidade, tamanho, testes de arquitectura, pyright.
5. **Cada correcção entra com o teste que falhava antes** e, quando toca no fio, com a verificação ao
   vivo local do adaptador no fim.
6. **Auditoria de deriva** antes de cada release; resultados das sondas com data.
7. **Higiene de release:** secção "Notas de actualização" no `CHANGELOG` e uma tag de pré-release por
   vaga, para as apps fixarem uma versão e subirem de propósito.

## 6. Ordem

| Vaga | Conteúdo | Notas |
|---|---|---|
| 0a | Dependências (secção 4) | Mecânico; o ensaio está verde. |
| 0b | Instrumentos e orçamentos de qualidade | Sem mudança de comportamento. |
| 0c | Desenho das quatro costuras, para aprovação | Só interfaces e mapas de módulos; nenhum código. |
| 1 | Costuras A e C, e a causa 1 | O único impeditivo. |
| 2 | Costuras B e D e as causas 2, 3 e 4, um adaptador de cada vez: Anthropic → Gemini → OpenAI → Meta → xAI | Verificação ao vivo local no fim de cada um. |
| 3 | Causa 5: tools | Em paralelo com a vaga 2; ficheiros disjuntos. |
| 4 | Preços e auditoria, causa 6, deriva do `AGENTS.md`, notas de actualização, tag | |

Depois retoma a frente C: o C01 liga-se à pipeline de tentativa; o C05 assenta nos perfis e no
`prepare()`; o C02 e o C03 nas garantias do executor; o C04 no meter.

## 7. Mudanças visíveis para quem depende do toolkit

Todas com entrada `Changed` e nota de actualização: mínimos e tectos das dependências; modelos sem
entrada exacta de preço passam a não ter preço (falham fechado sob `max_cost`); `csv_read` exige
aprovação e muda de namespace; `mediawiki_*` recusam hosts fora da lista; saídas de tools são cortadas
ao limite; um erro de construção deixa de contar como tentativa; falhas não cobradas deixam de
incrementar `unknown_cost_count`; os erros ganham uma base comum (os `except` existentes continuam a
apanhar os mesmos casos); o thinking da Anthropic muda de forma no fio; um grupo de tools vazio mantém
a identidade; 529 passa a ser repetido.

## 8. Decisões a fixar

| # | Decisão | Recomendo |
|---|---|---|
| R1 | Custo desconhecido com tecto conta para os tectos e não para `cost` | Sim. |
| R2 | Falhas sem política documentada: 429 não cobrado em todos; outras respostas de erro só onde documentado (Anthropic todas, Gemini 400 e 500); o resto fica incerto com tecto. Alternativa: toda a resposta de erro a zero | A primeira: nunca um zero silencioso, e os runs continuam. |
| R3 | Contrato de três fases como caminho único, com `complete`/`stream`/`stream_events` na base e um duplo de teste partilhado | Sim. |
| R4 | Pipeline de tentativa única em `core/_attempts.py`, em vez de acrescentar ramos às duas implementações actuais | Sim; é a mudança maior e a que mais simplifica. |
| R5 | Gemini fixo em `httpx` | Sim. |
| R6 | Preços por id exacto ou sufixo de snapshot, também em `register()`; prefixo só por opção | Sim; é a quebra mais visível. |
| R7 | Thinking antigo da Anthropic: enviar `max_tokens = budget + max_tokens` ou cortar o budget | Somar, e medir com o valor que vai no fio. |
| R8 | `mediawiki_*` com lista Wikimedia e factory governada; `youtube_*` como excepção declarada | Sim. |
| R9 | Valores por omissão do executor: `max_output_chars` generoso (200 000) e `timeout_s` sem valor; as tools do toolkit declaram os seus | Sim. |
| R10 | Verificação ao vivo local por adaptador (cêntimos; chaves disponíveis) | Autorizar por adaptador, no fim da sua vaga. |
| R11 | Tag de pré-release por vaga e notas de actualização | Sim. |
| R12 | Mínimos das dependências iguais à versão mais baixa testada no CI, com tecto na versão maior seguinte | Sim. |
| R13 | Orçamento de complexidade no CI: limiar 10 para código novo ou tocado em `core/`, linha de base a só descer | Sim. |
| R14 | Desenho das quatro costuras aprovado antes de qualquer código | Sim. |
| R15 | Imutabilidade: vista só de leitura documentada e congelamento barato à entrada, sem cópias profundas por step | Sim. |

## 9. Configuração estrita e contrato de imutabilidade

Dois achados de 2026-09-17 que não cabem nas seis causas mas são contrato, não manutenção:

- **Nada se descarta em silêncio.** `ReasoningSpec.from_mapping` ignora uma `policy` em dict, um
  `output_schema` malformado e chaves desconhecidas. Regra: o que não se reconhece levanta, com a
  chave no erro, como os manifestos já fazem. Entra na vaga 4.
- **Imutabilidade.** `State.snapshot()` promete uma cópia imutável e partilha os valores;
  `ReasoningSpec.knobs` é mutável num dataclass congelado. Decisão R15: documentar a vista só de
  leitura e congelar à entrada onde é barato (`MappingProxyType` nos knobs), sem cópias profundas
  por step, que trariam de volta o custo quadrático medido em N7.

## 10. Dívida de manutenção confirmada, fora deste plano

Apontada por duas revisões externas e verificada a 2026-09-17. Não parte comportamento; fica para uma
frente própria, depois desta, pela ordem abaixo.

1. `toolkit/memory/graph/_store.py` reimplementa a fachada de `core/graph/_store.py`: 20 dos 22
   métodos de `GraphStore` têm o mesmo nome em `Graph`, sem herança nem composição; 54 janelas de 8
   linhas duplicadas. Precisa de desenho (o store de memória tem embeddings, índice e backends seus).
2. As nove flow factories são closures com 13 a 20 parâmetros; cada knob comum repete-se nas
   factories, em `_builders.py` (`trace_capture=s.trace_capture` dez vezes), no schema do manifesto e
   nas docs. `budget_policy` existe nas nove factories e os builders nunca o passam: não é código
   morto (as factories são API pública), é um segundo caminho para o mesmo budget.
3. Contrato de chaves de estado por strings (`"task"`, `"response"`, `"answer"`, `"last_answer"`…),
   com `extract_text` a adivinhar por quatro quedas sucessivas (`toolkit/agents/_compile.py:44`).
   Aberto desde a auditoria de Julho.
4. Dois validadores de manifesto escritos à mão (agentes, 1006 linhas; prompts, 931), mais um JSON
   Schema à parte que pode divergir.
5. `_eval_expr` (radon 84) é uma cadeia de `isinstance` numa fronteira de segurança; uma tabela de
   despacho por tipo de nó torna o conjunto permitido enumerável e testável.
6. Seis estratégias repetem "criar um ReAct interno, correr, extrair a resposta, inspeccionar o
   trace"; o LLMCompiler tem escalonamento próprio dentro de um step.
7. `__all__` de topo com 198 nomes: superfície pública larga para pré-1.0; decidir antes da 1.0.

## 11. Fora deste plano

Capacidades novas (frente C), a app, sandbox de código, liquidação parcial de streams partidos com o
usage já recebido (Anthropic e xAI dão-no; fica como melhoria depois da causa 2) e a porta para a
Responses API do OpenAI.
