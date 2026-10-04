# A04 · As fontes que falham: o TLS do sistema, a chave do Semantic Scholar e a espera do GDELT (G-30)

- **Dono:** Claude (2026-10-04) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 4 (G-30) · **Decisões:** D51, D52, D53 ·
  **Regras:** `R00-rules.md`

## Problema (uma chamada a sério por fonte, a 2026-10-04, no `0619e1a`; gratuitas)

- **Já respondem:**
  - o `arxiv_search`, o `country_info`, o `uniprot_search` e o `pdb_search` (`d892cfd`);
  - o `define_word`: o dictionaryapi.dev voltou. A falha de 27/09 era do serviço.
- **Eurostat:**
  - responde no Python do Homebrew, e falha com `CERTIFICATE_VERIFY_FAILED` no do ai-network (o
    3.14 standalone do uv);
  - esse Python lê `/etc/ssl/cert.pem` (128 raízes), onde falta a `GlobalSign Root R46` da cadeia
    do `ec.europa.eu`. A raiz está no Keychain do sistema e no ficheiro do Homebrew (D51).
- **GDELT:** 429 com qualquer User-Agent (três testados), mesmo depois de 150 s de pausa. O
  toolkit espaçava pelo início dos pedidos e voltava a bater depois de um 429 (D53).
- **Semantic Scholar:** 429 logo ao primeiro pedido. O limite dos anónimos é partilhado, e o
  toolkit não tinha como mandar uma chave (D52).
- **User-Agent:** `ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)`: a versão não é a do
  pacote, e o URL não existe.

## Nota de desenho (antes do código)

- **D51:**
  - `_http._tls_context()`: um `truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)` se o `truststore`
    se importar, e senão o `ssl.create_default_context()`. O opener leva-o num `HTTPSHandler`;
  - um `URLError` cuja razão seja um `SSLCertVerificationError`, sem o `truststore`, acrescenta ao
    erro como resolver;
  - extra `truststore` no `pyproject.toml`, também no `dev`.
- **D52:** o `Api` ganha `key_env`, `key_header` e `key_url`. O `_send` lê a chave em cada pedido
  e manda-a no cabeçalho. Sem chave, o 429 diz que variável definir e onde a pedir. O Semantic
  Scholar declara-os, com `min_interval_s=1.0` (o limite de uma chave).
- **D53:**
  - o `_Throttle` ganha `cool(host, seconds)`, `cooling(host)` e `done(host, interval)`. Um
    pedido a um host em espera falha logo, sem sair: "{nome} asked to slow down (HTTP 429): try
    again in N s";
  - o `_send` regista o fim de cada pedido (`done`). Num 429, fecha o host pelo `Retry-After`
    (segundos), ou pelo `cooldown_s` do `Api`;
  - o GDELT declara `cooldown_s=60`.
- **User-Agent:** `ai-arch-toolkit/<versão do pacote> (+https://github.com/rgesteves5/ai-arch-toolkit)`,
  com a versão lida dos metadados.
- **Provas:**
  - com o `truststore` instalado, o opener verifica com ele, e sem ele com o contexto padrão; um
    erro de certificado sem o `truststore` diz como resolver;
  - ao vivo, com o Python standalone do uv: o Eurostat falha sem o `truststore` e responde com ele;
  - com `SEMANTIC_SCHOLAR_API_KEY` definida, o pedido leva `x-api-key`, e sem ela não leva nada.
    Um 429 sem chave diz a variável e o URL;
  - depois de um 429 do GDELT, o pedido seguinte não sai e diz quanto esperar; passada a espera,
    sai. O `Retry-After` manda quando vem;
  - o intervalo conta do fim do pedido anterior;
  - o User-Agent leva a versão do pacote e o URL do repositório.

## Ficheiros

- `src/ai_arch_toolkit/toolkit/tools/_http.py`, `_gdelt.py`, `_semantic_scholar.py`
- `pyproject.toml`, `uv.lock`
- `tests/toolkit/test_http.py`, `test_gdelt.py`, `test_semantic_scholar.py`
- `docs/tools-catalog.md`, `docs/tools.md` (se falar da rede), `README.md` (os extras),
  `.env.example`, `AGENTS.md`, `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-04), pela nota de desenho; o dono escolheu as três recomendações
  (D51, D52, D53).
- **D51 (Eurostat):**
  - `_tls_context()`, `_TLS` e `_SYSTEM_STORE` no `_http.py`. O opener leva o contexto num
    `HTTPSHandler`, e assim o `fetch_page` das tools perigosas também o usa;
  - sem o `truststore`, um `SSLCertVerificationError` acrescenta ao erro como instalar o
    pacote: `pip install truststore`, ou o extra. O toolkit não está no PyPI, por isso a frase
    nomeia os dois;
  - extra `truststore = ["truststore>=0.10"]`, também no `all` (e assim no `dev`); `uv.lock`
    actualizado.
- **D52 (Semantic Scholar):**
  - o `Api` ganhou `key_env`, `key_header` e `key_url`. O `_send` lê a chave em cada pedido, e o
    `_describe` acrescenta ao 429 sem chave a variável e o URL;
  - o Semantic Scholar declara a `SEMANTIC_SCHOLAR_API_KEY` no `x-api-key`, com
    `min_interval_s=1.0`;
  - `.env.example` com a variável comentada.
- **D53 (GDELT):**
  - o `_Throttle` ganhou `done`, `cool` e `resting`, e o `HttpError` o `retry_after_s` (um
    `Retry-After` em segundos ou em data HTTP);
  - um pedido a um host em espera falha antes de sair, com o estado 429;
  - o GDELT declara `cooldown_s=60.0`.
- **User-Agent:** `ai-arch-toolkit/0.1.0.dev0 (+https://github.com/rgesteves5/ai-arch-toolkit)`, com
  a versão lida dos metadados (`dev` numa árvore não instalada). O repositório é público.
- **Testes novos (19):**
  - `test_http.py`: o User-Agent (1), `TestTls` (5), `TestKeys` (4) e `TestRest` (5);
  - `test_gdelt.py`: depois de um 429, a chamada seguinte não sai e diz quanto esperar (1);
  - `test_semantic_scholar.py`: a chave no cabeçalho, sem chave nenhum cabeçalho, e o 429 sem
    chave (3).
- **Ao vivo** (gratuito, a 2026-10-04):
  - o arXiv, o `country_info`, o UniProt, o PDB, o Eurostat e o `define_word` respondem no Python
    do Homebrew;
  - no Python do ai-network (o 3.14.3 standalone do uv), num ambiente à parte, o Eurostat falha
    sem o `truststore`, com a frase de como resolver, e responde com ele;
  - o GDELT continua a dar 429, e a chamada seguinte responde em 0 s com "try again in 60 s";
  - o Semantic Scholar sem chave dá 429, com a variável e o URL da chave.
- Gate: 6317 passed, 42 skipped; ruff, formatação e pyright limpos; `uv lock --check` em dia.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 4 fechou:
  - o arXiv, os países, o UniProt e o PDB já respondiam (`d892cfd`), e o `define_word` voltou (a
    falha era do serviço);
  - o Eurostat responde no Python standalone do uv com o extra `truststore` (D51): na app,
    instalar `truststore` (ou `ai-arch-toolkit[truststore]`), sem mais nada;
  - o Semantic Scholar manda uma chave gratuita que encontre em `SEMANTIC_SCHOLAR_API_KEY`
    (D52), e sem ela o 429 diz onde a pedir;
  - o GDELT anónimo continua limitado por IP, sem cura do nosso lado: depois de um 429, as tools
    esperam 60 s e dizem quando voltar a tentar (D53);
  - o User-Agent identifica o toolkit, a versão e o repositório."
