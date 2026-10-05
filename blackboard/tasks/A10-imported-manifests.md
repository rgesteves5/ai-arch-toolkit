# A10 · Os manifestos importados: os aliases do YAML e o aninhamento com limite (G-34)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 10 (G-34) · **Decisões:** D59 ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `e199cc8`)

- O `yaml.safe_load` partilha o nó de um alias, sem o copiar. Mas o `_canonical_config` do
  `load_agent_manifest` percorre a árvore já expandida.
- 368 bytes, com cinco níveis de dez aliases, levam 1,7 s e 92 MB, e cada nível multiplica por
  dez. A revisão da E02-05 do ai-network mediu 412 bytes com seis níveis: 2,6 s e 170 MB.
- O `YamlCodec` dos recursos (por onde passam os manifestos de prompt e o conhecimento) também
  faz um `yaml.safe_load` sem limite. Um prompt só escapa porque a forma falha antes de alguém
  percorrer os dados.
- Um ficheiro de 4 KB com 2000 níveis de aninhamento levanta um `RecursionError` cru, em YAML e
  em JSON, tanto no manifesto de agente como no de prompt.
- Um alias recursivo (`&a [*a]`) já é recusado pelo `_canonical_config`, mas só nos manifestos de
  agente.

## Nota de desenho (antes do código)

- **Um sítio só:** `toolkit/_safe_data.py` (novo) lê o YAML, o JSON e o TOML que o toolkit
  carrega de ficheiros: os manifestos de agente e os codecs dos recursos, por onde passam os
  prompts e o conhecimento. Uma recusa é um `UnsafeDataError` (um `ValueError`), que cada sítio
  converte no seu erro (`AgentManifestError`, `ResourceDecodeError`).
- **Os aliases limitam-se, não se proíbem.**
  - O YAML compõe-se com o `SafeLoader`. Antes de construir os dados, um percurso iterativo do
    grafo de nós conta os nós distintos e os da árvore expandida.
  - Os aliases podem acrescentar até 10 000 nós, ou tantos quantos o documento tem, o que for
    maior. As âncoras e as merge keys (`<<: *base`) continuam a servir.
  - Um alias recursivo é recusado.
- **O aninhamento tem limite:** 100 níveis de mapas e listas, em qualquer formato. O
  `RecursionError` do parser passa a ser a mesma recusa. Nenhum percurso recursivo do toolkit
  chega perto do limite do Python.
- **Provas:**
  - a bomba de seis níveis é recusada depois de ler o ficheiro, em menos de um segundo, por
    `load_agent_manifest`, `load_prompt` e um recurso YAML, com uma mensagem que fala dos aliases;
  - um manifesto com uma âncora e uma merge key carrega, com os valores certos;
  - o limite é exacto: no limite passa, e um nó acima é recusado;
  - um alias recursivo é recusado no codec;
  - 2000 níveis em YAML, JSON e TOML dão o erro do sítio, não um `RecursionError`. 100 níveis
    passam e 101 não.

## Ficheiros

- `src/ai_arch_toolkit/toolkit/_safe_data.py` (novo), `toolkit/agents/_manifest.py`,
  `toolkit/resources/_codecs.py`
- `tests/toolkit/test_safe_data.py` (novo), `tests/agents/test_manifest.py`,
  `tests/resources/`
- `docs/agents.md` ou `docs/prompts.md` (onde falam do YAML), `docs/safety.md`, `AGENTS.md`,
  `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-05), pela nota de desenho (D59), alargada depois da revisão.
- **`toolkit/_safe_data.py`:**
  - `load_yaml`: um `SafeLoader` que conta os níveis enquanto compõe e pára no 101.º, sem esperar
    pelo `RecursionError`. Depois, o `_check_nodes` percorre o grafo de nós sem recursão. Para
    cada nó, o `_build` dá os nós, os caracteres, a profundidade e a cadeia de merge keys do que
    ele dá depois de expandido e fundido. Um ciclo é recusado;
  - os aliases podem acrescentar até 10 000 nós e 1 000 000 de caracteres, ou o tamanho do
    próprio documento;
  - `load_json`/`load_toml`: o `RecursionError` do parser é a mesma recusa. O `_check_tree` vê a
    árvore nível a nível;
  - `check_depth(data, above=)`, com memória por objecto, para os valores que a app passa;
  - `MAX_ALIAS_NODES`, `MAX_ALIAS_CHARS`, `MAX_DEPTH`, `UnsafeDataError`.
- **Quem o usa:** o `_read_manifest` dos manifestos de agente, o `_apply_overrides` (o caminho e
  o valor de um override) e os codecs JSON, TOML e YAML dos recursos.
- **As heranças:** o `_load_merged` dos manifestos de agente e o `_load_prompt` dos de prompt
  guardam o que já construíram por caminho (`loaded`), com a altura de cada um. O limite de
  herança e de inclusão conta-se por todos os caminhos.
- **O teste de arquitectura:** `test_yaml_is_read_only_by_the_safe_data_module` só deixa importar
  o PyYAML nos módulos que lista, e de cada um só os nomes que lista. Vê o `from yaml import`, os
  aliases do módulo e o `getattr`.
- **Revisão independente** (um agente sobre o diff, com fuzzing): o mecanismo estava certo, e os
  limites e a paridade com o `safe_load` foram verificados. Corrigido o que apontou:
  - **alta:** um alias para um escalar longo contava um nó só. 139 KB davam 3,3 s e 2 GB no
    `load_agent_manifest`, e 89 KB davam 512 MB num recurso. Daí o limite de caracteres;
  - **média:** o `extends` e o `include` multiplicavam as leituras. Dez manifestos de 757 bytes
    davam 349 525 leituras em 79,5 s; agora são 10. Oito prompts de 606 bytes davam 21 845
    leituras; agora são 8;
  - o detector de arquitectura fugia-se com o `from yaml import` e com aliases;
  - as merge keys contavam como nível;
  - os overrides passavam dos 100 níveis (e davam `RecursionError`);
  - as afirmações largas de mais na documentação ("todo o ficheiro", "a regra do go-yaml");
  - o self-alias nas notas de actualização;
  - o `load_json` lento.
- **Medido:**

  | Caso | Antes | Agora |
  |---|---|---|
  | Bomba de seis níveis, 418 bytes (manifesto de agente) | ×10 por nível sobre os 1,7 s e 92 MB dos cinco níveis | recusada em 0,03 s |
  | Alias para um texto de 100 KB, 139 KB | 3,3 s e 2 GB | recusada |
  | 2000 níveis em YAML | `RecursionError` em 0,8 s | `AgentManifestError` em 0,26 s |
  | 2000 níveis em JSON | `RecursionError` | `AgentManifestError` em 0,00 s |
  | YAML de 200 000 escalares | 2,87 s | 3,27 s |
  | JSON de 15 MB | 0,18 s | 0,30 s |
- **Testes:** `tests/toolkit/test_safe_data.py` (34), e dois de arquitectura.
- Gate: 6477 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 10 fechou.
  `load_agent_manifest`, `load_prompt` e os recursos lêem o YAML, o JSON e o TOML por um só
  sítio, que limita o que os aliases expandem: 10 000 nós e 1 000 000 de caracteres, ou o tamanho
  do próprio ficheiro.

  Nenhum ficheiro passa de 100 níveis, nem um override com o seu caminho. Um manifesto que
  vários herdam ou incluem lê-se uma vez. As âncoras e as merge keys continuam a servir.

  Na app, o `agents/bundles._safe_yaml` pode sair, ou ficar mais estrito: a app recusa âncoras e
  mais de 32 níveis, e o toolkit não as recusa."
