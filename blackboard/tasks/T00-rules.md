# T00 · Regras da frente do contrato das tools (valem para T01–T09)

Lê isto antes de pegares numa ficha T. A ficha diz o quê; este ficheiro diz como.

## Contexto

Duas conversas do ai-network mostraram um agente que não chegava ao que as páginas tinham: o motivo
do Nobel da Física de 1960 estava no carácter 13 888 de uma página que a tool cortava aos 4000, e as
secções do Wikibooks, pedidas como subpáginas que não existem, voltaram com `ok: true` e vazias. O
levantamento das 132 tools encontrou as mesmas causas em todo o lado. O contrato, as causas e os
inventários por módulo estão em `docs/internal/tools-contract-plan.md`; as decisões são D37–D42.

## Ler primeiro, por esta ordem

`AGENTS.md` → `blackboard/README.md` → `BOARD.md` (secção desta frente) → este ficheiro → a tua
ficha → `DECISIONS.md` D37–D42 → `docs/internal/tools-contract-plan.md` (secções 2 a 5 e os anexos
dos teus módulos) → `tasks/R00-rules.md`.

## O que vale da R00

Valem sem mudanças as secções de `R00-rules.md` "Regras de desenho", "Proibido", "Dúvidas e
bloqueios", "Verificação" e "Registo". As decisões novas desta frente entram em `DECISIONS.md` a
partir do número livre seguinte (hoje, D43).

## O contrato, em termos de código

Depois das costuras (T01 a T04), uma tool cumpre o contrato quando:

1. **Falha com tipo.** Quando não consegue responder, lança `ToolFailure(type, message)`, com `type`
   em `not_found`, `validation_error`, `upstream` ou `rate_limited` (D42). A mensagem diz o motivo
   da fonte e o passo seguinte ("… does not exist; search with `wiki_search`"). Nunca devolve uma
   string de erro.
2. **Zero resultados é sucesso**, e di-lo com a consulta.
3. **Nenhum corte é beco sem saída.** Todo o corte passa pela janela (`toolkit/tools/_window.py`,
   T03): o rodapé diz o que foi mostrado, o total quando se sabe e a chamada exacta para o resto.
4. **Os limites estão na assinatura** (T04a), e o schema mostra-os; a docstring não os repete de
   outra maneira.
5. **A saída lê-se sem descodificar:** datas em ISO 8601 (UTC), números sem notação científica e com
   unidade, códigos com o rótulo que a resposta já traz, tabelas em linhas.
6. **Cada identificador devolvido é aceite por uma tool.**

A invariante de contrato (T04b) verifica os pontos 1 a 4 em todas as tools; a lista de dívida diz
quais ainda não cumprem.

## Regras próprias desta frente

- **A lista de dívida só encolhe.** Quem migra um módulo apaga as linhas das suas tools e mais
  nenhuma; o coordenador junta as listas ao aplicar.
- **Uma tool por trabalho e por fonte (D41).** Fundir, renomear ou tirar tools é uma quebra: o
  `CHANGELOG` leva "**Breaking:**" e uma linha na tabela de migração das `Upgrade notes`. Sem aliases.
- **Rede.** Os testes simulam a fonte na costura `_http._open` (`tests/toolkit/http_fakes.py`). As
  respostas de erro vêm da documentação oficial de cada fonte; cita o URL na ficha. As verificações
  ao vivo, só a APIs gratuitas e sem chave, são do dono: escreve os comandos no fim da ficha.
- **O ai-network não se toca.** O aviso de cada quebra vai no `CHANGELOG`; o dono passa-o.
- **O `nanope` importa tools pelo nome**
  (`nanope/advanced_multi_purpose_configurable_agent/_tools.py`), e a R00 proíbe tocar-lhe. Se uma
  fusão tirar um nome que ele usa, escreve-o em "Bloqueios" e pergunta ao dono antes de o tirar.
- **Exemplos.** Depois da T05, os módulos da família wiki são o modelo a seguir, e o
  `CONTRIBUTING.md` passa a apontá-los.
