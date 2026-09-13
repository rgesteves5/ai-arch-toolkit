# F12 · Schema: uniões multi-tipo e varargs

- **Dono:** worker-C · **Estado:** done · **Depende de:** —
- **Plano:** L, F12

## Problema

- `src/ai_arch_toolkit/core/_tools/_schema.py::_hint_to_json_schema` devolve `({"type": "string"}, True)`
  para qualquer união com mais de um tipo não-`None`: `def procura(consulta: int | str)` →
  `required=[]`, e o modelo que omite o argumento (legal segundo o schema) recebe `validation_error`.
- `infer_schema` põe `*args`/`**kwargs` em `properties` e `required`: `def f(a, *args, **kwargs)` →
  `required=['a','args','kwargs']`; o modelo que obedece faz a função receber
  `args=() kwargs={'args': 2, 'kwargs': 'x'}`.

## Mudança

- União multi-tipo → `{"anyOf": [schema de cada membro não-None]}`; opcional só quando `None` faz
  parte da união. Retirar o aviso "collapsed to string". `X | None` fica como está.
- Uniões dentro de `list[...]`, dataclasses e TypedDicts seguem a mesma regra.
- Parâmetros `VAR_POSITIONAL`/`VAR_KEYWORD` nunca entram no schema.
- Compatibilidade com os providers: provar que cada adaptador aceita uma tool cujo schema tem `anyOf`.
  Anthropic, OpenAI e xAI passam o dict tal como está. O Gemini faz
  `types.FunctionDeclaration(parameters=schema)` (`src/ai_arch_toolkit/core/_providers/_gemini.py:206-213`):
  construir a declaração num teste; se o SDK instalado rejeitar ou perder o `anyOf`, passar a usar
  `parameters_json_schema` (se existir nessa versão do `google-genai`) e registar aqui.

## Ficheiros

- `src/ai_arch_toolkit/core/_tools/_schema.py`
- `tests/test_tools_schema.py`
- Se necessário: `src/ai_arch_toolkit/core/_providers/_gemini.py`, `tests/test_gemini_provider.py`

## Prova

- `f(q: int | str)` → `required == ["q"]` e `properties["q"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}`.
- `f(q: int | str | None = None)` → não obrigatório, `anyOf` de int e str.
- `f(a: int, *args: int, **kwargs: str)` → `properties` só com `a`, `required == ["a"]`.
- `list[int | str]` → `items` com `anyOf`.
- Uma declaração Gemini construída a partir de um schema com `anyOf` preserva as variantes.

## Registo do dono (transcrito pelo coordenador; o worker não conseguiu escrever fora da worktree)

- Estado: done — feito pelo worker-C na worktree `.claude/worktrees/agent-abbcbffc136b5debb`.
- Ficheiros tocados: `core/_tools/_schema.py`, `tests/test_tools_schema.py`.
- Testes novos: 8 em `TestHintToJsonSchema`, 5 em `TestInferSchema`, 6 em `TestAnyOfReachesProviderAdapters` (Anthropic, OpenAI, xAI e Gemini, incluindo o corpo do pedido do SDK real do Gemini). Todos falhavam em 48a43ac.
- CHANGELOG (Fixed): uniões multi-tipo passam a `anyOf` obrigatório (opcional só com `None`); membros com schema idêntico colapsam; `*args`/`**kwargs` nunca entram no schema.
- Desvios e achados: `_gemini.py` sem mudança (`FunctionDeclaration(parameters=...)` preserva `anyOf` no google-genai 2.11.0). Achados C1, C2 e C3 em `FINDINGS.md`.
- Nota do coordenador: `test_multi_type_union_with_none_is_optional_any_of` falhava na suite integrada por dependência de ordem do cache do `typing` em Python 3.13 (`Union[int, str] == Union[str, int]`); o teste passou a comparar as variantes sem ordem.
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
