# Internal docs

Historical audits, refactoring plans, and design-philosophy notes that informed
the framework. Useful for context when revisiting a decision; **not part of the
user-facing documentation** and intentionally excluded from `mkdocs.yml`.

| File | What it is |
|------|------------|
| `api-semantics-audit.md` | Naming + semantic drift audit across Client/Provider/Agent/Middleware layers (early 2026). |
| `core_audit.md` | Production-readiness audit of the `core/` layer against `research/python_best_practices.md` (2026-02-25). |
| `refactoring-plan.md` | The plan that aligned the package with the Content / Transform / Identity / Memory primitives. |
| `first_principles_llms.md` | Design-philosophy note on what an LLM is and the small set of primitives the rest reduces to. |
| `from_claude_chat/` | Briefing material and sketch APIs from the rewrite session — kept for traceability. |
