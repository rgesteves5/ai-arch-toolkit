# Internal docs

Historical audits, refactoring plans, and design-philosophy notes that informed
the framework. Useful for context when revisiting a decision; **not part of the
user-facing documentation** and intentionally excluded from `mkdocs.yml`.

| File | What it is |
|------|------------|
| `api-semantics-audit.md` | Naming + semantic drift audit across Client/Provider/Agent/Middleware layers (early 2026). |
| `core_audit.md` | Production-readiness audit of the `core/` layer against `research/python_best_practices.md` (2026-02-25). |
| `refactoring-plan.md` | The plan that aligned the package with the Content / Transform / Identity / Memory primitives. |
| `first_principles_llms.md` | Design-philosophy note on what an LLM is and the small set of primitives the rest reduces to (same text as `from_claude_chat/FIRST_PRINCIPLES.md`). |
| `from_claude_chat/` | Briefing material and sketch APIs from the rewrite session — kept for traceability. |
| `metering-plan.md` | Architecture plan for the neutral meter in `core/` and the budget controller in `toolkit/` (2026-06-28); historical — R01 superseded its failed-call accounting, and the current contract is [Cumulative budgets](../safety.md#cumulative-budgets). |
| `prompt-resource-system-design.md` | Design of the prompt and resource system: loaders, codecs, templates, layouts, manifests (2026-07); implemented, kept as the design record. |
| `phase-config-plan.md` | Plan that restored per-phase LLM, tools and prompts to `Agent`/`ReasoningSpec` and to agent manifests (2026-07); implemented, kept as the design record. |
| `agentes-app-toolkit-review.md` | Review of the toolkit's limitations for the Agentes app, at `48a43ac` (2026-09-12, Portuguese); later sections correct earlier ones. |
| `toolkit-fix-plan.md` | Verified findings from that review and the fix plan F01–F18 (Portuguese); implemented in 2026-09, per-fix records in `blackboard/`. |
| `hardening-plan.md` | Root-cause grouping of the findings open on 2026-09-17 and the proposed structural fixes, verification strategy and order (Portuguese); carried out as front R (R01–R03, done 2026-09-18), decisions D15–D36 in `blackboard/DECISIONS.md`. |
| `tools-contract-plan.md` | Audit of the 132 toolkit tools against what an agent needs (cuts, errors, contracts, output, duplicates) and the structural plan, with decisions D37–D42 (2026-09-29, Portuguese); front T in `blackboard/BOARD.md` since 2026-09-30. |
