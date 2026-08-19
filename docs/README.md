# Research documentation index

## Canonical current status

### Final closure (2026-08-19)

**Simulagent is closed as an active experimental program.** The
[final closure](simulagent-final-closure.md) records the supported claims and final pilot disposition. The
[unresolved-track registry](simulagent-unpursued-tracks.md) records work that was incomplete, blocked, intentionally
excluded, or better treated as a separate successor project. These two documents and the machine-readable closure at
`outputs/simulagent-final-closure/result.json` supersede all earlier experimental authorization.

No new collection, terminal continuation, model/API run, training, protected-language opening, ontology registration,
trusted-state mutation, service action, or execution is authorized. Historical results and locked pilot records remain
immutable. Documentation and non-mutating reproduction remain allowed.

### Final prospective single-speaker pilot disposition

An external-state change occurred after V224: a human speaker volunteered to author prospective requests and answer
later clarification questions. The [V1 protocol and testing plan](prospective-language-pilot-v1-plan.md) governs the
pilot, and the [participant guide](prospective-language-pilot-v1-participant-guide.md) explains the local workflow.
Phase 1 completed 16/16 requests with a passing integrity audit and zero assistant generations.

The frozen Phase 2 local architecture run then completed but failed its usability gates: 13/16 exact semantic
proposals were structurally valid and one final continuation reached its token cap. The
[Phase 2 plan](prospective-language-pilot-v1-phase2-plan.md) and
[negative result](prospective-language-pilot-v1-phase2-results.md) preserve that outcome. The deterministic controller
still safely covered every record with zero retries, APIs, services, actions, or executions.

A separately locked exploratory Phase 3 was opened for the 11 already-valid, non-fallback clarification outputs. It is
now stopped after one locked `unable_to_answer` / `do_not_know` response. The fictional scenario cards did not define
the additional world facts requested by arbitrary model-generated questions, so the participant was not an independent
observation channel for those facts. The remaining 10 responses will not be collected and no terminal continuation is
authorized.

The V224 conclusions remain the historical basis for this role separation; participant availability changes the
external feasibility premise but does not retroactively create evidence.

The final scientific record includes:

1. [Cross-track evidence synthesis](cross-track-evidence-synthesis-through-v224.md)
2. [Research stopping rule](research-stopping-rule-after-v224.md)
3. [Post-V224 roadmap](research-roadmap-after-v224.md)
4. [Model-free reference architecture](model-free-reference-architecture.md)
5. [Reference architecture integration result](model-free-reference-architecture-integration-results.md)
6. [Dependency-drift provenance addendum](dependency-drift-provenance-addendum-through-v224.md)
7. [Post-V224 consolidation result](post-v224-consolidation-results.md)
8. [Final closure](simulagent-final-closure.md)
9. [Unresolved and unpursued tracks](simulagent-unpursued-tracks.md)

The closure documents supersede every earlier roadmap authorization. A scientifically separate successor may reuse
the reference architecture only under a new frozen protocol and explicit authorization.

## Machine-readable evidence

- `outputs/cross-track-evidence-audit-through-v224/experiment-ledger.json`
- `outputs/cross-track-evidence-audit-through-v224/family-ledger.json`
- `outputs/cross-track-evidence-audit-through-v224/critical-chain.json`
- `outputs/cross-track-evidence-audit-through-v224/reproducibility-audit.json`
- `outputs/cross-track-evidence-audit-through-v224/dependency-drift-provenance-addendum.json`
- `outputs/model-free-reference-architecture/result.json`
- `outputs/model-free-reference-architecture/audit.json`
- `outputs/simulagent-final-closure/result.json`

## Historical roadmap rule

Every file matching `research-roadmap-after-v*.md` except `research-roadmap-after-v224.md` is an immutable historical
snapshot. Its successor authorization was conditional at the time and is now either completed, failed, superseded, or
closed. Do not execute a historical "next step" unless the canonical stopping rule is first replaced by a newly frozen
prospective authorization after an eligible external-state change.

The historical files are intentionally not edited to add banners because many are hash-linked dependencies of frozen
outcomes. This index makes their status explicit without creating new reproducibility drift.

## Claim hierarchy

| Evidence class | What it supports | What it does not support |
|---|---|---|
| Controlled/model-free mechanism | Exact belief, inquiry, sandbox, certificate, robustness, and defer behavior in the frozen construction | External natural-language validity |
| Fresh procedural confirmation | Repetition of the same mechanism on role-separated project-authored populations | Human or speaker semantics |
| Protected finite-menu model confirmation | Non-authoritative proposal reduction behind a trusted controller | Model confidence as posterior; open-world authority |
| External retrospective reconstruction | Published catalog/version state reconstruction | Prospective speaker intent |
| Metadata/source census | Whether an evidence-acquisition protocol is feasible | The missing semantic labels themselves |
| V224 stopping result | No currently audited source supplies the four-way record population | General impossibility of future external semantic evidence |

## Allowed maintenance

- reproduce and annotate non-sensitive artifacts;
- retain append-only provenance for hash drift;
- maintain the model-free reference architecture and its software tests;
- improve navigation without rewriting frozen historical evidence; and
- monitor for a genuinely eligible external evidence source or qualified speaker/expert channel.
