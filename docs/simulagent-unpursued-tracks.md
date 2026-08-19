# Simulagent unresolved and unpursued track registry

**Frozen:** 2026-08-19  
**Purpose:** record incomplete, blocked, intentionally excluded, and successor-only work without implying that every
item was required for Simulagent's completed claims.

## Status vocabulary

- **Partially addressed:** some relevant evidence exists, but the stated endpoint was not established.
- **Not pursued:** the project did not run the study.
- **Blocked:** the required independent evidence or role was unavailable.
- **Intentionally excluded:** pursuing the item would violate the project's safety or claim boundary.
- **Successor-only:** potentially valuable, but it should start under a new protocol rather than reopen Simulagent.
- **Historical audit gap:** an old artifact/provenance limitation is preserved rather than reconstructed.

## A. Central scientific premises

| ID | Track | Final status | Why it remains open | Minimum credible reopening condition |
|---|---|---|---|---|
| A1 | Formal theory of semantic identifiability and decision relevance | Partially addressed | The experiments supplied constructive examples and counterexamples, but no general theorem characterizes when language observations distinguish semantic hypotheses or when those distinctions change optimal action. | A standalone theory program with formal latent states, observation kernels, equivalence classes, decision losses, proofs, and executable counterexamples. |
| A2 | Executable fictional-world clarification oracle | Not pursued | The prospective cards specified enough to elicit an initial request but not enough to answer arbitrary follow-up questions. | Precompute complete hidden world states and a deterministic answer function before any request or model output is opened; verify every allowable question is answerable or explicitly unknowable. |
| A3 | Prospective external/new-speaker semantic gold (B2c) | Blocked | One speaker supplied natural requests, but the protocol did not provide independent four-way semantic adjudication or an answerable external world. Existing ontologies and catalogs provide definitions, not the missing speaker-intent observation. | Immutable requester language, catalog/version context, explicit independent record-level adjudication, adequate known/novel/insufficient/unsupported strata, and role separation. |
| A4 | Multi-speaker prospective replication | Not pursued | P001 was a single development participant and the architecture failed before a valid terminal evaluation. | Fresh preregistered population, multiple independent speakers/adjudicators, answerable tasks, contamination controls, minimum counts, and an ethics/privacy plan. |
| A5 | Calibration of semantic uncertainty on prospective language | Not pursued | No independent likelihood or gold population existed from which calibration could be measured. | A frozen externally grounded population with repeated or otherwise estimable semantic outcomes and prespecified calibration metrics. |
| A6 | Real prospective open-world language understanding | Not established | Controlled and retrospective evidence does not cover arbitrary real requests, changing catalogs, hidden facts, or consequential outcomes. | Narrow the domain and test fresh real tasks with independent semantics, controlled consequences, strong baselines, and reproducible outcome measurement. This is successor-only. |

## B. Prospective policy and evaluation

| ID | Track | Final status | Why it remains open | Minimum credible reopening condition |
|---|---|---|---|---|
| B1 | Simplified `ANSWER / CLARIFY / DEFER` model interface | Not pursued prospectively | The seven-field Phase 2 interface was brittle, but simplifying it on the same 16 inspected records would be post-hoc. | A fresh role-separated request set and predeclared compact schema, with no reuse of P001 for confirmation. |
| B2 | Terminal response after clarification | Stopped | Phase 3 stopped after 1/11 because its questions could request undefined fictional facts. | A2 must first pass; then freeze a fresh clarification and terminal-response protocol. |
| B3 | Participant-rated usefulness or task success | Not pursued | No valid terminal responses were produced, and fictional cards did not provide objective task outcomes. | Fresh answerable tasks, prespecified outcome measures, and evaluation separated from generation. |
| B4 | Causal baseline comparison | Not pursued on the prospective pilot | Direct LLM, always-clarify, always-defer, simple deterministic, and oracle-context baselines were not compared on a valid terminal population. | Fresh data and A2/A3; freeze baselines and primary contrasts before opening outputs. |
| B5 | Natural participant-originated real situations | Intentionally deferred | Real situations could make clarification answerable, but introduce privacy, reproducibility, risk, and adjudication problems not covered by the pilot. | Separate protocol with privacy review, risk exclusions, immutable records, objective outcomes, and independent adjudication. |
| B6 | Independent expert or speaker adjudication | Blocked | The assistant cannot substitute its own judgment, model consensus, or simulated people for the independent channel being evaluated. | Recruit qualified independent adjudicators or use an existing resource that already contains the necessary judgments. |

## C. Model and ontology extensions

| ID | Track | Final status | Why it remains open | Minimum credible reopening condition |
|---|---|---|---|---|
| C1 | Additional local scaling, quantization, or API models | Intentionally stopped | Existing runs showed that more capacity and reasoning did not create semantic ground truth. Another model comparison would optimize a downstream component while the upstream evidence premise remained unresolved. | Independent gold plus a meaningful deterministic residual and a prespecified incremental-recall endpoint. |
| C2 | Small/large model ensembles or cascades | Not pursued end-to-end | Combining proposers may improve recall or cost, but cannot independently validate candidates or define the catalog boundary. | Same as C1, plus a fixed candidate budget and deterministic authority layer. |
| C3 | Further reasoning-budget and prompt optimization | Intentionally stopped | Low-effort bounded reasoning reduced truncation; continued tuning risked optimizing the inspected development set and would not fix missing observations. | Fresh task and a hypothesis specifically about compute/reliability, not semantic authority. |
| C4 | Automatic ontology acquisition or registration | Intentionally excluded | Provisional model proposals were never authorized to become trusted concepts; doing so would turn semantic guesses into authority. | Independent validation, versioned provenance, contradiction handling, reversible registration, and explicit human/domain authority in a new project. |
| C5 | Additional ontology domains using only existing resources | Partially addressed | Existing resources can support catalog reconstruction and deterministic concept mechanics, but not prospective speaker meaning when the utterance underdetermines the distinction. | Use them for a clearly retrospective or executable task, or add independent prospective adjudication. |

## D. Safety, execution, and engineering

| ID | Track | Final status | Why it remains open | Minimum credible reopening condition |
|---|---|---|---|---|
| D1 | Real service calls and autonomous execution | Intentionally excluded | Simulagent evaluated shadow planning, reversibility, certificates, and deferral; it did not establish operational safety. | Separate safety case, scoped permissions, sandbox-to-live promotion rules, monitoring, rollback, and domain-specific evaluation. |
| D2 | Public benchmark/package release | Not pursued | The repository contains research infrastructure and a reference architecture, but not a polished supported benchmark with distributable protected participant data. | Data-governance review, redaction/licensing, reproducible install, stable API, and release-specific documentation. |
| D3 | Eight historical dependency drifts | Historical audit gap | The audit diagnosed and preserved drift rather than rewriting hash-linked outcomes. | Recover original bytes or append exact provenance; never silently replace historical dependencies. |
| D4 | Missing historical outcome locks (V58, V76, V77, V93, V99, V222) | Historical audit gap | The cross-track ledger records absent locks; reconstructing them after the fact would overstate provenance. | Only contemporaneous source artifacts or an explicitly labeled append-only reconstruction can narrow the gap. |

## E. Work that belongs elsewhere

| ID | Track | Final status | Why it remains open | Proper destination |
|---|---|---|---|---|
| E1 | Low-resource translation verification with active human queries | Successor-only | It has a different observable object, expert role, loss function, and evaluation population. It may offer the independent channel Simulagent lacked, but its evidence cannot retroactively validate Simulagent. | A new repository/protocol that treats translators or bilingual reviewers as the external channel and measures translation decisions. |
| E2 | Unrestricted everyday/fantasy/social/theological assistant competence | Successor-only | These domains are useful for diversity, but one protocol cannot supply objective semantics and outcomes for all of them. | Separate narrow-domain studies with domain-appropriate ground truth and risk controls. |

## Priority if any idea is revived

Do not begin with another LLM. The only high-value order is:

1. define an independently observable semantic or task outcome;
2. prove the observation can distinguish the hypotheses that matter for action;
3. construct fresh, role-separated records and deterministic/oracle baselines;
4. test the controller and clarification policy without a model;
5. add a bounded model only for a residual the preceding steps cannot solve; and
6. keep all candidates provisional until independently validated.

That sequence is a reusable lesson from Simulagent, not an authorization to continue it.

