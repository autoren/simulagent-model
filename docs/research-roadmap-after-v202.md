# Research roadmap after V202

## Post-V204 update (2026-08-19)

V203 found no genuinely partition-independent population covering all 14 frozen contracts, so Track B1 is parked as
positive development evidence rather than reopened with weaker independence. V204 then tested the first Track C
model-free open-world oracle. It was structurally informative but scientifically negative: exact open-world planning
beat MAP, posterior sampling, myopic control, and open-loop control, yet never deferred and had zero forced-commit
regret.

The cause was a terminal-accounting defect in the preregistered process, not an access or inference failure. A repair
had zero immediate reward and was charged only by a later settlement action, so a last-step repair could escape its
consequence when the finite horizon ended. V204 is frozen without tuning. Track C now requires a separately locked,
terminally proper fixed-stage oracle in which every repair receives an unavoidable automatic settlement and every
unfinished sensing trajectory receives the safe-deferral terminal value. See
`docs/v204-open-world-semantic-pomdp-oracle-results.md`.

## Post-V205 update (2026-08-19)

V205 repaired terminal accounting in a separately preregistered fixed-stage process and passed every gate. The exact
policy uniquely calibrated at the root, inspected after red and blue calibration evidence, deferred after green,
and reached both state-specific repairs. Its value was `2.13324`; normalized advantage over immediate deferral and
best open loop was `0.0344436667`. Closed-world and forced-commit policies each had `0.0149436667` normalized regret,
MAP had `0.105277`, and persistent posterior sampling had `0.084027`.

All seven exact-policy terminal paths were accountable: four mandatory automatic settlements and three safe
deferrals, with zero unsettled or horizon-escape paths. This establishes the model-free mechanism, not an empirical
LLM likelihood or external-language result. The active Track C successor is therefore a separately preregistered
metadata/source feasibility audit. It must seek a fresh external analogue that natively supplies action-dependent
reference and target sensing, an outside/invalid semantic or sensor regime, a safe defer option, delayed
state-dependent consequences, and machine-validatable likelihoods or an exact simulator. A positive audit still
cannot open language or run a model without another lock.

## Post-V206 update (2026-08-19)

V206 screened six fresh external families using only pinned official README and license metadata. None documented the
complete V205 analogue. The required ingredients were distributed across separate families: AgentAbstain documented
safe abstention and irreversible/runtime consequences; OpenAgent and Theory of Space documented active interaction;
MIntRec2.0 documented out-of-scope recognition; and FailureSensorIQ documented industrial sensor semantics. No source
also documented an in-episode reference/calibration pathway and an exact generative likelihood or simulator.

Strict external confirmation of V205 is therefore deferred rather than synthesized from unrelated benchmarks. A new,
separate Track F may study **behavioral LLM abstention** using deterministic paired labels and a shadow/no-execution
controller. That track cannot be described as V205 likelihood validation, posterior calibration, or ontology
acquisition. Its first gate is metadata/schema feasibility: verify license, immutable data revision, paired task
identity, deterministic pre-execution labels, label/text separation, a no-execution subset, and contamination controls
before any task language is opened or model is run.

## Evidence update

V202 converted V201's presentation-sensitivity result into a controller-level question. Using only frozen normalized
proposals, it compared four preregistered clarification summaries without reading language or loading a model.

The one-call `SINGLE_PRESENTATION_TOP3_FAMILY` qualified and was selected. Its worst-presentation primary cost was
`0.2141666667`, macro cost was `0.2119047619`, and its minimum improvement over matched `CHAR_LAST` was
`0.0441666667`. Target-hit disagreement across canonical, order-only, and opaque-ID presentations was
`0.0119047619`; mean per-record cost range was `0.0023809524`. The complete target universe remained available and
trusted answers produced exact completion with zero false terminals.

The cheaper one-call top-1 family failed one gate: target inclusion changed across presentations on 9 of 84 records
(`0.1071428571` versus the `0.10` maximum). Three-call plurality and consensus policies qualified, but the fixed
one-call-first rule rejected them because a one-call policy was already adequate.

## Revised language-interface claim

The supported mechanism is now:

> A bounded local LLM may propose three contracts for one trusted top-3-plus-`OTHER` clarification question. The
> proposal can reduce question cost even though lower ranks are not stable posterior mass. The full authoritative
> universe remains intact, and only a trusted answer can produce an exact decision.

This does not establish unrestricted open-world recognition, calibrated uncertainty, ontology truth, human-answer
reliability, authority, action, or execution.

## Research-track map

| Track | Status | Next evidence gate |
|---|---|---|
| A. Exact inference, active identification, and verified planning | Established through the earlier external finite-state sequence | Use as the exact reference architecture; no need to reopen completed baselines |
| B1. Finite-menu language proposals | Parked as positive development-only after V203 found no exact independent 14-contract population | Reopen only if a genuinely independent source exists; do not weaken contract or partition independence |
| B2. Open-world ontology acquisition | Deferred | Requires independently validated target semantics and sandboxed proposal validation; model ranks cannot register concepts |
| C. Richer semantic POMDP | V205 positive model-free mechanism; V204 negative terminal-accounting predecessor preserved | Audit fresh external sources for a native analogue before any language, candidate, or model evaluation |
| D. Human clarification channel | Deferred | Requires actual independent human evidence; model- or script-generated replies remain simulation only |
| E. Additional model/API scaling | Paused | Reopen only for a preregistered mechanism question that cannot be answered model-free |
| F. External behavioral LLM abstention | Candidate identified, not yet opened | Audit AgentAbstain metadata/schema for a deterministic paired pre-execution, no-execution subset; keep this distinct from V205 likelihood validation |

## Active V203: confirmation-feasibility audit

Before opening any new language, audit the pinned source identities and schemas for a truly independent population for
the same 14 semantic capability contracts. The audit must be fixed before its census and must require:

- exact semantic-contract identity rather than intent-name resemblance;
- coverage of all 14 targets and a fixed minimum number of unique dialogues per target;
- train/test partition independence from the V191 development population where available;
- zero overlap with every V183 and V191 source record or dialogue;
- deterministic one-target source-annotation linkage and complete target expressibility;
- independently authored language provenance and a reproducible license/source revision; and
- zero utterance text, protected-language, model, API, training, action, or execution access.

Previously audited alternatives that lack deterministic schema/state linkage or the exact 14-contract universe do not
become eligible by relabeling similar intents. If no source family passes all gates, freeze the negative availability
result and park B1 as development-only.

## Conditional successor

If V203 finds a qualifying population, separately preregister the exact canonical top-3-plus-`OTHER` confirmation:
same local model, prompt, low-effort two-phase budget, parser, menu, costs, gates, and trusted fallback. No tuning may
occur from the feasibility census.

If V203 is negative, proceed directly to Track C. Construct a small model-free semantic POMDP in which:

1. at least two semantic codebooks or admissible interpretations remain plausible;
2. different interpretations imply different hidden-state beliefs and later control actions;
3. sensing choices have different likelihoods, costs, and history-dependent value;
4. a wrong control action causes a delayed state-dependent penalty;
5. exact Bayes-adaptive, MAP, posterior-sampling, myopic, and open-loop policies are all evaluated; and
6. an oracle design census proves nonzero disagreement and regret before protected candidate evaluation.

Uncertainty must come from validated likelihoods or explicit admissible sets, never uncalibrated LLM rank positions.

## Durable boundaries

- Preserve V201's negative invariance result and V202's selected controller exactly.
- Do not reuse V198 protected language as fresh confirmation.
- Do not treat rank order, top-3 membership, or ensemble agreement as posterior probability.
- Keep `OTHER`, the full target universe, and trusted exact completion.
- Freeze source availability, population selection, model policy, costs, and gates before their corresponding evidence.
- No API, training, ontology mutation, trusted-state mutation, real service call, external side effect, action, or
  execution is authorized.

## Post-V207/V207r1/V207r2 update (2026-08-19)

V207 stopped before a scientific result because its Hugging Face dataset-tree request exceeded the endpoint's
1,000-item page limit. The failure and all diagnostic metadata reads were recorded; no task payload or language was
opened. V207r1 then locked a transport-only repair with bounded cursor pagination while retaining every V207 source,
pair, threshold, contamination, language-firewall, and no-execution rule.

The repair passed transport integrity over three pages and 1,586 file paths, but the original scientific feasibility
gate failed. Nonlanguage tree metadata exposed zero deterministic should-act/should-abstain pairs and zero
pre-execution pairs. The allowed code schema exposed identity, prompt, scenario, and explanation fields, but no
independent gold act/abstain field. AgentAbstain therefore cannot support the preregistered text-blind population
selection and deterministic scoring path without either reading task content to infer labels or using an LLM judge.
Neither is allowed.

The first V207r1 outcome verification caught a post-lock update to this living roadmap. V207r2 preserved that failed
audit, restored the locked roadmap bytes, and verified the already stored result without network access, task-language
access, source mutation, or a scientific/model rerun. Its verified outcome lock is
`156610afd53c9a10f9f7a55e89381093ba09fd8f523d8f6f140d61eb785e08ea`.

Track F is now split:

- **F1 external source eligibility:** search only source documentation and nonlanguage schema for a benchmark with
  explicit machine-readable pair ID, phase/scenario ID, and deterministic act/abstain gold labels separated from
  prompts and rationales. Preserve immutable revision, license, paired controls, and shadow/no-execution support.
- **F2 behavioral local-model evaluation:** remains unauthorized until F1 passes and a separate exact extraction
  lock freezes identifiers, prompt projection, label firewall, parser, costs, model/runtime, and gates.

Do not reopen AgentAbstain by weakening the label-independence gate. V207r1 is a source-interface negative, not an LLM
performance result and not a failure of the V205 semantic-POMDP mechanism. The next active branch is a focused fresh
F1 source-availability census; if no source passes, park external behavioral abstention and return to model-free
decision/interface work rather than synthesizing external confirmation.

## Post-V208 update (2026-08-19)

V208 prospectively froze six fresh external candidates and read only pinned repository/dataset metadata and recursive
file paths. No README/blob body, dataset row, task language, rationale, response, model output, or execution trace was
opened. None of OverSearchQA, AbstentionBench, AbstainEQA, HiL-Bench, ClarifyCodeBench, or Ask or Assume passed the
complete source gate.

The component split is now clear. OverSearchQA and AbstentionBench provide text-only abstention labels but not matched
pair identity. AbstainEQA provides a strong paired construction but depends on visual/embodied state and exposes no
pair-ID field in the allowed metadata. HiL-Bench and Ask or Assume provide matched modes but require runtime code/data
interaction and model-mediated help. ClarifyCodeBench contains only underspecified positives for this binary question
and judges clarification matches with an LLM. Several sources also lack recognized code or dataset license metadata.

Track F external behavioral abstention is parked. Do not open payloads, synthesize pair IDs, infer truth from task
language, or replace gold with an LLM judge. Outcome lock
`17e2e934580d4146bf8f41e078341919b38dcce4ef9464ef2a8a65cdc295d6b1` preserves the negative.

The next active branch is **Track G: controlled probabilistic language observations for exact decision-making**. Its
first experiment must be model-free. Build a terminally proper finite grammar whose latent semantic regime includes a
valid canonical codebook, an alternative valid codebook, and an outside/unknown regime. Clarification actions must
induce known history-dependent utterance likelihoods; final state-dependent actions must settle automatically; and
safe deferral must remain available. Exact planning must be compared against closed-world, MAP, posterior-sampling,
myopic, and open-loop controls under preregistered value and regret gates.

Track G is not natural-language validation. It establishes the language-channel decision contract needed before an
LLM can be inserted. Only after the exact channel passes may a separate fresh language population test a local LLM as
a non-authoritative observation mapper. Model scores or ranks must not be treated as likelihoods without an independent
calibration design, and the planner, ontology authority, trusted evidence, and executor remain deterministic.

## Post-V209/V209r1 update (2026-08-19)

V209 prospectively froze the Track G controlled-language mechanism. Its six unit tests passed, but the first oracle
attempt stopped before a scientific result because the kernel validator hard-coded three regimes and therefore
rejected the intentionally two-regime closed-world comparator. The full exact policy was computed only in memory; no
comparator result, regret, gate decision, summary, or result was produced. V209r1 separately locked the sole repair:
infer a positive regime dimension while retaining exactly two task states, three observations, and all original
normalization, support, cost, history, grammar, reward, comparator, and gate requirements.

The repaired oracle passed every gate. Exact planning selected `ask_reference` at the root with value `2.0268896`,
selected `ask_target` after `UTTERANCE_ALPHA` and `UTTERANCE_BETA`, and deferred after
`UTTERANCE_UNRESOLVED`. Both task-specific final actions were reachable. Four paths ended in mandatory automatic
settlement and three in safe deferral, with zero unsettled or horizon-escape paths. Exact planning beat immediate
deferral and best open loop by `0.0335574133` normalized units. Closed-world and forced commitment each incurred
about `0.01498855` regret, MAP `0.10439075`, posterior sampling `0.08314075`, and myopic control `0.10022408`.

The two matched surface families and an `ALPHA`/`BETA` vocabulary permutation produced exactly zero policy mismatch
and zero value change. This establishes the finite probabilistic language-channel decision contract only. It does not
establish LLM understanding, empirical likelihoods, natural-language robustness, or calibrated confidence.

The active successor is **V210: fresh controlled-language population and deterministic projection feasibility**.
Before generating or opening any population records, preregister:

- a deterministic generator from hidden state, semantic regime, clarification action, and history to semantic
  observation ID plus surface realization;
- development and protected seeds/identifiers fixed by hash, with protected surfaces sealed;
- held-out compositional surface constructions rather than mere sentence duplicates;
- matched direct/paraphrase and opaque-renaming counterfactual groups;
- a deterministic, non-model projection baseline and exact semantic truth kept outside the text interface;
- population balance, uniqueness, leakage, coverage, round-trip, normalization, and no-execution gates; and
- an explicit rule that a positive V210 authorizes only a separate deterministic baseline evaluation, not a model.

Do not add an LLM at population construction time. A model may become a later shadow observation mapper only after
the fresh population, deterministic projection, baseline residual, model/runtime policy, parser, and conservative
fallback are each separately frozen.

## Post-V210 update (2026-08-19)

V210 generated and independently regenerated a fresh 540-record controlled-language population: 270 development and
270 protected records, each organized as 90 groups of matched direct, paraphrase, and opaque-renaming surfaces. Each
role exactly covers three regimes, two task states, five clarification contexts, and three semantic observations.
Surface and truth artifacts are separate. All factor keys and record IDs are unique; role identifiers, groups,
normalized surfaces, template skeletons, and label lexicons have zero overlap. Counterfactual truth/probability and
opaque-renaming mismatches are zero, every probability group normalizes, and regeneration is byte exact.

The truth-blind explicit-marker projector accepted 90/270 development records with 100% accuracy and zero false
acceptance, then abstained on 180 paraphrase/opaque records. The residual is prediction-defined and still covers every
regime, state, context, and observation. Protected text was automatically integrity-audited but had zero baseline or
manual reads. Outcome lock `8b7b5e80a658fb9ecdb0a0dfa8fe56862ce13a1aa80681f4821851879f50c398`
freezes the result.

The active successor is **V211: deterministic residual baselines and exact decision impact**. It must stay on the 180
development residual and freeze all rules before reading residual truth. Compare at least:

- lexical normalization over the development paraphrase and opaque vocabularies;
- a template-independent compositional parser using context and response spans separately;
- abstention-first consensus between independent deterministic views; and
- the existing `ABSTAIN` projector as the safety control.

Predictions must be frozen before joining truth. Evaluate semantic-observation exactness, counterfactual agreement,
false acceptance, abstention, and downstream V209 policy/value impact under a fixed conservative rule. Define any
model-eligible residual from predictions alone and require it to retain meaningful factor coverage. V211 may not read
protected surfaces or run a model. Only a positive, nontrivial residual with a prospectively defined decision-impact
question can authorize a separate low-effort local-model design.

## Post-V211/V211r1 update (2026-08-19)

V211 froze a group-disjoint 90-record calibration and 90-record evaluation split over the V210 development residual.
The prediction worker received calibration surfaces and truth but evaluation surfaces only; evaluation group IDs,
truth, hidden factors, probabilities, protected artifacts, models, APIs, and execution remained unavailable.
Predictions were hashed before the sealed evaluation truth was joined for scoring.

The first V211 run was a technical bookkeeping failure: the compositional rows were emitted under the obsolete name
`CONTEXT_CONTRAST`, so the exact prediction-count invariant failed before any outcome lock. V211r1 prospectively
froze a name-only repair to `COMPOSITIONAL_RESPONSE_SPAN`. It reread no calibration data, changed no prediction
value, and proved that the normalized parent artifact and repaired artifact were otherwise identical.

The repaired result closes the proposed model branch for this population. Raw lexical normalization, compositional
response-span parsing, and abstention-first consensus each accepted all 90 evaluation records with 100% exact semantic
accuracy, zero false acceptance, zero counterfactual disagreement, and numerical-zero normalized V209 continuation
regret (`6.381542649304128e-19`). Abstain-always retained all 90 cases and incurred `0.03000790939597292` regret.
The V210 residual was therefore a withheld fixed-lexicon problem, not evidence that a language model was needed.
Outcome lock `f984831958c24bf0415b983b01337ae4e47333091fb4a3c50f30749b6ffe70a3` freezes zero model eligibility.

The active successor is **Track G2/B2a: model-free representational diagnosis**. The initial V212 draft treated
`OUTSIDE_UNKNOWN` as one semantic outcome. That is too coarse: it conflates an existing primitive, an existing
composition, a meaning requiring a missing DSL operator, a meaning irreducible relative to the frozen diagnostic
language, and lack of identifying evidence. The draft was superseded before audit, lock, or execution.

Revised V212 must enumerate executable behavioral hypotheses and collapse them into equivalence classes. It must cross
two orthogonal axes:

- expressibility: existing primitive, existing composition, missing-operator representation gap, or irreducible
  provisional extension relative to the frozen diagnostic language; and
- evidence: sufficient, ambiguous, or contradictory.

It must generate exact boundary witnesses between distinguishable equivalence classes, prove evidence sufficiency and
necessity, preserve arbitrary-symbol grounding and renaming/order invariance, and attach non-authoritative decision
consequences: reuse, compose, diagnose a representation gap, retain a provisional shadow candidate, clarify, or defer.
False primitive creation and false merging are separate noncompensatory failures.

If V212 passes, **V213** may generate a fresh role-separated programmatic concept population with hidden executable
semantics, definitions, positive/negative/boundary evidence, counterfactual renamings, and protected concept groups.
If V213 passes, **V214** must run retrieval, controlled parsing, bounded synthesis, exact version-space filtering,
equivalence collapse, boundary-witness generation, contradiction handling, and complete-safe-retention controls.

An LLM remains ineligible unless V214 leaves a nontrivial, factor-complete, behaviorally identifiable, and
decision-relevant residual defined without protected truth. A first model study may add only a bounded set of typed
candidate programs and must improve oracle-equivalence-class recall at a fixed candidate budget. Existing sandbox and
certificate-aware planning mechanisms will be reused only after genuine surviving candidates exist; they will not be
rebuilt as a substitute for candidate evidence.

After the V212 schema is frozen, a separate metadata-first census may seek versioned external ontologies or alignment
benchmarks for a retrospective temporal-holdout confirmation. Existing curated resources can support recovery and
safe staging claims, but not prospective discovery of correct new human meaning. No model, language population,
external payload, registration, authority, action, or execution is authorized by this roadmap update.
