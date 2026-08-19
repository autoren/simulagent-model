# Controlled Open-World Language Research Direction

The current evidence supports one constrained next step. V111 found that a frozen local model's typed
`ABSTAIN` response transfers as a useful but incomplete novelty signal: 80% precision and 50% recall on a
separate development subset. This signal is evidence, not a semantic decision. It cannot define a novel
capability, select an action, or authorize execution.

The next branch must preregister a complete development-only policy before evaluating it. The policy should
combine the frozen direct typed response, the frozen `ABSTAIN` novelty trigger, deterministic schema and
retrieval evidence, and a fail-closed ask/inspect outcome. It must retain all safe known and novelty
hypotheses and be judged jointly on known, novel-valid, unsupported, and insufficient-evidence behavior,
including calibration, selective risk, false-known acceptance, and downstream exact-planner regret.

Only a transferred full-policy pass may justify a separately locked protected-test protocol. Protected-test
access itself cannot authorize schema induction. Typed schema or mechanic induction remains a later,
sandboxed stage with deterministic type, precondition, effect, and intervention checks. A richer sequential
decision problem remains later still. Local or API models remain non-authoritative throughout, with zero
real service calls or external effects.

V112/V112r1 confirmed the narrow novelty signal on a genuinely fresh 192-record MASSIVE transfer set.
Typed abstention reached 70.21% novelty precision, 68.75% recall, a 9.72% non-novel false-positive rate,
and 0.0494 ECE. The full validation policy reduced mean regret to 0.8047 and false-known acceptance to
6.25%, but missed the known-accuracy gate (78.13% versus 80%) and top-80%-error gate (22.08% versus 20%).

Accordingly, the protected test and induction remain closed. The next branch must preserve the frozen
abstention novelty signal while prospectively improving known-request coverage and selective ranking on a
new population. It may not retune V112 or reinterpret its two failed gates.

V113 used V112 strictly as historical policy-design evidence. Among 239 frozen rescue rules, 80 cleared all
seventeen inherited gates. The selected rule accepts a disagreeing known proposal only when its retrieval
score is at least 0.60 and its gap behind the nearest intent is at most 0.15. Historically it rescued four
records, raised known accuracy to 82.29%, lowered top-80% error to 19.48%, and left novelty and false-known
metrics unchanged. This rule now requires evaluation on a new disjoint population; it is not yet a reason
to open the protected set or begin induction.

V114 supplied that record-disjoint evaluation on 192 unused MASSIVE test-partition records while excluding
all V101—including sealed protected—and V112 identifiers. The abstention novelty signal reached 68.75%
precision and 68.75% recall, but one additional false positive produced a 10.42% non-novel false-positive
rate. It therefore missed both frozen novelty precision and false-positive gates. The base policy also
failed exact-decision, known-accuracy, and selective-error gates.

The rescue rule triggered three times: two known corrections and one more-costly false-known acceptance on
a novel-valid request. Known accuracy improved from 73.96% to 76.04%, but false-known acceptance increased
from 9.38% to 10.42% and regret from 0.9688 to 1.0026. All triggers were concentrated in one scenario. The
formal mechanism conclusion is upstream-inconclusive because novelty failed, while the paired evidence is
independently unfavorable and does not support retaining or retuning the rule.

Freeze V114 negative. V112 remains positive evidence on its validation transfer population, but typed
abstention is not established as a stable discriminator beyond it. Keep the original protected set,
sandboxed induction, and richer sequential planning closed. The only justified successor is a separately
preregistered development branch that introduces genuinely new contrastive or multi-turn evidence while
preserving complete safe hypotheses, explicit uncertainty, deterministic validation, and zero model
authority or execution. Do not mine V114, relax gates, scale or ensemble models, use an API, or train an
adapter under this branch.

V115 tested the static contrastive alternative on another 192 balanced, record-disjoint MASSIVE test
records. The same local model first used the exact V112 typed-choice interface, then challenged the reviewed
candidate against the complete declared catalog. All 480 no-retry generations completed with 100% validity
and 100% two-pass abstention on missing-input controls.

The extra pass did not solve catalog absence. Explicit novel-capability evidence had 55.56% precision,
10.42% recall, 2.78% non-novel false-positive rate, and 0.2213 ECE. Among 48 novel-valid requests, the
review selected a known capability 28 times, explicit novelty five times, insufficient evidence fourteen
times, and unsupported once. The resulting conservative policy increased known accuracy from 76.04% to
77.08% but worsened false-known acceptance from 5.21% to 7.29% and regret from 0.7760 to 0.9375.

Freeze V115 negative and close the two-pass single-model catalog-fit branch. The result further localizes
the bottleneck: static semantic reconsideration is not new information. The next permissible branch is a
language-free feasibility audit for a typed multi-turn clarification channel. Only if that audit proves that
answers can separate the frozen hypotheses and improve a frozen policy may a new unprotected development
run be preregistered. Protected access, induction, richer planning, APIs, training, model authority, and
execution remain closed.

V116 completed that language-free audit. A single typed answer at 95% simulated correctness did not beat
the frozen 0.7760 regret baseline under uniform or moderate priors. Two conditionally independent answers
did: mean regret was 0.7368, 0.7058, and 0.7174 across uniform, moderate, and strong candidate priors, with
at least 90.25% known accuracy, at least 90.25% unsupported correctness, and at most 0.336% false-known
probability. At 90% reliability, the uniform-prior condition failed.

The fully correlated stress test failed at 95% because uniform and moderate priors had 0.8877 regret. Thus
V116 is conditional feasibility, not evidence for a deployed clarification channel: two answers must be
both about 95% reliable and genuinely independent. Repeating, paraphrasing, resampling, or asking the same
model twice cannot be declared independent. A next branch may only preregister an unprotected causal
simulator with orthogonal answer-generating observations and explicit correlated-error stress tests. No new
language or model run is yet authorized; protected access, induction, APIs, training, authority, and
execution remain closed.

V117 instantiated that causal simulator with separate candidate-confirmation and catalog-status mechanisms.
It failed the preregistered all-prior robustness gates. At 95% reliability and shared-failure correlations
from zero through 0.50, the correlation-aware policy had 1.2056--1.2085 regret and 0% known accuracy under
the uniform prior. Under a strong candidate prior, it had 0% unsupported correctness once correlation was
nonzero and regret rose as high as 1.0399. Only the moderate prior remained near the frozen 0.7760 baseline.

Perfect observations reached the exact 0.690625 control target under every prior, so the mechanisms are
identifying in the noiseless limit. Their 95% likelihood separation is not robust to the full prior set.
Freeze V117 negative and keep fresh language/model evaluation closed. The next justified step is only an
aggregate, model-free identifiability audit of the hypothesis/loss geometry and minimum evidence strength;
it cannot tune prompts or the V117 channel, inspect language, open protected data, induce capabilities, or
authorize actions or execution.

V118 then derived the frozen decision geometry without introducing another observation channel. Exact
candidate action requires roughly 89.47%--90.91% posterior mass. From the uniform 1/17 prior, that means a
Bayes factor of 136--160. V117 supplied only 38, 51.33, and 78 through correlation 0.50 because catalog
status cannot distinguish the candidate from another declared intent. This exactly accounts for the 0%
uniform-prior known accuracy.

The strong-prior unsupported failure is different. On the decisive observation, correlation 0.25 and 0.50
left 81.08% and 79.08% posterior mass on unsupported, but costly residual known mass made abstention optimal.
Only an additional unsupported-specific Bayes factor of 1.091--1.270 would cross that boundary. Therefore
the next model-free design must be asymmetric and adaptive: identity-specific evidence for known action,
support-specific evidence for boundary action. V118 authorizes only preregistration of such a causal
simulator. It does not establish real independent mechanisms or open language/model evaluation.

V119 tested that asymmetric design with the same 0.30 path cost and the V117 all-prior gates. It repaired
all known-accuracy, unsupported-correctness, false-known, perfect-channel, and misspecification failures.
At 95% reliability, known accuracy was at least 85.30%, unsupported correctness at least 88.05%, and
false-known probability below 1% through correlation 0.50.

The branch still failed the noncompensatory mean-regret gate. Regret was 0.7865 and 0.7775 under the uniform
prior at correlations 0.25 and 0.50, and 0.7866 under the strong prior at correlation 0.50, versus the frozen
0.7760 baseline. These are narrow failures, but they cannot be waived. Freeze V119 negative and keep
language/model evaluation closed. A successor may only decompose aggregate regret and test whether
selective, value-triggered querying is feasible without using individual records or retuning V119.

V120 separated V119's frozen 0.30 query cost from decision regret using aggregate metrics only. In all three
failing conditions, decision regret before query cost was 0.4775--0.4866, well below the 0.7760 historical
baseline. Maximum affordable average query cost was 0.2895--0.2986. In a zero-loss idealization, avoiding
only 0.47%--3.51% of queries would close the gaps.

This makes selective querying quantitatively coherent but does not supply a trigger. The next branch may
only preregister a model-free pre-query-signal audit that charges both query cost and the decision value lost
when clarification is skipped. It cannot use post-query evidence, mine individual V119 records, discount
the frozen cost, or open language/model work.

V121 converted that cost gap into necessary trigger-quality requirements. The queried subset must have
0.47%--3.64% higher average clarification value than the population. More importantly, at a 5% skip rate the
two hard conditions require skipped cases to have average query value no greater than 0.0895--0.0908,
roughly 31% of the population average. The V120 zero-value skip fraction was therefore only an optimistic
lower bound.

No trigger is certified. Aggregate metrics cannot establish the required pairing between a pre-query signal
and realized query value. The next step is limited to an inventory of independently available pre-query
signals and their provenance; evaluation, tuning, language/model use, and protected access remain closed.

V122 completed that definition-only inventory. Eleven pre-query signals were cataloged and nine leaking or
unavailable signals were excluded. Frozen character n-gram retrieval geometry—nearest similarity, nearest
intent as context, and the frozen status band—is the only semantic family computationally independent of
the LLM. Observation presence is merely a control. Direct choice and confidence depend on the proposer;
validator, agreement, score-gap, same-scenario, and policy-state signals all inherit proposal dependence.

This is not a claim that retrieval errors are statistically independent of LLM errors, or that retrieval
geometry predicts clarification value. Freeze V122 positive as provenance only. The next action may only
preregister a fresh paired model-free retrieval-geometry design. Signal evaluation, threshold fitting,
language/model access, protected data, induction, APIs, training, authority, and execution remain closed
until that separate design is frozen.

V123 then established that the remaining MASSIVE pool cannot support that fresh paired design. After all
previous evaluation records were excluded, only nine novel-valid turns remained and all came from one
scenario. Freeze this as an availability result rather than weakening the balance, freshness, or scenario
requirements. The branch therefore required an external controlled open-set source.

V124 audited the official Schema-Guided Dialogue dataset at the pinned source revision without emitting
utterances or slot values. Its test split supplies all three structural classes relative to the training
schemas: 3,101 exact known-service/intent turns, 6,601 turns from unseen services in train-seen domains, and
1,403 turns from unseen domains. The source is feasible for a controlled cross-dataset test; this result is
about source structure only, not language performance.

V125 froze an 11-choice action catalog and a balanced 576-record evaluation population spanning Hotels,
Movies, and Services for exact-known and novel-valid cases, plus four unseen domains for unsupported cases.
The six known service-intent choices use all 4,881 eligible training examples as the fixed retrieval base.
No language or model was inspected while selecting the population.

V126 evaluated the sole V122 semantic signal family on that fresh population using the exact frozen
character n-gram retriever and thresholds. The status band queried 478 of 576 cases and skipped 98. Across
all priors and correlation conditions, skipped cases had average clarification value 1.9171--1.9650 while
queried cases had only 0.4810--0.5461. Selective regret was 1.1820--1.2361, worse than both the 1.1667
ask-always control and the 0.8988--0.9609 query-all policy. An oracle with the same 17.01% skip fraction
achieved 0.8392--0.8908 regret, showing that useful selectivity exists but the retrieval band ranks it in
the wrong direction.

Freeze V126 as a decisive negative and close the current pre-query-trigger inventory. Do not refit the
thresholds, relabel the same similarity, or reopen language/model evaluation from this result. A successor
requires a preregistered, mechanistically distinct source of pre-query evidence and a fresh utility test.
Until such a signal is defined independently of outcomes and post-query evidence, ask-always remains the
honest policy for this simulated channel; protected access, induction, APIs, training, authority, and
execution remain closed.

V127 tested the first mechanistically distinct upper bound on another 576 balanced SGD records: perfect
source-authored dialogue-state slot names against the six declared known schemas. Slot evidence was present
for 70.31% of records and the unique-compatibility rule skipped 14.93%, but skipped-action precision was only
87.21%. Skipped clarification value was 1.0169--1.0861, far above the 0.30 cost, and exact-known probability
was only 66.58%--68.71%. Freeze V127 negative. Because the input was perfect annotation, do not build an LLM
or parser to realize this slot-set rule.

V128 then tested richer typed `act(slot)`, accumulated-state-slot, and requested-slot relations. Six support
sets were frozen from 4,881 known training frames and evaluated on 432 records disjoint from V125 and V127.
The rule skipped 23.61% with 93.14% precision, but skipped cases still had 0.4600--0.5336 clarification value
and selective regret exceeded query-all in every condition. Exact-known probability remained only
72.09%--72.56%.

Freeze V128 negative and close the complete annotation-signature trigger family. The failure is no longer
plausibly attributable to language parsing: both audits used oracle source annotations. The next problem is
the clarification interface itself. Its candidate-specific form can reject a wrong known candidate but
cannot identify a different exact known intent. A successor may only audit, model-free, a complete typed
clarification interface that preserves every safe hypothesis and charges the same costs. Do not search for
another pre-query threshold or reopen language/model work, protected data, induction, richer planning,
APIs, training, authority, or execution.

V129 performed that complete-interface audit over all 66 safe truth/presented-candidate pairs. A perfect
answer produced 100% known and unsupported decisions, so the interface is identifying. At 95% reliability,
uniform and moderate priors generally reached 95% exact-known probability with 0.8350--1.1242 regret. But a
75% wrong-candidate prior still dominated one noisy answer: symmetric and candidate-attracted errors
produced only 15.83% exact-known probability, with regret 1.2323 and 1.4847. Candidate-attracted error was
also worse than the frozen candidate-specific comparator.

Freeze V129 negative. The full menu repairs expressivity but not evidence strength. Do not realize it with
a human or model on the assumption that one 95%-correct response is sufficient. The next permissible work
is a model-free evidence-strength audit deriving the minimum one-answer reliability and the number of
genuinely independent answers required under biased and correlated error. Repeated samples or paraphrases
from one LLM cannot be declared independent. All language/model, protected, induction, richer-planning,
API, training, authority, and execution paths remain closed.

V130 derived the evidence-strength boundary for that complete interface without reading language or
running a model. On the frozen 0.0005 reliability grid, every one-answer condition passed by 97.25%:
six passed at 95%, uniform candidate-attracted error required 96%, strong candidate-attracted error
required 96.7%, and a strong wrong-candidate prior with symmetric error required 97.25%. Perfect answers
retained and identified every safe hypothesis, so the abstract single-source route is feasible below the
preregistered 99% ceiling.

The repeated-answer route did not pass its robustness gate. Two 95%-correct answers were sufficient only
under stipulated independence. With common-shock correlation of 0.25 or 0.50, none of one, two, or three
answers passed the hardest strong-prior conditions. Freeze V130 as a positive abstract feasibility result,
not as evidence that a person or model realizes the required reliability. It authorizes only a separately
preregistered evidence-realization audit. Repeated generations from one model family may not be counted as
independent; protected access, induction, richer planning, APIs, training, authority, and execution remain
closed.

V131 then froze a record-disjoint realization population from the unconsumed SGD test remainder. Its 264
fixtures cover all 66 truth-by-presented-candidate cells four times: 240 distinct source records plus 24
missing-observation controls. Every truth appears 24 times, every known candidate 44 times, and no selected
identifier overlaps V125, V127, or V128. This was a text-free population result only.

V132 ran the exact locked one-pass local realization condition. The pinned Qwen3.8-27B 4-bit model returned
valid JSON on all 264 fixtures, but exact complete-answer accuracy was 74.24%, versus the required 97.25%; the
one-sided 95% Wilson lower bound was 69.59%. Accuracy was 80.56% for known truths, 55.56% for valid-undeclared
truths, 66.67% for unsupported truths, and 100% for missing controls. False-known answers reached 18.33%.
The downstream frozen policies had 1.7072--1.7773 regret, 66.67% unsupported correctness, and 17.50%--18.33%
false-known probability, so every performance family failed while all access and safety gates passed.

Freeze V132 negative and close the one-pass local complete-answer realization branch. The complete menu
fixed abstract expressivity but did not create the evidence strength required by V130. Do not retry, revise
the prompt, substitute a larger or API model, or mine individual responses. Before interpreting this as a
pure model-capacity failure, audit the benchmark's semantic label identifiability: SGD's structural novelty
rule can mark an unseen service-intent pair novel even when its intent name is identical to a declared
capability. Protected access, induction, richer planning, APIs, training, authority, and execution remain
closed.

V133 audited the source-label geometry using only the pinned train and test schemas. Forty-eight of 72
selected novel-valid fixtures (66.67%) came from unseen service-intent definitions whose normalized intent
name exactly matched a declared known choice. `N02` and `N03` were entirely name-colliding; only `N01` had
no such collision. No raw descriptions, dialogue language, slot values, model responses, or manual semantic
judgments were read.

The novel schemas were not exact full-definition duplicates: normalized descriptions and required/optional
slot signatures differed. Therefore V132 remains a valid negative for service/schema-version discrimination,
but it is not a clean test of obviously distinct novel capabilities. Freeze V133 negative and retract the
pure novel-capability interpretation of V131/V132. The only permissible successor is a text-free source and
catalog design whose selected novel names do not collide with declared choices. Do not rerun or revise the
model condition; protected access, induction, richer planning, APIs, training, authority, and execution
remain closed.

V134 constructed that corrected asset from the untouched SGD development partition. The frozen eleven-choice
catalog uses six declared intents from rental cars, weather, events, and travel; novel composites from banks,
flights, and media; one unsupported alarm composite; and one insufficient-evidence control. Its 264 fixtures
cover all 66 truth-by-candidate cells four times, with 240 unique source identifiers and 24 missing controls.

Selected novel definitions have zero normalized name collisions and zero exact full-signature collisions
with declared choices. No development utterance, dialogue history, slot value, or model output was opened.
Freeze V134 positive as a future benchmark asset only. The current authorization does not permit language
extraction or another local/API model run. The next research move requires an independently justified
realization source or explicit new authorization; induction, richer planning, training, authority, actions,
and execution remain closed.

## V135–V136 controlled identifiability and clarification value (2026-08-18)

V135 replaced the confounded source distinction with a fully controlled nine-choice benchmark. Forty
five-stage minimal-pair groups contain clear-left, clear-right, deliberately ambiguous, clarified-left, and
clarified-right conversations across five boundary families. The 200 fixtures are split evenly between
development and test, preserve the complete safe choice universe, and expose no hidden labels in prompts.

V136 proved model-free that the targeted clarification channel has decision value. At 95% answer
reliability, every family and prior condition improved cost; the worst reliability threshold for positive
query value was 91.3%, below the historical 97.25% complete-answer boundary. Clear cases correctly skip the
query. This established that the controlled task measures useful abstention rather than abstention for its
own sake.

## V137–V138 thinking measurement repair (2026-08-18)

V137's direct condition was valid and reached 96% overall accuracy but only 80% ambiguous abstention. Its
thinking condition was technically invalid: the pinned Qwen template opens `<think>` in the prompt, while
the parser required balanced tags inside the generated suffix. Ninety-three normal closing-tag suffixes
were rejected and seven outputs reached the 512-token ceiling. Raw text was not retained, so the condition
could not be retrospectively recovered.

V138 confirmed the exact template/parser mismatch without loading a model or reading response text. It
froze a stateful parser that begins generated-suffix parsing at thinking depth one. V137 remains unchanged;
the repair authorized only one fresh comparison on the unused V135 development split.

## V139 repaired direct-versus-thinking comparison (2026-08-18)

V139 completed 200 deterministic local generations with the pinned Qwen3.8-27B 4-bit model, one load, no
retry, no raw trace persistence, no API, and zero execution. Direct inference reached 94% overall accuracy,
70% ambiguous abstention, and 1.16 sequential cost. Thinking reached 97%, 90%, and 0.62 respectively. It
also reduced difficult-branch false-known decisions from 10% to 5% and increased safe non-known decisions
from 90% to 95%. Paired, thinking repaired four direct errors and introduced one.

Thinking still failed qualification. Three outputs hit the preregistered 1,024-token ceiling without
closing the trace, yielding 97% structural validity versus 99% required. Two valid ambiguous outputs still
overcommitted, leaving 90% apparent ambiguity accuracy versus 95% required; valid-only ambiguity accuracy
was 16/18. Thinking was about 3.42 times slower and generated about 20.4 times as many tokens.

Freeze V139 as a positive mechanism result but negative qualification result. It shows that deliberation
can materially improve abstention and decision cost, not that it is sufficient or externally valid.

## V140 two-mechanism qualification gap (2026-08-18)

V140 established model-free that neither remaining problem can explain the other. Making the three fallback
answers structurally valid would leave ambiguity at 90%; fixing the two valid semantic overcommitments would
leave structural validity at 97%. Qualification requires at least two fewer invalid outputs and at least one
additional apparent ambiguous correction.

The next authorized work is only a model-free feasibility study for a bounded finalizer plus an explicit
evidence-sufficiency mechanism on a fresh future population. A larger token ceiling, post-hoc retry, or
same-model multi-pass scheme presented as independent evidence is not an acceptable repair. V134 and V139
remain closed; external language, APIs, training, induction, authority, action, and execution remain closed.

## V141–V143 certificate-controller feasibility and oracle validation (2026-08-18)

V141 derived a conservative two-stage controller envelope without assuming independent errors. A bounded
finalizer, ambiguity-sensitivity mechanism, decidable-specificity mechanism, and correct proposer could
jointly satisfy the structural, decision, and sequential gates at roughly 98–99% marginal reliability.
This established feasibility, not model performance.

V142 then froze a fresh nine-choice certificate interface and 288 project-authored controlled fixtures in
48 six-stage groups, evenly split between development and test. A certificate reports evidence sufficiency,
the complete compatible-choice set, and a non-authoritative proposal. The deterministic finalizer maps every
invalid or inconsistent certificate to programmatic `A00` and never prunes the authoritative universe.

V143's oracle produced exact certificates and final choices on all 288 fixtures, achieved 0.3 sequential
cost, and made all nine malformed-certificate classes fail closed. A well-formed wrong singleton remained
structurally undetectable, correctly locating semantic correctness in the empirical proposer rather than the
validator. Passing authorized only one separately preregistered local development realization.

## V144 pinned local certificate realization (2026-08-18)

V144 completed all 144 locked development generations with the pinned local Qwen3.8-27B 4-bit model, one
load, no retry, no raw trace persistence, no API, zero test generations, and zero execution. The fail-closed
system achieved 97.22% final exact accuracy, 0% false-known errors, 0.3417 sequential cost, and 0.8250
improvement over not querying.

The model did not realize the certificate contract reliably. Twenty-four traces hit the 1,024-token ceiling
without closing; twenty were ambiguous cases. Certificate validity, compatible-set accuracy, and certificate
option retention were each 83.33%. Observable ambiguity sensitivity was only 16.67%, and decidable specificity
was 96.67%. Although all 116 completed sufficient certificates were correct, that conditional result cannot
resolve the missing outputs.

Freeze V144 negative. The safe ambiguous final decisions came from deterministic fallback, not reliable model
ambiguity certificates. The V142 test split remains sealed. Do not tune the prompt, increase the ceiling,
retry, rerun, change models, or move to an API within this branch. External transfer, induction, training,
authority, action, and execution remain closed pending a newly justified research direction.

## V145 finite certificate codebook feasibility (2026-08-18)

V145 removed free-form completion from the abstract interface without reading language or running a model.
Eight sufficient singleton certificates and six registered ambiguity-pair certificates cover the complete
controlled topology with only 14 fixed codes. The oracle recovered exact codes, certificates, and final
choices on all 288 abstract rows, and ten malformed-code classes all failed closed to valid `A00`.

The codebook does not validate semantics: a registered wrong singleton is structurally indistinguishable
from a correct one. Freeze V145 positive only as interface feasibility. It authorizes design of a fresh
population and a score-all-registered-alternatives protocol, not a language or model run. V142 test access,
V144 tuning or rerun, APIs, training, induction, authority, action, and execution remain closed.

## V146 fresh codebook-scoring population (2026-08-18)

V146 froze a new nine-choice population with dental visits, diagnostic-kit shipments, audiobook plans,
security monitoring, retail parcels, appliance warranties, rental-vehicle coverage, retroactive erasure,
and insufficient evidence. Its 48 six-stage groups yield 288 fixtures split evenly between development and
test. The 14-code certificate codebook has exact oracle coverage.

Every compatibility label is structurally checkable, public rows hide all ground truth, and exact conversation
overlap with V135 plus V142 is zero. No model, API, training, or execution occurred. Freeze V146 positive as
a prospective asset only. It authorizes separate scoring-protocol preregistration, not an immediate model run,
V142 test access, induction, authority, action, or execution.

## V147 closed-alternative likelihood scoring (2026-08-18)

V147 prospectively scored all 14 registered certificate aliases for each of 144 fresh V146 development
fixtures with the pinned local Qwen3.8-27B 4-bit model. It completed exactly 2,016 sequence scores with one
model load, zero generated tokens, no retries, no API, zero test scores, and zero execution. Structural
validity rose to 99.31%, so free-form completion was successfully removed as the dominant measurement
confound.

Semantic qualification failed decisively. Overall exact accuracy was 77.78%, compatible-set accuracy was
74.31%, ambiguity sensitivity was 50%, and false-known errors on non-known truths were 15.28%. Familiar
known requests were perfect, but novel-valid requests were only 43.75% correct and insufficient-evidence
requests only 54.17%. Eleven of 24 ambiguous truths were collapsed to a known capability. Sequential cost
was 2.6521 and difficult-branch false-known decisions were 45.83%.

Freeze V147 negative. Score-all-fixed-alternatives solves syntax and completion, not the open-world
membership boundary. Do not rescore, rerun, fit a post-hoc threshold, change aliases or prompts, open an API
condition, train, induce capabilities, grant authority, or execute. The V146 test split received zero scores
but is retired because two rows were displayed during pre-preregistration implementation inspection. Any
future empirical claim requires a wholly new blind population and a separately justified prospective
selective-evidence protocol.

## V148 typed semantic-witness firewall (2026-08-18)

V148 changed the trust boundary model-free. A known capability can be accepted only from an exact witness
issued through a closed typed clarification channel. An optional LLM label is causally irrelevant. Valid
unregistered mechanics route to generic, non-executable `N00`; an exact forbidden mechanic routes to `U30`;
and missing, malformed, contradictory, or insufficient evidence routes to `A00`.

All 98 valid-witness-by-proposal cases were exact and invariant to the seven possible LLM proposals. All ten
malformed witnesses failed closed, and all twelve one-field near-known mutations avoided known acceptance
even when the proposed label named that known capability. Known precision and recall under oracle typed
witnesses, novel/unsupported/insufficient routing, output validity, and complete hypothesis retention were
all 100%. No language, model, API, training, registration, action, or execution occurred.

Freeze V148 positive as structural feasibility only. It does not show that an LLM or utterance can provide
a trustworthy witness. The only authorized successor is design of a wholly new blind population with
independently checkable closed clarification answers. Any model access requires a separate prospective lock;
calibration fitting, induction, authority, action, and execution remain closed.

## V149 fresh closed-interaction population (2026-08-18)

V149 froze 288 new project-authored interaction fixtures in 48 complete six-stage groups, evenly divided
between development and evaluation. Six registered two-option questions map structured answer events to
typed witnesses for four known capabilities, generic `N00`, or forbidden `U40`; absent evidence remains
`A00`. All 96 closed answers routed exactly, all 192 pre-answer fixtures remained fail-closed, and eight
malformed event classes failed closed. Exact conversation overlap with V135, V142, and V146 was zero.

Freeze V149 positive as population design only. It is synthetic, not external transfer, and establishes no
model performance. It authorizes only a model-free oracle interaction policy.

## V150 oracle closed-interaction policy (2026-08-18)

V150 evaluated 48 latent-side development episodes under explicit costs. The exact planner selected the
registered discriminating question in every episode, achieved 100% final accuracy and 0.3 mean cost versus
1.0 for no-query abstention, and resolved one episode per question. Five irrelevant queries per episode
produced no selection and safely remained `A00`.

All results were invariant across 2,352 combinations of seven state proposals and seven query proposals.
False-known outcomes were 0%, safe non-known outcomes 100%, and complete hypothesis retention 100%, with no
language, model, API, training, action, or execution access. Freeze V150 positive as oracle feasibility only.
The authorized successor is design of a local development proposal protocol; model execution still requires
a separate prospective lock, and evaluation access, calibration fitting, induction, authority, and execution
remain closed.

## V151/V151r1 local proposal and query ranking (2026-08-18)

The prospectively locked V151 development run was externally interrupted after 59 attempts. Before semantic
output inspection, V151r1 froze a no-retry recovery: retain 58 durable normalized outputs, force the one
interrupted fixture to the registered invalid fail-closed proposal, and generate only the 37 never-started
fixtures. The completed recovery used exactly 96 attempts, two total model/tokenizer loads, no retries, no
evaluation or closed-answer generations, no raw-response inspection or persistence, no API or training, and
zero execution. The interruption cannot account for the substantive failed gates.

The trusted architecture remained exact. Across 120 sequential episodes, final accuracy after closed answers
was 100%, false-known outcomes were zero, every irrelevant intermediate answer failed closed, and the full
authoritative state set was always retained. Model question ordering reduced mean decision cost to 0.3525
from the 1.0 no-query baseline and produced 0.6475 improvement, passing both registered utility gates.

The model component did not qualify. Compatible-state retention was 84.375%, exact compatible-set recovery
57.292%, ambiguity sensitivity 50%, query top-1 accuracy 83.333%, and query MRR 0.9141. It failed the 97%,
90%, 95%, 90%, and 0.95 gates respectively; 84% of semantic errors were attracted toward a presented
candidate. Self-reported confidence was strongly overconfident and remains diagnostic only.

Freeze V151r1 negative and leave the V149 evaluation split unopened. The result supports a narrower
architectural observation: imperfect question ordering can save trusted-interaction cost without gaining
semantic authority. Candidate-set proposal and query ranking must no longer be treated as one claim. The
authorized successor is prospective, model-free design of a fresh question-order-only benchmark. It must
discard candidate-set pruning, retain the complete authoritative state universe, and compare a registered
question-order heuristic with fixed, random, and oracle orders under the same typed-answer firewall. No
V151/V151r1 rerun, prompt or reasoning change, threshold fit, calibration fit, API, training, induction,
authority, action, or execution is authorized.

## V152–V154/V154r1 question-order-only sequence (2026-08-18)

V152 implemented the architectural separation recommended after V151r1. It froze 288 wholly fresh
project-authored fixtures in 48 complete groups, evenly divided between development and evaluation. The
interface exposes six registered closed questions but no candidate-state proposal, confidence, state
ranking, or hypothesis-pruning field. A trusted typed answer alone determines the final state. All structural,
freshness, witness-firewall, and access gates passed; the evaluation half remained sealed.

V153 established the model-free cost envelope on development metadata. Safe no-query abstention cost 1.0;
source and seeded-random order both had mean correct-query rank 3.5 and cost 1.05; oracle order had rank 1
and cost 0.3. Every question-asking comparator ended exactly after the trusted answer, and all 600 irrelevant
intermediate answers failed closed. This proved that question order can improve efficiency without receiving
semantic authority.

V154 then tested the pinned local Qwen3.8-27B 4-bit model under a prospectively adaptive protocol. Direct
decoding ran first. It produced valid rankings on all 96 development requests, placed the discriminating
question first on 80 and second on 16, and achieved top-1 accuracy 83.333%, MRR 0.9167, mean rank 1.1667,
and cost 0.34. The heuristic was far better than source/random order and close to the 0.3 oracle cost, but
missed the preregistered 90% top-1 and 0.95 MRR gates.

Because direct failed, the locked rule triggered bounded low reasoning. Every one of its 96 reasoning phases
used the full 48-token mechanical budget and required a forced `</think>` close. The reserved final phase
prevented reasoning from swallowing the answer, but performance worsened: five invalid JSON outputs,
81.25% top-1, 0.8938 MRR, mean rank 1.3125, and cost 0.375. No final phase hit its token limit. This isolates
two conclusions: a hard reasoning budget is a useful completion safeguard, while enabling low reasoning is
not beneficial for this compact ranking task. Direct decoding remains the appropriate local configuration.

Both conditions preserved the trust boundary: all 240 sequential episodes reached the exact final state
after typed answers, irrelevant intermediates failed closed, authoritative retention was 100%, candidate
proposal fields were absent, and execution was zero. Freeze V154 negative without opening evaluation or
tuning the same language.

The initial locked V154 outcome verifier exposed a representation-only defect: recomputed `rank_counts`
used integer mapping keys while JSON persistence returned their exact string equivalents. V154r1 preserved
the failed audit and original verifier, prospectively locked the diagnosis, and passed a repaired verifier
using recursive JSON canonicalization. It made no model or tokenizer load and changed no metric, gate,
decision, or claim. The frozen decision remains
`local_question_order_conditions_fail_development_gates_close_without_evaluation_or_tuning`.

The next branch must be scientifically distinct from decoding and prompt adjustments. It should
prospectively test a deterministic retrieval policy on wholly fresh language using only visible request text
and registered question metadata. A hybrid policy may use the LLM only as a non-authoritative tie-breaker
after model-free retrieval feasibility is established. Wrong ordering may increase question cost but must
never accept a capability, remove a state, synthesize a trusted answer, or execute. V152 evaluation access,
V154 reruns or tuning, APIs, training, induction, authority, action, and execution remain closed.

## V155–V156 explicit-metadata deterministic retrieval (2026-08-18)

V155 froze a wholly fresh synthetic retrieval population with 288 fixtures in 48 complete groups, balanced
between development and evaluation. Six registered clarification questions expose fixed anchor phrases,
primary terms, and secondary terms. Public fixtures contain no truth, oracle, witness, candidate-state,
confidence, or pruning field. Trusted typed answers route exactly, pre-answer and malformed evidence remains
`A00`, the complete authoritative universe is retained, and exact conversation overlap with V135, V142,
V146, V149, and V152 is zero. No retrieval policy or model was run during design.

V156 then prospectively froze a deterministic retrieval rule before reading development truth: phrase,
primary-term, secondary-term, and visible question-surface weights of 8, 3, 1, and 0.25, with source order
as the only tie-break. A state-free catalog projection removed all choice, witness, truth, and oracle fields.
The single development census scored 96 requests against six queries without fitting or learned parameters.

The policy selected the discriminating question first on all 96 requests. Top-1 accuracy and MRR were 1.0,
mean rank was 1.0, and top-score tie rate was zero. Across 120 sequential episodes it matched the oracle:
mean cost 0.3 and improvement 0.7 over safe no-query abstention. Source order cost 1.05 and seeded random
cost 1.005. Every trusted final answer was exact; all irrelevant intermediates failed closed; hypothesis
retention was complete; candidate fields and execution were zero.

This positive result is catalog-coverage feasibility, not unrestricted language understanding. The synthetic
requests deliberately use vocabulary represented by the explicit profiles. It shows that an LLM is
unnecessary—and inferior to a verifiable deterministic policy—when the registered catalog covers the
language. It does not show that retrieval handles uncatalogued paraphrases, relational distinctions, lexical
ties, or requests with too little evidence to privilege any question.

Freeze V156 positive without opening V155 evaluation or adding a model. The authorized successor is only
design of a new hard-tie population with prospectively labeled lexical controls, uncatalogued paraphrases,
near/equal topical ties requiring relational evidence, and genuinely insufficient requests. Model-free
retrieval plus an explicit safe fallback must be measured first. A local LLM may later be tested only as a
permanently non-authoritative tie-breaker on a separately locked fresh split. APIs, training, capability
induction, authority, action, and execution remain closed.

## V157–V158 hard-tie generic routing (2026-08-18)

V157 added a second, non-semantic clarification level on wholly fresh synthetic language. Its 384 fixtures
form 48 complete groups across lexical controls, uncatalogued paraphrases, cross-family relational ties,
and genuinely insufficient requests. Generic query `Q70` has six family-route options plus `UNCLEAR` but
no state or witness field. All 96 generic route answers were valid and produced zero semantic witnesses;
all 96 specific answers routed exactly. Requests, route-only answers, and malformed events remained `A00`,
hypothesis retention was complete, prior conversation overlap was zero, and execution was zero.

V158 prospectively froze a model-free router: ask a specific question only if lexical top score is at least
6, top-two margin at least 4, and the top score unique; otherwise ask `Q70`. A wrong specific question must
fail closed and then fall back to `Q70`. Specific and generic questions cost 0.3 and 0.2 respectively.

The policy failed qualification. It was exact on every lexical control and insufficient request, but sent
one uncatalogued paraphrase and four relational ties to a wrong specific question. Initial-action accuracy
was 94.792%, relational generic routing 83.333%, uncatalogued generic routing 95.833%, and false-specific
rate on fallback strata 6.944%. Mean cost was 0.4225 rather than at most 0.4; improvement over no query was
0.3775 rather than 0.4; improvement over always-generic was 0.0175 rather than 0.04.

The five errors were high-margin, not low-confidence ties. The food-rescue paraphrase was attracted to the
school-meal question at score 7.0 and margin 6.25. All four environmental-monitor relational variants were
attracted to the heritage-image question at score 6.75 and margin 6.25. A lexical margin is therefore not a
semantic evidence-sufficiency certificate: one well-covered decoy relation can create apparent confidence.

The firewall still achieved its purpose. Wrong specific answers produced no witness and automatically fell
back through `Q70`; all 120 router episodes ended exactly. Every irrelevant intermediate failed closed,
hypothesis retention was 100%, candidate fields were absent, and execution was zero. Model-free errors cost
questions without causing false capability acceptance.

Freeze V158 negative. Do not change its terms, weights, 6/4 thresholds, costs, strata, or gates; do not open
V157 evaluation or introduce a local/API model. The next branch must use fresh language and a prospective
relational/grammar conflict mechanism that detects asserted alternative relations rather than interpreting
score margin as confidence. Model-free grammar and deterministic fallback must be tested before any LLM
tie-breaker. Training, calibration fitting, induction, authority, action, and execution remain closed.
