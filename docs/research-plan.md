# Research plan

## Question

Can counterfactual transition supervision teach a compact language model to predict
context-dependent affordance changes more accurately than prompting or action imitation alone?

## Measurement firewall

The simulator is the oracle. Model-generated predictions and rationales are never used as
transition truth. Targets come only from cloned calls to `resolveAction`.

## Audit gate

The first full-corpus audit found that every test agent prompt occurs verbatim in training and
that 77.3% of records belong to exact-prompt groups with multiple targets. The privileged
snapshot reduces but does not remove the problem: 53.5% of its records remain in contradictory
groups because scenario transition rules are not included in that snapshot.

Dataset v2 implements the audit-gate revision:

1. All transition-relevant scenario rules are part of the privileged Markov input.
2. Exact agent prompts are collapsed into observational equivalence classes.
3. Ambiguous agent targets contain complete empirical possible-outcome sets.
4. Prompt and state-context signatures do not cross evaluation boundaries.

The next audit found a remaining split gate: ambiguity is 39.08% in training, 64.14% in
validation, and 26.11% in test. The next compiler must preserve whole context groups while
constraining ambiguity and mechanic-family distributions across splits. After that, the remaining
corpus-expansion gate is to add renamed or procedurally varied worlds so success cannot rely on
the small fixed vocabulary of rooms, actions, and mechanics.

Dataset v3 implements that split revision for the agent calibration track. Whole observation
contexts remain atomic, while a deterministic multi-restart group assignment aligns ambiguity,
exact outcome-count, action-family, scenario-family, and sufficiently supported mechanic-tag
distributions. The compiled split has a 0.93-point ambiguity-rate range and zero prompt or context
overlap.

## Tracks

1. **Privileged dynamics:** complete runtime state and transition rules plus candidate action to
   exact transition.
2. **Agent-view epistemics:** observation history plus candidate action to identifiability and the
   complete empirical possible-transition set.
3. **Belief revision:** future track for updating possible worlds after a prediction mismatch.

The tracks now answer complementary questions: whether the complete simulator state is sufficient
for exact prediction, and whether an agent can preserve uncertainty when its observations are not.

## Conditions

- A: prompted, untuned Qwen3.5.
- B: action-only imitation (future adapter).
- C: counterfactual transition supervision.
- D: transition plus structured mismatch revision (future adapter).

Train separate adapters before attempting a multitask adapter so causal attribution stays clear.

## Splitting

Dataset v2 splits on observation-state context before adding the candidate action, so every action
from one perceived state remains in one split. Privileged v2 uses the same context-disjoint rule.
Exact prompt signatures are also validated as split-disjoint.

Future split axes should add isomorphism IDs, entity renamings, held-out mechanic combinations,
and genuinely unseen primitives.

## Initial metrics

- Exact transition-object match.
- Field-level accuracy.
- Set precision/recall/F1 for inventory and action-surface deltas.
- Counterfactual accuracy across all available actions.
- Agent-view versus privileged-state performance.
- Scenario-family and split-group breakdowns.

Calibration metrics now include balanced identifiability accuracy, ambiguity precision/recall/F1,
exact outcome count, macro accuracy across observed counts, and count MAE. Training-time sampled
loss is never a substitute for full generated validation.

The V3 count experiment selects checkpoints separately for seeds 0, 1, and 2 using constrained
digit logits over the complete fixed validation split. The frozen gate requires every seed above
50% balanced identifiability, at least two seeds and their mean at or above 55%, a maximum
ten-point range, and nonconstant predictions. Test remains closed. If the gate passes, the next
causal diagnostic is oracle-count transition generation; otherwise the count representation or
corpus must be revised before another long-form adapter run.

The completed three-seed run failed the gate. Every validation-selected checkpoint was a constant
count-1 predictor at 50% balanced identifiability. One late seed checkpoint flipped almost wholly
to count 2 rather than learning a stable boundary. In contrast, a balanced-prior token Naive Bayes
baseline on the same visible input reached 73.08% validation balanced identifiability. The next
revision should therefore use a binary classifier objective with a context-group calibration fold,
then learn exact ambiguous counts hierarchically. Oracle-count generation remains gated off.

Dataset and experiment V4 implemented the binary revision. It used balanced single-token `A`/`B`
supervision, carved calibration only from V3 training contexts, selected checkpoints and
thresholds on that fold, and evaluated each frozen seed once on V3 validation. The three selected
seeds reached 49.15%, 48.96%, and 55.49% validation balanced identifiability, for a 51.20% mean.
Both preregistered gates failed. The selected score margins also collapsed to only two or three
distinct values because the language-model vocabulary logits were emitted at coarse precision.

The next revision should therefore separate representation from language-model output
calibration: extract frozen hidden states from 0.8B, 4B, and 9B Qwen3.5, train a class-balanced
float32 linear probe, and select layer/regularization only on calibration. If a frozen probe
succeeds, add LoRA behind the same discriminative head; if it fails across layers and sizes,
revise serialization and corpus variation. Another digit/letter vocabulary-label run is not
eligible. Exact ambiguous counts and oracle-count generation remain gated off.

The V5 0.8B gate passed before larger-model extraction was needed. Across three true-float32
probe fits, layer 6 mean pooling reached 96.15% validation balanced accuracy and 0.9998 AUC on
full input. Removing history and memories still reached 91.64% and 0.9803. The latter errors were
concentrated in three of 19 validation contexts, with two contexts wholly misclassified. This
establishes linearly accessible signal in the smallest model, but not semantic epistemic
generalization. Freeze the discriminative method and generate the planned shortcut-resistant
challenge holdout before interpreting LoRA improvements; 4B/9B frozen probes are deferred because
the scale-capacity question is already answered at 0.8B.

The method and threshold were then hash-locked and evaluated once on 120 new base cases (360 with
paired surfaces) from 63 contexts. The challenge used short-start relock and a previously unseen
power-trip mechanic, fixed simulator seeds 8101–8103, entity renamings, paraphrases, and two narrow
evidence-contrast groups. It had zero prompt or source-scenario overlap with V4 and never read V3
test. The frozen probe failed: 49.58% canonical balanced accuracy, 0.457 AUC, a 47.62–51.59%
context-group bootstrap interval, 50.00% on power-trip, and 52.28% on short-start relock. Renamed
and paraphrased surfaces were also near chance. The two evidence groups preserved correct pairwise
score direction, but are too narrow to override the failed generalization gate.

This revises the V5 interpretation. The 0.8B representation linearly encoded features that nearly
perfectly predicted labels inside the existing generated distribution, but the locked boundary did
not transfer to new structural/mechanic families. Do not proceed directly to LoRA on the old corpus:
first redesign training data around cross-template evidence interventions, mechanic-held-out
validation, and paired invariance objectives. Exact counts and transition generation remain gated.

V6 implemented that redesign without exposing exact counts or transition targets. Short-start
relock and power-trip seeds 9101–9106 supply 143 training and 49 calibration base cases; each has
canonical, entity-renamed, and paraphrased views with one shared binary target. A new
mirror-rejection simulator mechanic, seeds 9301–9303, supplies 76 base cases in a one-shot holdout.
The corpus has no prior prompt/scenario overlap, no cross-split context or prompt overlap, and no
V3 test access. The fixed layer-6 mean, `C=10`, seed-0 probe passes development calibration at
90.32% balanced accuracy.

The locked holdout is nevertheless a LoRA no-go. Canonical balanced accuracy is 63.16% with 0.592
AUC and a 51.55%–82.44% context-group interval. Entity-renamed inputs reach 69.74%, paraphrases
57.89%, and complete-triplet accuracy is 56.58%. Six of ten gates fail. Most importantly, all 12
label-changing evidence comparisons are ordered in the wrong direction. The corpus audit explains
why: announced development examples are exclusively ambiguous, while all announced consequence
and procedure examples in the new mechanic are identifiable. Surface augmentation improved
transfer by 13.58 points over V5, but did not remove the evidence-wording shortcut.

The next eligible corpus revision must explicitly counterbalance evidence wording by label and
mechanic before model access: every evidence rung/action template needs all four cells of
mechanic × identifiability where the simulator permits them, and the audit must cap conditional
label-rate gaps before hashing. LoRA, exact-count generation, and transition-target training remain
gated.

V7 implemented that revision and closed the shortcut gates before feature extraction. The final
development corpus uses relock-short and power-trip, with exact 50/50 labels within every supported
evidence × mechanic × action-template × surface cell. Metadata-only and evidence-text-only
classifiers both remain at chance. The fixed 0.8B layer-6 mean probe clears the preregistered
development gate at 76.16% canonical balanced accuracy, so the probe, threshold, corpus, simulator,
and evaluation code are frozen before opening tone drift, a newly implemented mechanic with fresh
seeds.

The sole tone-drift evaluation fails decisively. Canonical balanced accuracy is 48.57% with a
42.86%–50.00% context-group bootstrap interval; entity-renamed and paraphrased balanced accuracy
are 50.00% and 45.71%. Both thresholded evidence-direction accuracy and paired score-direction
accuracy are 0% on the two oracle label-changing groups. Canonical AUC is nevertheless 0.807,
showing that some rank signal transfers while score location and causal direction do not. This is
not sufficient to reopen LoRA: the frozen decision rule is the preregistered unit of evaluation,
and eight holdout gates fail. Any future work must be a new protocol focused on cross-mechanic
score calibration and denser oracle label-changing pairs; it must not tune on or rescore the V7
tone-drift reserve.

Next metrics: goal invariance, entity-renaming invariance, one-flag minimal-pair sensitivity,
multi-step rollout divergence, and post-mismatch recovery.
