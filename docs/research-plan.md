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

Next metrics: goal invariance, entity-renaming invariance, one-flag minimal-pair sensitivity,
multi-step rollout divergence, and post-mismatch recovery.
