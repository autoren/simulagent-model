# V8 structured causal-development protocol

## Claim under test

V8 deliberately separates mechanism knowledge from evidence reasoning. The model receives a compact action-dependency schema containing determinant roles and a complete table from determinant-value combinations to opaque transition codes. It does not receive current hidden determinant values, an identifiability label, or a decisive-role annotation.

The development claim is narrow: given this rule schema and an evidence ledger, a frozen Qwen3.5-0.8B layer-6 representation should encode whether resolving the relevant fact changes the set of possible transitions, including when the action mechanic and surface are omitted from probe training.

This is not a rule-induction, transition-generation, action-selection, recovery, or LoRA experiment.

## Data firewall

- The six exposed development mechanics are hatch traversal, generator tuning, beacon calibration, mirror-triggered power trip, mirror rejection, and pressure-triggered hatch relock.
- Labels and opaque transition codes are recomputed from `resolveAction` over all Boolean determinant assignments.
- Tone Drift records, scores, features, and labels are forbidden. Tone Drift remains a V7 postmortem artifact and cannot be used for V8 selection.
- V3 test records, previous holdouts, and previous model results are forbidden.
- V8 creates no untouched final mechanic. A new final mechanic is eligible only after the development architecture passes every fold.

## Intervention construction

For each mechanic, assignment, determinant, and replica, V8 creates a matched pair:

1. The primary transition determinant is unresolved.
2. The same determinant is resolved to its actual true or false value.

All other actual-world values, the action, dependency table, context, and surface remain fixed. Seven evidence rows are present in every input: six are confirmed with exactly three active and three inactive values, while one is unresolved and hidden. Uncertainty moves between the relevant determinant and an irrelevant context fact, so pair members have identical serialized length and token bags.

The oracle assigns `oracle_label_flip` when the unresolved member admits multiple transition hashes and the resolved member admits one. It assigns `same_label_causal_control` when both members admit one transition. The structured target distinguishes resolved true, resolved false, unresolved outcome-sensitive, unresolved outcome-invariant, and irrelevant facts.

Every base record has canonical, entity-renamed, and paraphrased views with one target. Entity-renamed and paraphrased inputs replace model-facing determinant identifiers as well as labels.

## Pre-model gates

Before Qwen features are extracted, leave-one-mechanic-out baselines are fit on the other mechanics' training records, thresholds are selected on their canonical calibration records, and the omitted mechanic is evaluated without recalibration.

Hard ceilings are read from `configs/dataset.v8.json`:

- metadata worst-fold balanced accuracy: 0.55;
- unigram worst-fold balanced accuracy: 0.55 and AUC separation: 0.65;
- role-scrubbed character n-gram worst-fold balanced accuracy: 0.60 and AUC separation: 0.65;
- serialized-length worst-fold balanced accuracy: 0.55 and AUC separation: 0.55.

The role-scrubbed character baseline retains counts, formatting, transition-code multiplicities, and evidence value/status multisets but removes role-to-status binding. A full role-aware character baseline is reported as a legitimate relational baseline, not treated as a leak.

## Frozen diagnostic

The method is fixed before extraction:

- model: `mlx-community/Qwen3.5-0.8B-4bit`;
- adapter: none;
- representation: layer 6 mean pooling in float32;
- pointwise and pair-difference heads: class-balanced logistic regression, `C=10`, seed 0, standardized inputs;
- folds: leave one of six mechanics out;
- pointwise threshold: selected only on canonical calibration records from the five training mechanics;
- pair-difference training: both orientations of `h(ambiguous) - h(identifiable)` from label-flipping training groups;
- held-out evaluation: all records from the omitted mechanic, separately for every surface;
- maximum sequence length: 1,024 tokens; truncation is reported and invalidates interpretation if it removes the dependency schema.

The pair-difference head is diagnostic. It tests whether the frozen representation linearly encodes intervention direction; it is not a deployed pointwise classifier.

## Advancement gates

All surfaces and all leave-one-mechanic-out folds are hard requirements:

- minimum pair-difference direction accuracy: 0.60;
- mean pair-difference direction accuracy: 0.70;
- minimum pointwise-score direction accuracy: 0.60.

Absolute balanced accuracy, AUC, fixed-threshold transfer, known-true/known-false direction, and margins are reported but do not override a direction failure.

If all gates pass, V8 may advance to a separately implemented small action-conditioned structured head. If they fail, mean pooling is declared insufficient and the next development-only experiment must use token-aware action/determinant pooling. Neither result authorizes LoRA or a final-mechanic evaluation.
