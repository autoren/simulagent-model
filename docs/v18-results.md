# V18 development results: executable transition-schema induction

The development protocol and exact symbolic baseline pass every gate. This authorizes
integration with the already frozen V15 language-grounding pipeline; it does not authorize
LoRA or construction of a new final mechanic.

Decision: `authorize_frozen_grounding_integration`.

## Corpus and firewall

- 72 episodes and 3,056 queries;
- 24 training, 8 calibration, and 40 development episodes;
- eight development episodes on each of five isolated axes;
- no action-dependency table or target expression field in any agent input; and
- zero V17 record reads, V17 result reads, adapter runs, or new final-mechanic constructions.

## Full-support baselines

| Metric | Exact DSL | Depth-3 tree | Conditional union | Literal lookup | Program prior |
|---|---:|---:|---:|---:|---:|
| Transition-set exact match | 1.000 | 0.574 | 0.273 | 0.070 | 0.000 |
| Identifiability balanced accuracy | 1.000 | 0.766 | 0.464 | 0.512 | 0.500 |
| Outcome-invariant unknown accuracy | 1.000 | 0.705 | 0.632 | 0.043 | 0.000 |
| Outcome-sensitive unknown accuracy | 1.000 | 0.456 | 0.135 | 0.135 | 0.000 |

Exact program execution equivalence is 1.000; relevant-determinant exact match is 1.000. All 40/40 development episodes are completely correct, and the inducer needs 4–8 traces per episode (mean 5.56) under the target-conditioned support policy.

## Development axes

| Axis | Episodes recovered | Complete episodes | Queries | Transition-set exact |
|---|---:|---:|---:|---:|
| `known_primitive_recombination` | 8/8 | 8/8 | 340 | 1.000 |
| `structural_composition` | 8/8 | 8/8 | 329 | 1.000 |
| `determinant_vocabulary` | 8/8 | 8/8 | 341 | 1.000 |
| `composition_depth` | 8/8 | 8/8 | 339 | 1.000 |
| `outcome_invariance` | 8/8 | 8/8 | 341 | 1.000 |

## Minimal-support curve

The curve uses development episodes only. `full` is each episode's greedily selected
behavior-identifying trace set, not the complete 16-row truth table.

| Support budget | Transition-set exact | Identifiability BA | Median version space | Unique target episodes |
|---|---:|---:|---:|---:|
| 1 | 0.000 | 0.500 | 1600.0 | 0.000 |
| 2 | 0.047 | 0.500 | 266.0 | 0.000 |
| 4 | 0.406 | 0.658 | 6.0 | 0.025 |
| 8 | 1.000 | 1.000 | 1.0 | 1.000 |
| full | 1.000 | 1.000 | 1.0 | 1.000 |

## Semantic split and support-policy audit

Complete truth-table hashes have zero training overlap for recombination, structure, depth,
and invariance. The vocabulary axis intentionally reuses all eight corresponding training
behaviors so that its symbolic factor is held fixed.

With transition codes masked, the version space contains 12,996 behavioral programs and never uniquely recovers the target. An ordered assignment-only nearest-training control recovers 0.200 of development behavior signatures and 0.400 of relevant-determinant sets. Its leave-one-out development-axis accuracy is 0.200 versus 0.200 chance.

These controls do not remove the side channel: the greedy assignment schedule is selected
using target outcomes and remains an oracle intervention policy. The sample-efficiency curve
therefore applies only under that selected-support policy.

## Interpretation and next gate

The baseline gap establishes generalization beyond literal observed-assignment lookup; it
does not prove that every non-symbolic learner must fail. A correctly specified executable
hypothesis class is sufficient across recombination, nested structure, held-out determinant
vocabulary under oracle grounding, greater depth, and non-injective mechanics.

The next eligible experiment replaces target-side oracle groundings with outputs from the
frozen V15 grounding pipeline while leaving the corpus, support traces, DSL search, executor,
and metrics unchanged. A learned proposal model is warranted only if grounding integration
or exact search becomes the demonstrated bottleneck.
