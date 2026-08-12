# V19 results: frozen grounding × executable schema induction

The locked supported-language integration passes every preregistered gate. The frozen
V15 grounder and unchanged V18 schema inducer complete all 40 development episodes exactly.
This authorizes design of a fresh multi-mechanic final suite; it does not authorize LoRA.

Decision: `authorize_fresh_multi_mechanic_final_suite_design`.

## Grounding views

| View | Role | Allowed values | Exact scenes | All-support episodes |
|---|---|---:|---:|---:|
| `supported` | primary | 1.000 | 1.000 | 1.000 |
| `novel_ontology` | diagnostic | 0.969 | 0.875 | 0.700 |

The primary view is perfect for active, inactive, and unresolved values. The diagnostic
view retains perfect span and temporal accuracy, but current polarity falls to 0.963; inactive-value accuracy is 0.925.

## Supported-view two-by-two decomposition

| Condition | Episode macro | Complete episodes | Target retained | Empty version |
|---|---:|---:|---:|---:|
| `frozen_support_frozen_query` | 1.000 | 40/40 | 1.000 | 0.000 |
| `frozen_support_oracle_query` | 1.000 | 40/40 | 1.000 | 0.000 |
| `oracle_support_frozen_query` | 1.000 | 40/40 | 1.000 | 0.000 |
| `oracle_support_oracle_query` | 1.000 | 40/40 | 1.000 | 0.000 |

## Novel-ontology diagnostic

| Condition | Episode macro | Complete episodes | Target retained | Empty version |
|---|---:|---:|---:|---:|
| `frozen_support_frozen_query` | 0.740 | 28/40 | 0.750 | 0.225 |
| `frozen_support_oracle_query` | 0.734 | 28/40 | 0.750 | 0.225 |
| `oracle_support_frozen_query` | 0.919 | 29/40 | 1.000 | 0.000 |
| `oracle_support_oracle_query` | 1.000 | 40/40 | 1.000 | 0.000 |

Support grounding is the larger novel-ontology failure mode: frozen-support/oracle-query
episode accuracy is 0.734, versus 0.919 with oracle supports and frozen queries. The full diagnostic reaches 0.740, retains the target behavior in 0.750 of episodes, and invokes the locked empty-version rule in 0.225.

The scope-correct error replay finds 28 zero-support-error episodes, all perfect. The one episode with one support error scores 0.000; the 11 episodes with multiple support errors average 0.144, retain the target in 0.182, and produce an empty version space in 0.727.

## Development axes in the novel diagnostic

| Axis | Episode macro | Complete episodes | Target retained | Empty version |
|---|---:|---:|---:|---:|
| `composition_depth` | 0.625 | 5/8 | 0.625 | 0.375 |
| `determinant_vocabulary` | 1.000 | 8/8 | 1.000 | 0.000 |
| `known_primitive_recombination` | 0.695 | 5/8 | 0.750 | 0.250 |
| `outcome_invariance` | 0.686 | 5/8 | 0.750 | 0.250 |
| `structural_composition` | 0.692 | 5/8 | 0.625 | 0.250 |

The V18 `determinant_vocabulary` category is 8/8 here, while other fresh lexicons expose
polarity errors. This is a diagnostic observation, not a tuned vocabulary selection.

## Reproducibility and firewall

- 6,912 grounding scenes over two exactly paired views;
- 240 unique base prompts and 480 unique NLI prompts;
- one locked extraction with 720 model forward passes and no truncation;
- one locked integration evaluation;
- frozen deployment heads reproduced bit-for-bit from V15-only development features;
- no adapter training, head selection/refitting, target-guided repair, support deletion, or DSL expansion;
- zero V17 record or model-result reads; and
- post-result replay reproduces all 6,912 saved grounding predictions and eight integration condition reports.

The locked evaluator's non-gating grounding-error histograms initially paired determinant
lists by serialization order. A zero-forward-pass scope-correct replay joins by determinant
id and supersedes only those three diagnostic fields. Grounding predictions, schema search,
query answers, gates, and the primary decision are unchanged.

## Next decision

Freeze the design of a genuinely fresh multi-mechanic final suite with mechanics as the
sampling unit. It should include one- and two-bit outcomes, injective and non-injective
tables, multiple depths, and read-only plus state-changing actions. The supported-language
view is the primary preregistered condition; novel ontology should remain a separately
reported transfer diagnostic. V17 remains exposed and cannot be reused.
