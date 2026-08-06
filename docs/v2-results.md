# Dataset v2 and model results

Run date: 2026-08-05. Hardware: Apple M4 Pro with 64 GB unified memory.

## What changed from v1

Dataset v1 assigned the transition from one hidden world to an agent prompt that often did not
identify that world. It also allowed identical prompts to cross splits. Dataset v2 changes the
scientific task, not merely the file format:

- Agent records are exact observational-equivalence classes, each labeled as identifiable or
  ambiguous and paired with its complete empirical possible-transition set.
- All transition-relevant scenario rules are included in the privileged input.
- Observation-state context groups are split before candidate actions are attached, keeping all
  counterfactual actions from one context in one split.
- Validators reject prompt overlap, context overlap, contradictory privileged targets, and
  malformed equivalence classes.

The generated corpus has SHA-256
`07f633be63abce4b1fd92ec1c260fda18082494c759d117aa081773ceed5d6e0`.

## Corpus audit

| Track | Records | Train | Validation | Test | Cross-split prompts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agent epistemic | 1,525 | 1,223 | 145 | 157 | 0 |
| Privileged transition | 10,038 | 7,974 | 1,028 | 1,036 | 0 |

Of the unique agent prompts, 913 (59.87%) are identifiable and 612 (40.13%) are
ambiguous. The privileged track has zero identical full inputs with contradictory targets.
The context-disjoint hash split is not class-stratified: ambiguity is 39.08% in training, 64.14%
in validation, and 26.11% in test. Metrics must therefore be stratified, and this imbalance is the
main remaining dataset gate.

## Deterministic baselines

The strongest agent baseline depends on the metric: action majority reaches 42.68% exact target
match, while nearest neighbour reaches 61.61% micro-F1 over individual possible outcomes. Every
baseline has 0% exact match on ambiguous examples because each emits one transition.

On the privileged track, nearest neighbour reaches 83.49% exact transition match and 98.33%
macro field accuracy. Because test prompts are disjoint from training, exact-prompt lookup falls
back to the 47.88% action-majority result instead of measuring duplication.

## Qwen3.5 experiments

### 9B, 100 updates

The eight-layer 9B LoRA run trained 5.410M parameters and used 54.39 GB peak unified memory. Its
validation loss fell from 0.474 to 0.067 at step 75, then rose to 0.079 at step 100. The adapter
produced valid exact-schema JSON, but a 40-example partial generation predicted every case as
identifiable, including all 22 ambiguous examples encountered. This is a useful negative result:
low teacher-forced loss after roughly 8% of an epoch did not imply learned uncertainty
preservation.

### 4B, extended run

The follow-up used 16 LoRA layers (8.116M trainable parameters). Validation loss fell from 0.422
initially to 0.044 at step 100 and was 0.046 at the saved step-200 checkpoint; peak unified-memory
allocation was 53.79 GB. On the same 40-example probe, it also emitted exactly one outcome for all
18 identifiable and 22 ambiguous prompts. Longer teacher-forced training did not fix the branch
collapse.

### Outcome-count calibration

The next objective separates calibration from transition generation. It maps the same agent input
to the number of empirically supported transitions. Always predicting one is a strong
class-imbalanced baseline: 73.89% overall count accuracy, but 25.00% macro accuracy across observed
counts and 0% exact accuracy on ambiguous prompts.

A compact JSON target still collapsed because one meaningful number was surrounded by fixed JSON
tokens. The final calibration target is therefore one digit from 1 through 5. A 0.8B Qwen3.5
adapter trains this target with 3.608M LoRA weights and 6.28 GB peak unified memory.

| Generated test result | Count exact | Macro count | Ambiguous exact | Balanced ID | Ambiguity F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always-one baseline | 73.89% | 25.00% | 0.00% | 50.00% | 0.00% |
| Unbalanced 0.8B, step 200 | 73.89% | 25.00% | 0.00% | 50.00% | 0.00% |
| Unbalanced 0.8B, step 400 | 55.41% | 31.54% | 43.90% | 55.35% | 38.53% |
| Unbalanced continuation, effective step 600 | 73.89% | 25.00% | 0.00% | 50.00% | 0.00% |
| Balanced 0.8B, step 200 | 19.75% | 23.02% | 56.10% | 46.13% | 38.04% |
| Balanced 0.8B, step 400 | 73.89% | 25.00% | 0.00% | 50.00% | 0.00% |

The unbalanced step-400 checkpoint is the only test checkpoint above the trivial 50% balanced-ID
boundary. It correctly detects 21 of 41 ambiguous prompts but makes 47 false ambiguity calls.
Crucially, it reaches only 44.47% balanced ID accuracy on the full validation set, so it cannot be
selected as a robust model without consulting test labels. The balanced run oscillates from
almost-all-two at step 200 to all-one at step 400. Validation-selected linear adapter blends also
collapse to all-one and do not exceed 50% balanced accuracy.

## Interpretation

Dataset v2 supports prompt-disjoint evaluation and exposes failures that v1 would have hidden: a
model can learn the output schema and frequent deterministic transition patterns without learning
when its observation history underdetermines the next state, while an apparently improved test
checkpoint can fail on a class-shifted validation split. The decisive metrics are generated
ambiguity recall/precision and ambiguous-example outcome-set exact match, not loss alone.

The next compiler revision should assign whole context groups with stratification or constrained
balancing so train, validation, and test have comparable ambiguity and mechanic distributions.
Training should evaluate the complete fixed validation set rather than randomly sampled batches.
Only after that gate should modeling continue with the two-stage design: calibrate outcome count,
then train conditional outcome generation with that count supplied explicitly. Broader procedural
worlds, renamed entities, and held-out mechanic combinations remain the next generalization gate.
