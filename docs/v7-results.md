# V7 causal-evidence curriculum and tone-drift transfer result

## Decision

**NO-GO for LoRA.** The frozen 0.8B layer-6 mean probe reached 48.57% canonical balanced accuracy and 0.807 AUC on the untouched tone-drift mechanic. The context-group bootstrap interval is 42.86%–50.00%.

LoRA remains ineligible because these preregistered gates failed: holdout_canonical_balanced_accuracy, holdout_bootstrap_lower_bound, surface_entity_renamed_balanced_accuracy, surface_paraphrased_balanced_accuracy, complete_surface_triplet_accuracy, evidence_directional_accuracy, paired_score_directional_accuracy, worst_stratum_balanced_accuracy.

## Pre-model rejection gates

The maximum conditional label gap was 0.00%. Metadata-only calibration balanced accuracy was 50.00%; evidence-card-only balanced accuracy/AUC were 50.00%/0.500. All pre-model gates passed before Qwen features were read.

## Preregistered model gates

| Gate | Observed | Minimum | Result |
| --- | ---: | ---: | --- |
| development_calibration_balanced_accuracy | 76.16% | 75.00% | pass |
| holdout_canonical_balanced_accuracy | 48.57% | 70.00% | fail |
| holdout_bootstrap_lower_bound | 42.86% | 55.00% | fail |
| surface_entity_renamed_balanced_accuracy | 50.00% | 65.00% | fail |
| surface_paraphrased_balanced_accuracy | 45.71% | 65.00% | fail |
| surface_entity_renamed_prediction_agreement | 98.57% | 85.00% | pass |
| surface_paraphrased_prediction_agreement | 97.14% | 85.00% | pass |
| complete_surface_triplet_accuracy | 45.71% | 60.00% | fail |
| evidence_directional_accuracy | 0.00% | 75.00% | fail |
| paired_score_directional_accuracy | 0.00% | 75.00% | fail |
| worst_stratum_balanced_accuracy | 50.00% | 55.00% | fail |

## Corpus and firewall

The corpus contains 408 training, 172 calibration, and 70 untouched base records, each with a complete canonical/entity-renamed/paraphrased group.

Development contains 4/2 training/calibration oracle label-changing groups; tone drift contains 2. There are zero cross-split contexts, prompts, or evidence groups, zero forbidden target fields, zero V3 reads, and zero prior-holdout reads.

## Development calibration

Canonical calibration balanced accuracy was 76.16%. The method and threshold were frozen before the tone-drift records were opened.

## Surface transfer

| Surface | Balanced accuracy | AUC |
| --- | ---: | ---: |
| canonical | 48.57% | 0.807 |
| entity_renamed | 50.00% | 0.870 |
| paraphrased | 45.71% | 0.777 |

Complete-triplet accuracy was 45.71%. Canonical/entity-renamed and canonical/paraphrased agreement were 98.57% and 97.14%.

## Grouped, paired, directional, and worst-stratum metrics

The evaluation contains 41 context groups; macro context accuracy was 13.82%. The 2 oracle label-changing evidence groups produced 2 comparisons: score-directional accuracy was 0.00%, thresholded evidence-directional accuracy was 0.00%, and complete-group accuracy was 0.00%.

The worst supported stratum was `action_template=use:mirror` with 35 examples and 50.00% balanced accuracy.

## One-shot audit

- Dataset SHA-256: `c5ddff9fef1e39a8e50f7a3868c3ebb1395d43aaaa1c07886e348c16a815e3af`.
- Frozen probe SHA-256: `599002acd64b51e7467adfa6b4996e18de2e3fd8f03638a76005923f987adb51`.
- Untouched evaluations: 1.
- Untouched records scored: 210.
- Truncated prompts: 0.
- Prior holdout records read: 0.
- V3 test records read: 0.

## Revision and lock audit

The first development-only V7 construction failed its preregistered calibration gate (63.02% balanced accuracy versus the 75.00% minimum). Its untouched partition was never read or model-scored. The curriculum was therefore revised using development evidence only, and the final run used fresh development seeds 9601–9612 and fresh reserve seeds 9801–9804.

Before the first evaluation, probe-lock generation exposed a stale pathname default (`data/v7` instead of `data/v7r2`). Only that pathname was corrected in the lock; the preregistered untouched-record hash, dataset, model, probe, threshold, and implementation hashes were unchanged, and no reserve model score existed. The corrected lock hash is `d7ba80414deb49c853745a9f0a31ec10c464f7d39df83318897b7053cb647ca6`, which is the lock recorded by the sole evaluation.
