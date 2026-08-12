# V6 shortcut-resistant corpus and mechanic-transfer result

## Decision

**NO-GO for LoRA.** The fixed 0.8B layer-6 mean probe reached 63.16% canonical balanced accuracy and 0.592 AUC on the untouched mirror-rejection mechanic. The context-group bootstrap interval is 51.55%–82.44%.

## Preregistered gates

| Gate | Observed | Minimum | Result |
| --- | ---: | ---: | --- |
| development_calibration_balanced_accuracy | 90.32% | 75.00% | pass |
| holdout_canonical_balanced_accuracy | 63.16% | 70.00% | fail |
| holdout_bootstrap_lower_bound | 51.55% | 55.00% | fail |
| surface_entity_renamed_balanced_accuracy | 69.74% | 65.00% | pass |
| surface_paraphrased_balanced_accuracy | 57.89% | 65.00% | fail |
| surface_entity_renamed_prediction_agreement | 90.79% | 85.00% | pass |
| surface_paraphrased_prediction_agreement | 94.74% | 85.00% | pass |
| complete_surface_triplet_accuracy | 56.58% | 60.00% | fail |
| absolute_improvement_over_v5_challenge | 13.58% | 15.00% | fail |
| evidence_directional_accuracy | 0.00% | 75.00% | fail |

## Corpus audit

The corpus contains 143 training, 49 calibration, and 76 holdout base records, each with three surface views. Training uses short-start relock and power-trip; mirror rejection is reserved for the one-shot mechanic holdout.

Training/calibration include 18/8 evidence-intervention groups. The holdout includes 16 groups, of which 4 change the binary label.

Leakage audit: zero cross-split contexts, zero cross-split prompts, zero overlap with V4 development or V5 challenge prompts/scenarios, zero privileged target fields, and zero V3 test reads.

## Development calibration

Canonical calibration balanced accuracy was 90.32%; no layer, pooling, regularization, model size, or threshold source was selected on the mechanic holdout.

## Surface transfer

| Surface | Balanced accuracy | AUC |
| --- | ---: | ---: |
| canonical | 63.16% | 0.592 |
| entity_renamed | 69.74% | 0.639 |
| paraphrased | 57.89% | 0.538 |

Complete-triplet accuracy was 56.58%. Canonical/entity-renamed agreement was 90.79%; canonical/paraphrased agreement was 94.74%.

## Evidence interventions

The label-changing holdout groups contain 12 cross-label comparisons. Directional accuracy was 0.00%, and complete-group classification was 0.00%.

## Failure diagnosis

The transfer gain does not represent a stable evidence rule. In development, all 36 canonical
`announced`, `announced-consequence`, and `announced-procedure` records are ambiguous. In the
mirror-rejection holdout, all 32 canonical consequence/procedure records are identifiable. Their
mean ambiguity scores are 4.19 and 9.28, both above the locked 3.39 threshold. The probe therefore
learned an evidence-wording correlation and applied it in the wrong direction on the new mechanic.

V7 must counterbalance evidence wording by identifiability within each mechanic/action template
and reject the corpus before model access when conditional label-rate gaps are large. LoRA remains
ineligible; exact-count and transition-target supervision remain gated.

## Firewall

- V6 dataset SHA-256: `1fd29263b8aa262b2153aacb8d753fc942b43b1f6a869ec6c215c8690a61186d`.
- Frozen probe SHA-256: `b32f3aba28234b80bf764da7316f41a0bf4d805acc71ead2dc150100c4925e73`.
- Holdout evaluations: 1.
- Holdout records scored: 228.
- Truncated prompts: 0.
- V3 test records read: 0.
