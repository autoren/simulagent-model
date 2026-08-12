# V5 locked shortcut-challenge results

## Decision

The preregistered frozen-probe challenge **fails**. The locked canonical probe reaches 49.58% balanced accuracy and 0.457 AUC on new simulator worlds. Its context-group bootstrap interval is 47.62%–51.59%.

## Preregistered gates

| Gate | Observed | Minimum | Result |
| --- | ---: | ---: | --- |
| canonical_balanced_accuracy | 49.58% | 75.00% | fail |
| mechanic_powertrip_balanced_accuracy | 50.00% | 65.00% | fail |
| mechanic_relockshort_balanced_accuracy | 52.28% | 65.00% | fail |
| surface_entity_renamed_balanced_accuracy | 46.37% | 70.00% | fail |
| surface_paraphrased_balanced_accuracy | 49.04% | 70.00% | fail |
| surface_entity_renamed_prediction_agreement | 80.83% | 85.00% | fail |
| surface_paraphrased_prediction_agreement | 94.17% | 85.00% | pass |
| evidence_directional_accuracy | 100.00% | 75.00% | pass |

## Surface robustness

| Surface | Balanced accuracy | AUC |
| --- | ---: | ---: |
| canonical | 49.58% | 0.457 |
| entity_renamed | 46.37% | 0.470 |
| paraphrased | 49.04% | 0.453 |

Canonical/entity-renamed prediction agreement is 80.83%; canonical/paraphrased agreement is 94.17%. All three surfaces are simultaneously correct for 36.67% of base records.

## Held-out mechanics

| Mechanic | Canonical balanced accuracy | AUC |
| --- | ---: | ---: |
| powertrip | 50.00% | 0.427 |
| relockshort | 52.28% | 0.458 |

## Evidence contrasts

The two simulator-derived evidence groups contain 9 cross-label comparisons. Directional accuracy is 100.00%, and complete group classification is 50.00%. Because both groups come from the short-start relock family, this remains a narrow diagnostic rather than a broad evidence-rung generalization claim.

## Firewall

- Frozen lock SHA-256: `77681c50b74a9ea644e3fd9f7deb865342ec9cad157e8845a136ba2a43b37532`.
- Challenge dataset SHA-256: `ddf04fcf163d05db68d8a72d13094d958cccc22c8c6bf6a4e6df12e04e6778ec`.
- Records / base records / context groups: 360 / 120 / 63.
- Truncated prompts: 0.
- Challenge evaluations: 1.
- V3 test records read: 0.
