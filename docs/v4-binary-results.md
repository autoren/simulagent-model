# V4 binary identifiability results

Each seed selected its LoRA checkpoint and A/B logit threshold only on the
context-disjoint calibration fold. The selected pair was then evaluated once on V3
validation. V3 test remained closed.

| Seed | Step | Threshold | Calibration balanced | Validation balanced | Validation F1 | Validation AUC | Predictions (I/A) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 200 | 0.312 | 51.66% | 49.15% | 47.74% | 0.491 | 62/92 |
| 1 | 300 | -0.062 | 50.41% | 48.96% | 54.17% | 0.492 | 25/129 |
| 2 | 200 | -0.438 | 54.21% | 55.49% | 51.06% | 0.555 | 76/78 |

**Engineering stability gate: FAIL**

**Scientific token-baseline gate: FAIL**

Mean validation balanced accuracy: 51.20%.
Primary full-input token baseline: 58.24%.
