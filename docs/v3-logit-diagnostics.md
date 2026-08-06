# V3 count-logit diagnostics

These are secondary diagnostics on the same validation data used to inspect checkpoints.
Post-hoc thresholds are optimistic and are not eligible for checkpoint selection or the
calibration gate. ROC AUC measures ranking independently of the default digit argmax.

| Seed | Step | ROC AUC | Post-hoc best balanced ID |
| ---: | ---: | ---: | ---: |
| 0 | 100 | 0.474 | 50.00% |
| 0 | 200 | 0.361 | 50.00% |
| 0 | 300 | 0.536 | 55.43% |
| 0 | 400 | 0.550 | 59.52% |
| 1 | 100 | 0.428 | 50.00% |
| 1 | 200 | 0.465 | 50.55% |
| 1 | 300 | 0.479 | 50.00% |
| 1 | 400 | 0.563 | 56.47% |
| 2 | 100 | 0.518 | 52.26% |
| 2 | 200 | 0.560 | 55.98% |
| 2 | 300 | 0.458 | 50.61% |
| 2 | 400 | 0.528 | 52.32% |
