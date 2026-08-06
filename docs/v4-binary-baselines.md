# V4 binary token-baseline ablations

Thresholds are fitted once on the context-disjoint calibration fold. Validation is not
used for fitting or variant selection. V3 test remains closed.

| Input | Calibration balanced | Validation balanced | Validation F1 | Validation AUC |
| --- | ---: | ---: | ---: | ---: |
| Full visible input | 54.33% | 58.24% | 62.38% | 0.714 |
| Remove turn | 54.33% | 58.24% | 62.38% | 0.710 |
| Remove pressure + signal | 54.33% | 58.24% | 62.38% | 0.697 |
| Remove history + memories | 54.97% | 76.92% | 75.00% | 0.728 |
| Remove turn + pressure + signal | 54.33% | 58.24% | 62.38% | 0.697 |
| Remove scalars + history | 54.59% | 67.03% | 67.74% | 0.700 |
