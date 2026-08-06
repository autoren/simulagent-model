# Deterministic baseline results

Track: `privileged`. Evaluated on all 1,036 test transitions using 7,974 training transitions.

| Baseline | Exact match | Changed targets | Unchanged targets | Macro field accuracy |
| --- | ---: | ---: | ---: | ---: |
| No change | 32.82% | 0.00% | 88.77% | 85.47% |
| Action majority | 47.88% | 32.31% | 74.41% | 93.56% |
| Exact-prompt lookup | 47.88% | 32.31% | 74.41% | 93.56% |
| Nearest neighbour | 83.49% | 76.72% | 95.04% | 98.33% |

Exact-prompt training coverage of test: 0.00%.
When this coverage is high, exact-prompt lookup measures input duplication rather than
generalization. The no-change result shows why changed and unchanged targets must always be
reported separately.
