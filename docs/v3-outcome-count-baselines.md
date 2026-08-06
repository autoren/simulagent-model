# Dataset v3 outcome-count validation baselines

Prompt-disjoint valid set: 154 unique agent prompts.

| Baseline | Exact count | Macro count | Ambiguous exact | Balanced ID | Ambiguity F1 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always one | 59.09% | 20.00% | 0.00% | 50.00% | 0.00% | 0.545 |
| Action count majority | 59.09% | 20.00% | 0.00% | 50.00% | 0.00% | 0.545 |
| Nearest neighbour | 38.96% | 16.53% | 26.98% | 38.71% | 29.23% | 0.786 |
| Token Naive Bayes | 74.03% | 31.55% | 52.38% | 73.08% | 66.06% | 0.377 |
