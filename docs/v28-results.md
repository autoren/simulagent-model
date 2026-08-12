# V28 results: marginal program MAP

Decision: `marginal_program_map_improves_support_continue_query_repair_no_lora`.

V28 reused V27's complete frozen support candidate set and changed only program
selection from joint/Viterbi MAP to marginal MAP. It made zero model calls and fit
no parameter or threshold. Query predictions are byte-identical to V27.

## V27 to V28

| Metric | V27 | V28 |
|---|---:|---:|
| Evaluation exact support graph | 0.694 | 0.694 |
| Frozen support / oracle query exact | 0.744 | 0.756 |
| Frozen / frozen exact | 0.718 | 0.731 |
| Target retention | 0.833 | 0.917 |
| Empty version space | 0.000 | 0.000 |

## Marginal diagnostics

Evaluation target-program top-1 rate: 0.917.
Median target-program rank: 1.0.
Compatibility-state deduplication: 0.547.
Post-result integrity audit: `pass`.
