# V29 results: posterior-marginal support graph decoding

Decision: `posterior_graph_decoding_insufficient_revisit_language_scores_no_lora`.

V29 changed only the Bayes decision rule for support graphs. It integrated over the
shared program and all other support graphs, emitted one graph per scene, made zero
model calls, and reused query predictions byte-for-byte.

| Metric | V28 | V29 |
|---|---:|---:|
| Evaluation exact support graph | 0.694 | 0.611 |
| Frozen support / oracle query exact | 0.756 | 0.577 |
| Frozen / frozen exact | 0.731 | 0.551 |
| Target retention | 0.917 | 0.917 |
| Empty version space | 0.000 | 0.000 |

Mean selected evaluation graph posterior: 0.334.
Post-result integrity audit: `pass`.
