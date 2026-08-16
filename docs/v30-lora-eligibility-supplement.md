# V30 LoRA-eligibility supplement

Status: post-result descriptive supplement. This document does not alter the locked V30 protocol,
gates, predictions, or decision.

V30 registered LoRA eligibility only when the structural audit passed, both the primary extractor
and oracle-atom NLI diagnostic failed, at least two evaluation surface families of one supported
operator fell below 0.90 truth accuracy in both systems, aggregate oracle-atom NLI truth top-two
retention fell below 0.98, and evaluation selected no method or threshold.

Every condition passed. The shared family failures were:

| Operator | Shared families below 0.90 truth accuracy |
|---|---|
| `double_negation` | `eval_a`, `eval_b`, `eval_c` |
| `negated_opposite` | `eval_b`, `eval_c` |

The oracle-atom NLI diagnostic reached 0.874 truth accuracy and 0.956 aggregate top-two retention.
Its sealed family-level top-two retention was:

| Surface family | Correct truth in top two |
|---|---:|
| `affirmative_gold.eval_a` | 1.000 |
| `affirmative_gold.eval_b` | 0.962 |
| `affirmative_gold.eval_c` | 1.000 |
| `contrastive_both.eval_a` | 1.000 |
| `contrastive_both.eval_b` | 1.000 |
| `contrastive_both.eval_c` | 1.000 |
| `double_negation.eval_a` | 0.769 |
| `double_negation.eval_b` | 0.923 |
| `double_negation.eval_c` | 0.923 |
| `explicit_unknown.eval_a` | 1.000 |
| `explicit_unknown.eval_b` | 1.000 |
| `explicit_unknown.eval_c` | 1.000 |
| `negated_opposite.eval_a` | 1.000 |
| `negated_opposite.eval_b` | 0.923 |
| `negated_opposite.eval_c` | 0.846 |

These values make the eligibility decision independently legible. They authorize only a new,
separately preregistered adaptation study. They do not establish that backbone adaptation is
necessary and do not authorize an adapter-trained V28 replay.
