# V13 results: 4B token-local relation diagnostic

## Verdict

Token-local pooling substantially improves construction transfer, but it does not satisfy the locked worst-family gate. All three heads are perfect on the context fold and most held-out construction families; all three are also exactly reversed on held-out Direct Assertion, with 0.000 accuracy and 0.000 ROC AUC.

The decision is `token_local_frozen_readout_insufficient_stop_probes_redesign_supervision`. Frozen feature probing stops here. The next phase should redesign supervision and evaluation around semantic-operator support rather than try another model scale, layer, pooling rule, or post-hoc classifier.

This result is more informative than a generic negative. The hypothesis-mean linear head reaches at least 0.751 on every non-Direct fold and at least 0.977 on five of six held-out template families. The single 0.000 fold is explained by a structural support gap in the current template taxonomy.

## Locked V13 results

| Head | Context accuracy | Worst fold accuracy | Worst surface accuracy | Convergence warnings |
| --- | ---: | ---: | ---: | ---: |
| Last-token linear | 1.000 | 0.000 | 0.000 | 0 |
| Hypothesis-mean linear | 1.000 | 0.000 | 0.000 | 0 |
| Token-joint 32-unit MLP | 1.000 | 0.000 | 0.000 | 0 |

Required minima were 0.70 in every fold and 0.65 in every fold-by-surface cell. None passes because Direct Assertion is exactly inverted.

| Held-out template | Last linear acc / AUC | Mean linear acc / AUC | Joint MLP acc / AUC |
| --- | ---: | ---: | ---: |
| Contrastive Correction | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Denied Claim | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Direct Assertion | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| Explicit Negation | 0.962 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Rejected Claim | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Scoped Rejection | 0.629 / 0.630 | 0.977 / 1.000 | 0.749 / 0.786 |

The hypothesis-mean representation is the strongest frozen polarity representation tested so far. It solves Contrastive Correction, Denied Claim, Explicit Negation, Rejected Claim, and nearly all Scoped Rejection examples at 4B. The MLP adds no robust benefit and the last token is weaker on Scoped Rejection.

## Operator-support audit

For each current determinant, the audit asks whether the gold evidence literally contains the gold hypothesis, the opposite hypothesis, or both. Counts cover all 7,380 current determinants.

| Template | Gold only | Opposite only | Both | Signature absent when held out |
| --- | ---: | ---: | ---: | --- |
| Contrastive Correction | 0 | 0 | 1230 | both |
| Denied Claim | 0 | 1230 | 0 | none |
| Direct Assertion | 1230 | 0 | 0 | gold_only |
| Explicit Negation | 0 | 1230 | 0 | none |
| Rejected Claim | 0 | 1230 | 0 | none |
| Scoped Rejection | 0 | 1230 | 0 | none |

Direct Assertion is the only `gold_only` family. When it is held out, training contains zero examples in which the literally mentioned hypothesis is the true one: four remaining families mention the opposite hypothesis under rejection/negation, and Contrastive Correction mentions both. Likewise, Contrastive Correction is the only `both` family, although the token-local representation happens to generalize to it.

Therefore the current `template` fold conflates two tests:

1. surface-form transfer within a known logical operator; and
2. zero-shot transfer to an entirely absent semantic mention-orientation operator.

The Direct Assertion zero is valid evidence that the supervised head did not perform zero-shot logical transfer. It is not evidence that a broader classifier architecture would fix the issue, and it should not be treated as an ordinary paraphrase failure.

## Correct V14 direction

V14 should rebuild the linguistic taxonomy before any additional model training:

- define semantic operator signatures such as affirmative assertion (`gold_only`), negated/rejected opposite (`opposite_only`), and contrastive correction (`both`);
- create at least two independently worded surface families per operator signature;
- use primary folds that hold out a surface family while retaining the same operator signature in training;
- report a separate, explicitly labeled zero-shot operator-holdout benchmark, never mixing it into the paraphrase-transfer gate;
- balance temporal operators independently so temporal holdouts have analogous support;
- retain the exact symbolic evaluator and protected-data firewall.

The strongest fixed baseline for that redesigned corpus should be 4B layer-8 hypothesis-mean signed comparison. Only after it passes supported surface holdouts should a separately locked adapter objective be considered. LoRA is not authorized by V13 failure.

## Reproducibility

- V13 protocol lock: `bdecdfbd806a93de88624dd18c65668602928795f61bdb54ae06624bef2965fd`;
- token-local feature artifact: `bd30db46cd4d3180457e3f7fdee3c56c2bd0d99ad4ce77e99ff17596602dae8a`;
- V13 result: `a7a00ee19b3f31e5945a4d7ccb223b02e3e12d2d0977343d587c1a8d68ad70a0`;
- operator-support audit: `d38bfd9a06b3094daa4178df9c31d9d6b2a3dedcb89003e58a8d6b003be8b255`;
- prompt count/hash: 6,984 / `3b5475a39376f7c057afbaa01285f3593c328b5879b2cc73a9dd46ac6dbb69ce`;
- fitted heads: 72 (three heads × 24 folds);
- convergence warnings, truncations, adapter runs, and protected-data accesses: zero.
