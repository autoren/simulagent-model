# V11 results: frozen-scale polarity capacity diagnostic

## Verdict

Frozen scale alone does not solve V10's construction-transfer failure. Both pinned larger backbones remain perfect on the context fold, and both now pass every span gate, but neither exposes construction-independent current-state polarity through the locked NLI-final linear readout.

The minimum oracle-span/oracle-temporal polarity accuracy and hypothesis-pair consistency are 0.000 for 4B and 9B, exactly as in 0.8B. The result is not monotonic evidence for a capacity threshold: 4B improves selected template cells, while 9B regresses on several of them. LoRA and final-mechanic access remain closed.

The locked next decision is to test a frozen joint/nonlinear token-aware relation readout before adapting model weights.

## Locked comparison

V11 reused V10's 3,240 records, exact target arrays, 3,492 base prompts, 6,984 NLI prompts, three readouts, 24 folds, four oracle ablations, linear-head settings, gates, and deterministic symbolic evaluator. Only the frozen backbone and the preregistered homologous depth changed.

V10 layer 6 of 24 fixed the relative depth at 25%. V11 therefore used layer 8 of 32 for both larger checkpoints. The exact 4-bit model revisions were pinned before access, and no alternative layer was extracted.

| Backbone | Context oracle polarity | Minimum span | Minimum temporal | Minimum oracle polarity | Minimum pair consistency | Minimum full allowed values | Minimum symbolic BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B, layer 6/24 | 1.000 | 0.628 | 0.102 | 0.000 | 0.000 | 0.167 | 0.497 |
| 4B, layer 8/32 | 1.000 | 0.867 | 0.160 | 0.000 | 0.000 | 0.186 | 0.500 |
| 9B, layer 8/32 | 1.000 | 0.857 | 0.181 | 0.000 | 0.000 | 0.186 | 0.500 |

Both larger models pass the aggregate and surface span gates:

- 4B minimum fold/surface span: 0.867 / 0.803;
- 9B minimum fold/surface span: 0.857 / 0.718.

They fail the other twelve gates, including temporal transfer, oracle polarity, pair consistency, allowed ledgers, symbolic balanced accuracy, flip pairs, and complete intervention groups.

## Held-out language families

| Template | 0.8B temporal | 4B temporal | 9B temporal | 0.8B oracle polarity | 4B oracle polarity | 9B oracle polarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contrastive Correction | 0.102 | 0.160 | 0.181 | 0.000 | 0.000 | 0.000 |
| Denied Claim | 0.964 | 1.000 | 1.000 | 0.070 | 0.632 | 0.158 |
| Direct Assertion | 0.648 | 0.193 | 0.198 | 0.015 | 0.000 | 0.000 |
| Explicit Negation | 0.624 | 0.981 | 0.983 | 0.433 | 0.175 | 0.000 |
| Rejected Claim | 0.119 | 0.164 | 0.233 | 0.088 | 0.029 | 0.026 |
| Scoped Rejection | 0.195 | 0.957 | 0.186 | 0.608 | 0.807 | 0.167 |

The direct diagnostics do not reveal a hidden monotonic scale effect either:

| Backbone | Mean-direct minimum oracle polarity | Span-direct minimum oracle polarity |
| --- | ---: | ---: |
| 0.8B | 0.477 | 0.170 |
| 4B | 0.336 | 0.076 |
| 9B | 0.193 | 0.053 |

## Interpretation

Scale clearly improves determinant/evidence matching: the worst primary surface span rises from 0.458 in V10 to 0.803 at 4B and 0.718 at 9B. That validates the comparison's ability to detect a genuine capacity-related change.

The polarity result is different. Even with the correct evidence and temporal status, each larger model has at least one held-out family for which every current-state decision is wrong or unresolved. Direct Assertion remains nearly perfectly pair-consistent while oriented in the wrong direction, and Contrastive Correction remains completely unresolved. The independent-hypothesis linear head is therefore the leading bottleneck after scale is controlled.

Temporal classification also remains construction-bound. Contrastive Correction, Rejected Claim, and—in 9B—Scoped Rejection are often treated as non-current despite gold spans. This is a separate operator-transfer problem, but it cannot explain the oracle-polarity failure.

The appropriate next diagnostic should reuse the saved frozen features and evaluate the active/inactive hypothesis pair jointly. A fixed joint linear head followed by a small preregistered nonlinear head can test whether relation information is present but not independently linearly separable. Token-span interaction features should be extracted only if both joint heads fail. This sequence is cheaper and more diagnostic than LoRA.

## Decision and firewall

Combined decision: `frozen_scale_insufficient_test_nonlinear_token_aware_readout`.

No adapter, alternate layer, final mechanic, Tone Drift, V3 test record, prior holdout, untouched V8 mechanic, or V7 model result was accessed. V11 authorizes neither LoRA nor final evaluation.

## Reproducibility

- V11 protocol lock: `d28e99579a6774a2ebb028fa33dadcfa59c861c4f4bd57eb70d08ae6467790af`;
- 4B features: `666ba5027ae654defe74fc673e1e073a83116dc3656acdd4194ab9c5e452811c`;
- 9B features: `00e082ee35a3bc3adb58f6f97bea873b9aa0eb7f554cea3953eb04aa193d070d`;
- 4B result: `27320f2110bd04e1594dac7350b65383eb0e77bf91423f49cf459050ca944ae5`;
- 9B result: `b1bff0f28b50496caef07d2403f08e8aef1132aeed379cb0cd7d3e8ae9cbe731`;
- combined result: `84c5685b4d4071c73981477189c600e798832ce557bc40b04c240b296c1e7ba7`.
