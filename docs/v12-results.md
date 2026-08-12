# V12 results: frozen joint-hypothesis readout

## Verdict

Jointly comparing the active and inactive hypotheses does not repair construction transfer at any tested scale. The signed-difference linear comparator fails the locked gate for 0.8B, 4B, and 9B, so the preregistered joint MLP ran; it also fails for all three.

This is a clean representation-level negative, not an optimization failure. Every model reaches 1.000 context-fold accuracy, no head emits a convergence warning, and the held-out Direct Assertion family is nearly or exactly perfectly rank-reversed. The final-token hypothesis representations contain a stable distinction, but its orientation remains tied to the language families present during training.

The locked decision is `frozen_final_state_joint_readout_insufficient_extract_token_span_interactions`. The next permitted experiment is one token/span-local extraction at the smallest backbone that already passed V11's span gates (4B). LoRA and final-mechanic access remain closed.

## Gate results

V12 used 7,380 gold-matched, gold-current determinant examples, exactly balanced between active and inactive. It reused all 24 locked V10 folds and did no new model inference before the result below.

| Head | Backbone | Context accuracy | Worst fold accuracy | Worst surface accuracy | Convergence warnings |
| --- | --- | ---: | ---: | ---: | ---: |
| Signed-difference linear | 0.8B | 1.000 | 0.053 | 0.000 | 0 |
| Signed-difference linear | 4B | 1.000 | 0.000 | 0.000 | 0 |
| Signed-difference linear | 9B | 1.000 | 0.000 | 0.000 | 0 |
| Joint 32-unit MLP | 0.8B | 1.000 | 0.000 | 0.000 | 0 |
| Joint 32-unit MLP | 4B | 1.000 | 0.000 | 0.000 | 0 |
| Joint 32-unit MLP | 9B | 1.000 | 0.000 | 0.000 | 0 |

The required minima were 0.70 for every fold and 0.65 for every non-empty fold-by-surface cell. No head/backbone combination passes.

## Held-out construction families

| Held-out template | Linear 0.8B | Linear 4B | Linear 9B | MLP 0.8B | MLP 4B | MLP 9B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contrastive Correction | 0.576 | 0.193 | 0.231 | 0.535 | 0.363 | 0.453 |
| Denied Claim | 0.836 | 0.977 | 0.962 | 1.000 | 1.000 | 0.985 |
| Direct Assertion | 0.053 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Explicit Negation | 0.690 | 0.611 | 0.781 | 0.193 | 0.015 | 0.000 |
| Rejected Claim | 0.760 | 0.962 | 0.918 | 0.830 | 0.991 | 0.985 |
| Scoped Rejection | 0.825 | 0.971 | 0.921 | 0.944 | 1.000 | 1.000 |

The result is highly structured rather than uniformly noisy. Denied Claim, Rejected Claim, and Scoped Rejection often transfer well, while Direct Assertion is systematically inverted and Contrastive Correction remains weak. Increasing scale does not remove that split.

## Why the zeroes are informative

| Head | Backbone | Direct Assertion accuracy | ROC AUC | Swap-complement accuracy |
| --- | --- | ---: | ---: | ---: |
| Signed-difference linear | 0.8B | 0.053 | 0.004 | 0.918 |
| Signed-difference linear | 4B | 0.000 | 0.000 | 0.985 |
| Signed-difference linear | 9B | 0.000 | 0.000 | 1.000 |
| Joint 32-unit MLP | 0.8B | 0.000 | 0.000 | 0.842 |
| Joint 32-unit MLP | 4B | 0.000 | 0.000 | 0.947 |
| Joint 32-unit MLP | 9B | 0.000 | 0.000 | 1.000 |

An ROC AUC of 0.000 means the held-out active/inactive ordering is perfectly reversed, not absent. For 9B, both heads also have 1.000 swap-complement accuracy on Direct Assertion: swapping the two hypothesis vectors flips the prediction exactly, but the learned orientation is wrong for every example. A larger or nonlinear final-token head therefore cannot by itself supply the missing construction-independent semantics.

The MLP result also narrows the next step. Its input contained both the hypothesis-pair mean and signed difference, so it could condition the relative decision on common evidence/context information. Its failure rules out a simple missing mean-by-difference interaction at the final token.

## Next experiment

The next representation should stay frozen and use 4B only. It should preserve the locked layer and prompts, but extract the last contextualized token inside each hypothesis rather than the generic assistant-generation token. In this causal prompt order, that token has attended to the evidence and the complete hypothesis and is the narrowest token-local relation representation available without changing the language task. A signed pair comparison should remain primary; one fixed nonlinear head may be conditional.

Temporal operator transfer remains a separate unresolved component. Even a successful token-local polarity readout must be combined with a repaired temporal head and re-run through the full locked symbolic pipeline before any final-mechanic access.

## Reproducibility and firewall

- V12 protocol lock: `96a9568851aafd31e7c316c086158ee516784cddf4064c50637e7736eaadf230`;
- V12 result: `1bba4ee7f25af3fd12248af2c1418f309f947da44d59bc5ea2329fe27c526d0f`;
- fitted head artifacts: 144 (two heads × three models × 24 folds);
- new feature extractions: 0;
- adapter runs and final-mechanic evaluations: zero.

All protected-access counters remain zero. V12 authorizes neither LoRA nor final evaluation.
