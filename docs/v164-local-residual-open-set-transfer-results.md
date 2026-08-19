# V164 local residual open-set transfer results

## Outcome

V164 completed exactly as preregistered: one pinned Qwen3.8-27B 4-bit model load and 76 independent
direct/no-thinking generations on the frozen V163 residual, with no retries. The run took 928.81 seconds
(15.48 minutes) and peaked at 17.33 GB active memory.

The model condition is nonqualifying. The deterministic-plus-model hybrid passed ordinary exact-accuracy and
false-known gates, but it failed the central novelty, calibration, and cost-sensitive gates. Protected transfer
therefore remains sealed.

## Results

The prompt revision successfully removed the earlier output-contract confound: all 76 responses were valid
typed JSON. Residual performance was:

| Metric | Result |
|---|---:|
| Structured validity | 100% (76/76) |
| Exact decision accuracy | 63.16% |
| Status macro F1 | 56.88% |
| Known exact-intent accuracy | 86.49% |
| Novel exact-scenario accuracy | 0% |
| Unsupported recall | 100% |
| Unsupported precision | 61.54% |
| False-known acceptance | 2.56% |
| Confidence ECE | 0.2458 |
| Top-confidence 80% error | 22.95% |
| Mean decision regret | 1.2303 |

The class pattern is decisive. The model was exact on all 13 familiar-known and all 16 unsupported residual
requests. It was exact on 19 of 24 unfamiliar-known requests. It made zero NOVEL predictions and therefore
resolved none of the 23 valid undeclared capabilities. Its 26 UNSUPPORTED predictions contained all 16 truly
unsupported cases but also ten other requests, producing only 61.54% precision.

The combined hybrid kept V163's 20 deterministic nonresidual decisions and used the model only on the 76-case
residual. This raised exact decision accuracy from V163 consensus's 19.79% to 69.79%, with 89.58% exact known-
intent accuracy, 100% unsupported recall, and 2.08% false-known acceptance. However, it still had zero novelty
recall. Combined mean regret increased from `0.953125` to `1.015625`, rather than improving by the required
0.10. Novel-valid cases alone had mean regret `2.8958` under the hybrid.

This is the important result:

> A deterministic residual gate plus a stronger local model can greatly improve closed-catalog accuracy while
> making the open-world decision policy worse, because the model collapses valid novelty into abstention or
> rejection and is overconfident about those mistakes.

The experiment therefore rejects an accuracy-first interpretation. The model is useful at known-intent
grounding and wholly out-of-scope rejection, but it does not solve the membership boundary between an unusual
known request, a valid undeclared capability, and an unsupported request.

## What changed relative to V107

V107's exact V105 prompt produced only 52.34% valid responses on observed language and zero exact known-intent
accuracy. V164's explicit conditional JSON contract produced 100% validity and 86.49% residual known-intent
accuracy. That establishes that the earlier formatting and typed-target failures were substantially interface-
sensitive.

The semantic open-world failure did not disappear. V107 made some NOVEL predictions but grounded known intents
poorly; V164 grounded known intents well but made no NOVEL predictions. A complete menu and valid output format
make the answer expressible, but they do not teach the model the catalog membership boundary.

## Access, safety, and decision

Development language was read automatically once. Protected language, manual utterance inspection, manual raw-
response inspection, API calls, training, real services, side effects, and execution were zero. The model loaded
once and generated exactly 76 responses. Missing observations abstained deterministically without a model call.
The complete 17-state safe universe remained intact, and the model never changed a nonresidual decision or
received capability, belief, action, tool, or execution authority.

Freeze V164 as a clean negative external-development result:

`local_residual_hybrid_is_nonqualifying_and_protected_transfer_remains_sealed`

Do not reopen it by changing the prompt, model, reasoning mode, quantization, parser, confidence, thresholds,
costs, gates, or residual. Do not run the protected transfer population under this protocol.

## Roadmap consequence

Track A should stop at this boundary for now. The next work should not be another classifier or decoding tweak
on the same language. The result strengthens the case for the two distinct successor tracks already documented:

1. a fixed-ontology, reversible sandbox decision study where ambiguity can be resolved by explicit information-
   gathering actions and model mistakes cannot create capabilities; and
2. a shadow-only ontology-acquisition study that separates whether a request is expressible by alias or
   composition of known primitives from whether the current evidence is sufficient, ambiguous, or contradictory.

The second track is the scientifically direct response to V164's failure: similarity-based classification is
being asked to answer two different questions—catalog expressibility and evidence sufficiency—with one status.
Those questions should be factored before any further language model is tested.

## Claim boundary

V164 is record-disjoint MASSIVE development evidence for a deterministic-first local shadow proposer. It is not
protected evidence, unrestricted open-world understanding, calibrated deployment safety, ontology learning,
planning evidence, action, or execution.
