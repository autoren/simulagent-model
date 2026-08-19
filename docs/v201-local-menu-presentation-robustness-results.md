# V201 local menu-presentation robustness results

## Verdict

V201 is a clean negative result under its preregistered definition of semantic-ranking invariance.

The unchanged model preserved task accuracy, controller cost, and its incremental advantage over `CHAR_LAST` under
both exact menu shifts. It did not preserve the complete three-contract proposal set closely enough: mean top-3 set
Jaccard was below the frozen `0.80` gate in both variants.

This distinction matters:

> The model is robust enough for the trusted top-3 clarification controller on this development population, but its
> secondary alternatives are presentation-sensitive and should not be interpreted as a stable posterior or stable
> semantic belief set.

## Order-only variant

When only record-keyed menu order changed:

- primary top-3 recall remained exactly `0.9291666667`;
- macro top-3 recall remained exactly `0.9404761905`;
- primary top-3 cost remained exactly `0.2141666667`;
- improvement over transformed `CHAR_LAST` remained `0.0441666667`;
- top-1 contract agreement with canonical output was `0.9166666667`;
- target-inclusion disagreement was `0.0`; but
- mean top-3 contract-set Jaccard was `0.6666666667`, failing the `0.80` gate.

Thus, reordering the same semantic rows changed many second- or third-ranked alternatives even though it changed no
target inclusion or aggregate decision cost.

## Order plus opaque-ID variant

With record-keyed order and `Q01`–`Q14` reassignment:

- primary top-3 recall improved slightly to `0.9347222222`;
- macro top-3 recall improved to `0.9523809524`;
- primary top-3 cost decreased slightly to `0.2130555556`;
- improvement over transformed `CHAR_LAST` was `0.0513888889`;
- top-1 contract agreement was `0.9166666667`;
- target-inclusion disagreement was `0.0119047619`; but
- mean top-3 contract-set Jaccard was `0.6369047619`, again failing the `0.80` gate.

Opaque IDs therefore did not damage the trusted-controller outcome, but neither did they stabilize the lower-ranked
semantic alternatives.

## Safety and runtime

Structural validity was `1.0` in both variants. Final truncation, false terminal decisions, missing-control
generations, retries, raw-response persistence or inspection, protected access, APIs, training, ontology mutation,
services, side effects, action, and execution were zero. Target retention and trusted exact completion were `1.0`.

Every reasoning phase again used all 48 tokens and none naturally closed. Mean final length was `23.4107` tokens, and
the 64-token final cap was never hit. The run made exactly 168 reasoning plus 168 final generations with one model and
tokenizer load. Runtime was about 2,189 seconds and peak active MLX memory was 16,459,443,306 bytes.

## Freeze and direction

Freeze:

`freeze_V201_negative_or_presentation_sensitive_without_retry_reprompt_model_selection_or_API`

Do not open paired protected robustness, relax the Jaccard gate, retune the prompt, or rerun. The result is already
strong enough to refine the architecture: use model output as a lossy action-ranking heuristic, not as a calibrated
multi-hypothesis belief state. Before menu expansion or a richer POMDP consumes ranked alternatives, the next branch
should be model-free and ask which controller summaries are decision-sufficient under presentation sensitivity—for
example, target-inclusion stability and expected decision loss—without granting semantic authority to unstable
secondary ranks.

