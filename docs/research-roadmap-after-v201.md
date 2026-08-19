# Research roadmap after V201/V201r2

## Evidence update

V198 confirmed that the fixed local model improves a finite clarification menu over `CHAR_LAST` on dialogue-isolated
protected language. V199/V200 then froze exact development-menu presentation shifts and their transformed deterministic
controls. V201 applied the unchanged confirmed model policy once to both shifts.

Task-level behavior was robust:

- `ORDER_ONLY` reproduced canonical primary top-3 recall `0.9291666667` and cost `0.2141666667` exactly;
- `ORDER_AND_OPAQUE_ID` reached recall `0.9347222222` and cost `0.2130555556`;
- the local model improved over same-variant `CHAR_LAST` by `0.0441666667` and `0.0513888889`;
- top-1 contract agreement with canonical output was `0.9166666667` in both variants;
- target-inclusion disagreement was `0.0` and `0.0119047619`; and
- structural validity, target retention, and trusted completion were `1.0`, with zero final truncation.

But complete top-3 semantic sets were not invariant. Mean contract-set Jaccard was `0.6666666667` for order-only and
`0.6369047619` for order plus opaque IDs, below the preregistered `0.80` gate in both cases. V201 is therefore a formal
negative robustness result. V201r1/V201r2 repaired only two verifier presentation fields—an elapsed-time snapshot and a
decision-label overwrite—without changing any fixture, score, metric, gate, or decision.

## Revised claim boundary

The evidence now supports:

> The local model is a useful, non-authoritative proposal heuristic for trusted clarification under these finite-menu
> presentation shifts, but its lower-ranked alternatives are not a stable semantic belief set.

Do not interpret rank position, top-3 membership, or agreement across one prompt as a calibrated posterior. The
trusted controller may use proposals to choose a question while retaining the full authoritative hypothesis universe.
Any richer POMDP must obtain uncertainty weights from an independently validated observation model, not directly
from these ranks.

## Active Track B1b: model-free decision sufficiency

Before another generation, freeze and compare controller summaries using only the already persisted normalized V195
and V201 proposals:

1. `TOP1_PLUS_OTHER` for each single presentation.
2. `TOP3_PLUS_OTHER` for each single presentation.
3. A three-presentation top-1 plurality with a deterministic canonical tie-break.
4. A three-presentation inclusion consensus containing contracts present in at least two top-3 sets, with `OTHER`
   and a preregistered maximum menu size.
5. Always use the fixed hierarchy for invalid or insufficient output.

Evaluate expected human-question cost, worst-presentation cost, per-record cost range, target-hit disagreement,
contract-set stability, target retention, exact completion, and incremental value over same-presentation `CHAR_LAST`.
Report three-presentation policies with their threefold model-call requirement even though V202 itself performs no
generation. Do not select a controller solely from mean accuracy; exact safety and worst-presentation decision loss
are primary.

V202 is development evidence. Any controller selected from it requires a new, separately frozen external or fresh
confirmation population. V201 does not authorize reuse of V198 protected language for a paired pass.

## Subsequent Track B

Menu expansion and distractors are paused. They would magnify unstable lower-ranked membership before the controller
has a justified decision-sufficient summary. Resume them only if B1b identifies a safe robust summary and a future
fresh confirmation supports it.

Natural vocabulary/paraphrase/ambiguity shifts remain later. Synthetic transformations must remain labeled stress
tests; without independent human validation they cannot establish semantic equivalence or natural ambiguity.

## Later tracks

- **Shadow ontology acquisition:** still deferred; model ranks cannot register, merge, prune, or delete concepts.
- **Human factors:** scripted or model-generated clarification answers remain simulations only.
- **Richer POMDP:** proceed after controller sufficiency, using externally validated likelihoods or explicit sets for
  semantic uncertainty and delayed state-dependent consequences.
- **Additional model/API:** not justified by V201; the scientific bottleneck is representation and controller design,
  not another capacity point.

## Durable rules

- Preserve V201 as a negative result; do not relax the 0.80 Jaccard gate or rerun transformed prompts.
- Keep the complete authoritative target universe and trusted `OTHER` path.
- Treat model output as evidence for question selection only, never terminal authority or a calibrated posterior.
- Freeze controller policies, costs, worst-case metrics, and stop rules before evaluating normalized proposals.
- No protected access, API, training, ontology mutation, service calls, side effects, action, or execution follows.

