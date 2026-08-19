# V188 binary clarification-channel frontier plan

## Purpose

V187 showed that a fully identifying binary codebook is not cost-effective at 0.10 per question from the complete 14-contract version space. V188 diagnoses that result without changing it.

## Frozen information controls

Reuse the V187 development prior and the unchanged V186 questions. Compute Shannon entropy and a deterministic optimal Huffman prefix code. Then solve the exact target-blind decision tree constrained to the 25 frozen partition-distinct questions, with no generic answer, unit question cost, and horizon 13. Every leaf must be a singleton and retain its target.

## Frozen cost frontier

Keep generic trusted clarification at 0.40 and the V187 horizon at four questions. Evaluate exact adaptive and best fixed open-loop policies at all 81 rational costs `i/400`, for `i=0..80`. Singleton early stopping and all terminal authority rules remain unchanged. Record complete target-path signatures and every grid transition where either policy changes.

The frontier must reproduce the V187 point at 0.10 exactly. Any lower-cost positive region is diagnostic only; it cannot be used to relabel V187 as positive or to select a replacement cost.

## Successor decision

A separate multiway feasibility protocol may be designed only if V187 remains at the generic boundary, at least one lower binary cost has positive value, the target-informed oracle gap is positive, and the highest positive-value binary cost is below 0.10. Passing authorizes design only, not a multiway run.

No language, protected data, model/API, training, registration, authority, action, or execution is permitted.
