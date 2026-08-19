# V139 Repaired Direct-versus-Thinking Realization Plan

## Question

On a fresh synthetic population, does thinking-enabled inference with the V138-corrected parser improve
the local model's ability to abstain on genuinely underdetermined requests and use targeted clarification?

## Population and conditions

V139 uses all 100 fixtures in the V135 development split, which has not previously received model
generations. Its twenty groups contain clear-left, clear-right, ambiguous, clarified-left, and
clarified-right variants. No fixture may be removed, relabeled, or regenerated.

The same pinned Qwen3.8-27B 4-bit snapshot and semantic prompt are used once per fixture in each condition:

- direct: thinking disabled, 32 generated-token ceiling;
- thinking: thinking enabled, 1,024 generated-token ceiling and the V138 prompt-opened parser.

Temperature is zero, with one sample, no retry, one model load, and exactly 200 generations. Because the
reasoning regimes have different token budgets, the result compares complete inference regimes rather than
isolating a pure boolean thinking effect.

## Evaluation

The inherited noncompensatory gates cover structural validity, exact clear/ambiguous/clarified decisions,
complete five-stage groups, false-known and candidate-attracted errors, sequential query cost, and safe
non-known behavior. Invalid output maps to `A00` for safety but still fails structural validity.

Only answer IDs, validation categories, hashes, token counts, and timing may be retained. Raw generations
and reasoning traces may not be stored or inspected.

## Boundary

Passing is synthetic development evidence only and may justify a separately designed externally authored
transfer study. It does not open V134, establish human or open-world validity, authorize an API or training,
or grant the model authority, actions, or execution.
