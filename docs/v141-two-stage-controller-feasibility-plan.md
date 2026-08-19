# V141 Correlated Two-Stage Controller Feasibility Plan

## Question

Can a bounded finalizer plus an explicit evidence-sufficiency gate satisfy V139's inherited decision gates
without assuming that stages driven by the same model make independent errors?

## Abstract controller

The audit assigns four marginal reliabilities:

- `v`: the bounded finalizer emits a structurally valid typed choice;
- `s`: the sufficiency gate detects genuine ambiguity;
- `t`: the sufficiency gate permits a genuinely decidable request;
- `p`: the typed proposer selects the correct member of the complete safe universe when permitted.

Invalid finalization and detected ambiguity both map to `A00`. A decidable answer is correct only when the
finalizer, sufficiency gate, and proposer are all correct. Errors may be arbitrarily dependent. The audit
therefore uses Fréchet lower bounds rather than multiplying reliabilities. Across the five fixtures in a
minimal-pair group, it uses another Fréchet bound rather than assuming independent records.

For sequential behavior, missed ambiguity may select the presented known candidate and any incorrect final
proposal is charged the worst frozen decision cost. Query/no-query and final-decision correctness may also
be adversarially dependent. This produces conservative upper bounds on cost and false-known action.

## Grid and gates

The exact 0.001 grid spans 0.95 through 1.00. The reference point sets all four marginals to 0.99. The audit
finds the minimum symmetric marginal and one-at-a-time marginal thresholds while holding the other three at
0.99. All inherited V139 gates that can be bounded from these marginals are noncompensatory.

Candidate attraction among residual errors is intentionally not identifiable from marginal reliability and
is not claimed. It remains a mandatory empirical gate in any future realization study.

## Boundary

Passing establishes abstract feasibility only. It authorizes a fresh interface and population design, not
a language/model run. Same-model stages remain correlated; V134 and V139 remain closed. APIs, training,
induction, authority, action, and execution are not authorized.
