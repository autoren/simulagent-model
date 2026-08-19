# V132 Local Complete-answer Realization Plan

## Question

Can the pinned local Qwen3.8-27B 4-bit model realize the single-answer evidence strength that V130 showed is
sufficient in the abstract, when the complete safe answer menu and one fallible preliminary candidate are
shown explicitly?

## Prospective condition

Run once on the frozen V131 census: all 66 truth-by-candidate cells, four fixtures per cell. Automatically
extract only the 240 selected current user turns from the pinned SGD archive and add the 24 frozen missing-
observation controls. Known-choice definitions come from the pinned train schemas. Novel choices reveal
only their visible domain and the meaning “valid but undeclared”; their member intent identifiers and every
record's hidden labels remain absent from the prompt.

Use the already pinned `mlx-community/Qwen3.8-27B-4bit` snapshot, temperature zero, thinking disabled, one
sample, no demonstrations, no retries, and a one-key exact JSON response. Invalid output maps to `A00` and
is still counted invalid.

## Noncompensatory gates

The point estimate for exact eleven-way answers must reach V130's hardest 97.25% boundary and its one-sided
95% Wilson lower bound must exceed 95%. Every truth choice must reach 95%; known exact answers must reach
97.25%; novel, unsupported, and missing controls must each reach 95%. Structured validity must reach 99%,
false-known answers must remain at most 10%, and neither candidate-attracted nor abstention-attracted errors
may exceed the 75% bias modeled by V130.

Separately, feed the observed answers into the frozen V130 Bayesian policies at 97.25% assumed reliability.
Regret, known, unsupported, and false-known gates must pass for every frozen prior and assumed error regime.
All hypotheses remain retained and every action remains counterfactual.

Passing would authorize only an independently sourced confirmation design. Failure closes this one-pass
local realization branch. Neither outcome authorizes a human-equivalence claim, repeated-sample independence,
capability induction, protected access, APIs, training, authority, or execution.
