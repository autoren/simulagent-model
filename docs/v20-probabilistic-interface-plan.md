# V20 preregistration: calibrated grounding–program uncertainty

## Objective

V20 tests one explanation of the V19 novel-ontology failure: the frozen representation contains
useful polarity information, but collapsing its score to a hard label causes exact schema search to
discard the true behavior. V20 is development-only. It uses the saved V19 features and unchanged
deployment heads and performs no model inference, fitting, adapter training, or final-suite access.

The hard V19 pipeline remains the confirmatory final system. V20 is a separately preregistered
challenger; it cannot retroactively alter the V19 result.

## Frozen and variable components

The Qwen revision, layer-8 features, evidence matcher, temporal head, polarity head, Boolean DSL,
behavioral hypothesis enumeration, visible transition codes, and query executor are frozen. Only
the interface between the polarity score and program induction changes.

The evidence span and temporal class remain hard because V19 localized the novel-ontology errors to
current polarity: span and temporal accuracy were perfect. `UNKNOWN_CURRENT` remains semantic
uncertainty over both Boolean values and is never converted into a probability claim about the
world.

## Calibration rule

For a selected current-evidence prompt with frozen binary-logistic score `s`, V20 defines
`p(active) = sigmoid(s)` and `p(inactive) = 1 - p(active)`.

Calibration is performed separately for each registered view using only V19's `calibration` split.
Exact selected base prompts are deduplicated before calibration. For each unique prompt with gold
label `y`, the LAC nonconformity is `1 - p(y)`. At fixed `alpha = 0.10`, the finite-sample corrected
`higher` quantile is used. A label is retained when its nonconformity does not exceed that quantile.
If numerical edge cases return an empty set, the MAP label is retained. The threshold is not chosen
from development accuracy and no alternative alpha is searched.

The calibration set is generated and correlated rather than an i.i.d. natural-language sample.
Coverage is therefore a development diagnostic, not a distribution-free external guarantee.

## Posterior over executable behaviors

Each support trace induces a finite distribution over complete assignments by taking the product of
the retained per-determinant label probabilities and renormalizing within the conformal label sets.
For program `P` and trace `(G_t, o_t)`, the likelihood is

`sum_a G_t(a) * 1[P(a) = o_t]`.

Trace likelihoods are multiplied in log space under a uniform prior over V18's behaviorally unique
program hypotheses. Programs with zero likelihood are removed. The credible program set is the
smallest deterministically tie-broken set whose normalized posterior mass reaches 0.95.

For a query, V20 returns the union of transition codes over every credible program and every
assignment allowed by either semantic unresolvedness or the conformal polarity set. It does not
return only the most probable outcome. If no program has nonzero likelihood, it returns the complete
outcome vocabulary and `identifiable = false`, matching the V19 failure policy.

## Conditions

1. `hard_support_hard_query`: immutable V19 baseline.
2. `probabilistic_support_oracle_query`: support-interface effect.
3. `oracle_support_probabilistic_query`: query-interface effect.
4. `probabilistic_support_probabilistic_query`: complete challenger.

The supported view is a preservation test. The novel view is a diagnostic repair test.

## Metrics and anti-widening controls

V20 reports episode-macro transition-set exact match, complete episodes, target retention in the
nonzero posterior and 0.95 credible set, empty-posterior rate, target posterior probability,
credible-program count and mass, posterior entropy/effective size, exact answer sets, mean predicted
set size, mean target set size, mean excess outcomes, and missing-target-outcome rate.

An improvement is not accepted merely because the challenger returns more outcomes. The novel
diagnostic requires nonnegative episode and target-retention gains over the hard baseline, no worse
empty rate than V19's 0.225, and at most 0.25 excess outcomes per query. Supported performance must
remain exactly 1.0 across all 40 episodes with no empty posterior and complete credible target
retention.

## Decision rule

- If supported preservation and novel diagnostic improvement both pass, freeze this probabilistic
  interface as the V21 challenger.
- If supported preservation passes but novel improvement fails, retain it only as a negative
  development result; V21 may still report it but cannot call it a repair.
- If supported preservation fails, exclude it from V21 scoring and keep the hard interface primary.
- No V20 outcome authorizes LoRA.
