# V47 sampled transition-probability estimation

## Why this is next

V46 showed that the architecture can represent, identify, and execute exact stochastic mechanics when support supplies oracle distributions. V47 replaces those distributions with finite realized trials. This isolates statistical estimation while keeping the ontology, DSL, probability vocabulary, and intervention policy declared.

The claim remains deliberately narrow: V47 does not learn an ontology, ground language, choose experiments, or infer arbitrary continuous probabilities.

## Fresh population and sealed samples

The population contains 48 fresh programs, with 12 from each V46 family and zero program overlap with V46. The three probability values are exactly balanced. Each mechanic receives 12 fixed support interventions with nested budgets of 8, 32, and 128 realized trajectories per intervention. The nesting makes the learning curve paired rather than confounded by different samples.

Twenty-four structurally disjoint query interventions receive 64 held-out realized trajectories each. Support exposes only realized trajectories. Query outcomes and exact joint trajectory distributions remain scorer-only. Randomness, seeds, trial ordering, and branch-draw semantics are frozen before construction.

## Estimator

The primary estimator places a uniform prior over the frozen finite program registry, computes exact complete-trajectory likelihoods, and retains the full posterior. Query predictions are posterior mixtures of exact joint trajectory distributions. This makes uncertainty explicit when finite samples do not identify a single program.

The MAP plug-in ablates program uncertainty. Uniformized outcome mass ablates probability estimation, literal empirical lookup tests non-lifted memorization, and the true program is used only as an unattainable scoring reference.

## Evaluation

Predictions are evaluated with held-out joint-trajectory log loss and multiclass Brier score. Scorer-only oracle distributions provide total-variation and parameter-recovery diagnostics. Ten-bin multiclass reliability error measures calibration. All uncertainty intervals cluster by mechanic, the statistical unit.

The primary 128-trial gates require normalized forecasts, mean total variation at most 0.05, every-family mean at most 0.08, probability MAE at most 0.05, MAP schema recovery at least 0.90, mean target posterior mass at least 0.85, and calibration error at most 0.05. Mean TV must strictly improve at both nested budget increases. Posterior-predictive log loss must not be worse than the MAP plug-in and must beat uniformized mass by at least 0.03 nats.

## Decision

A full pass authorizes preregistration—not construction—of stochastic language composition. A calibration-only failure calls for revising the posterior interface. Broad failure across budgets calls for revisiting stochastic identifiability before adding language, active selection, or neural training.
