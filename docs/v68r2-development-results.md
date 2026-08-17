# V68r2 development-only result

V68r2 completed the sealed 59-record census over four development models. All records were retained, all metrics were finite, beliefs normalized, source models validated, the 65-node planner agreed with the 129-node convergence planner, and no confirmatory model was scored.

The preregistered decision is nevertheless negative. Exact Bayes-adaptive planning disagreed with the MAP control at the root on 0 records. MAP and persistent posterior sampling each had 0 records above the 0.005 material-regret threshold. Maximum normalized MAP regret was 0.0015135820, below the frozen 0.01 gate. Four noncompensatory gates therefore failed. Open-loop, myopic-reward, and information-only controls did show gaps, so the evaluator can detect planning deficiencies; the specific command-channel uncertainty family does not create enough control-relevant model ambiguity for the intended Bayes-adaptive comparison.

The exact-zero off-support rules were exercised but remained narrow: MAP entered fallback branches on 2 records and posterior sampling on 4. There was no epsilon smoothing and no model reselection or resampling.

Per the frozen hierarchy, the unchanged family stops here. Confirmatory models remain unscored. The next defensible stage is a new development-only preregistration with a materially more control-relevant unknown-dynamics family, not threshold relaxation or holdout evaluation.
