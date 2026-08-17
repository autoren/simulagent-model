# V68r2 development point-control repair preregistration

V68 and V68r1 each stopped before persisting a record result. The first exposed a posterior-sampling policy that was partial on the full mixture's observation support; V68r1 repaired that control and then exposed the same mathematical defect in the MAP point-model policy. The failures therefore diagnose a common policy-domain problem, not an outcome-dependent performance issue.

V68r2 totalizes every point-model control used by the frozen development evaluator: MAP and persistent posterior sampling. At histories supported by the selected point model, selection, Bayesian updating, planning, and deterministic tie-breaking are unchanged. If and only if an observation has exactly zero predictive probability under that point model and positive probability under the exact mixture, the remaining actions repeat the first action in the model's already-frozen canonical cycle. The fallback does not inspect subsequent observations, smooth probabilities, resample or reselect a model, or use the exact posterior for action selection.

The full 59-record census, four development models, priors, dynamics construction, horizon, quadrature, exact Bayes-adaptive planner, other controls, normalization, and all 19 gates remain frozen. The V68r2 evaluator will use a new output directory and one durable attempt marker. It is not authorized to score any confirmatory model.

If every unchanged development gate passes, the only next authorization is to preregister a confirmatory replication. If any gate fails, the frozen family stops before holdout scoring.
