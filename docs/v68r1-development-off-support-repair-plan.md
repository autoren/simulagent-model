# V68r1 posterior-sampling off-support repair

## Failure being repaired

The first locked V68 development attempt wrote its attempt marker and then stopped before persisting
any record result. A point model sampled from the posterior assigned exact zero probability to an
observation that remained reachable under the full exact mixture. Its finite policy tree therefore
had no branch on which the common-environment evaluator could continue. No confirmatory model was
loaded, and the failed evaluator will not be modified or rerun.

This is not a numerical issue and will not be hidden with epsilon smoothing. A point-model policy is
mathematically partial outside that model's support, while a control evaluated under posterior model
uncertainty must be total on the union support.

## Frozen totalization

V68r1 changes only the persistent posterior-sampling control. The same 17 systematic posterior atoms
and offset are retained. At a history supported by the sampled model, planning and belief updates
continue under that same fixed model. If the next observation has exactly zero sampled-model
predictive probability but positive full-mixture probability, the policy switches after that
observation to a deterministic fallback for all remaining actions: repeat the first action in the
environment's already frozen canonical cycle.

The fallback does not read later observations, does not resample a model, and does not use the exact
posterior to select an action. The policy is evaluated in the unchanged exact joint environment.
Off-support branch counts and expected entry probability are reported. This rule is deliberately
simple and fully specified; its performance is not a claim about alternative Thompson-sampling
repairs.

## Everything else remains locked

The four development models, 59-record census and order, family, action cycles, horizon, quadrature,
exact Bayes-adaptive planner, MAP/open-loop/myopic controls, normalization, gates, and zero-holdout
firewall are unchanged. Passing all original gates authorizes only preregistration of the
confirmatory design. Any failure stops before an untouched model is scored.
