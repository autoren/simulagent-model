# V22r2a nondiscretionary execution amendment

The first locked V22r2 evaluation process aborted before producing any prediction, metric, or
evaluation artifact. The binary atom-matching head fitted in memory, after which the three-class
truth-status fit raised an API error: the installed scikit-learn version requires multiclass
`liblinear` to be wrapped explicitly in `OneVsRestClassifier`.

This amendment makes that execution semantics explicit. It changes no data, frozen feature,
representation, split, negative sample, regularization value, class weighting, base solver, random
seed, gate, integration condition, or decision rule. Calibration and evaluation remain unread. The
replacement process may refit the unchanged binary atom head because the aborted process wrote no
recoverable head artifact. That technical refit is counted and disclosed; it is not a model or
hyperparameter selection.

The failed attempt ledger and exception record are inputs to the amendment lock. Exactly one
replacement evaluation is permitted, with zero new model forwards and zero adapter runs.
