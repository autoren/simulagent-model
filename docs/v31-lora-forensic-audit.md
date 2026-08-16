# V31 LoRA forensic audit

## Verdict

The registered LoRA branch was functionally equal to the frozen branch at initialization, saved exactly the allowed head and LoRA tensors, contained no non-finite parameter values, and collapsed on the fit population itself. The negative transfer is therefore consistent with destructive optimization under the registered objective, not merely held-out surface failure.

## Evidence

Initial maximum absolute logit deltas: `[0.0, 0.0, 0.0]`.
Final fit exact-fact accuracies: `[0.123, 0.125, 0.062]`.
Final calibration exact-fact accuracies: `[0.123, 0.119, 0.062]`.
Head update L2 norms: `[8.679, 8.711, 8.698]`.
Combined adapter update L2 norms: `[10.425, 10.084, 10.674]`.

All audit computations were read-only. No training update, checkpoint selection, hyperparameter selection, or V31 evaluation reuse for selection occurred.

## Limits

Full loss curves and per-field gradient histories were not retained by the locked trainer and cannot be reconstructed without retraining. The audit does not claim those unavailable observations.
