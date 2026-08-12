# V23 results: probabilistic relational support replay

Decision: `probabilistic_support_insufficient_revise_language_interface`.

V23 is an exposed-data development diagnostic, not a holdout or final result. It reused the
frozen V22r2 features and heads with zero model forwards, fits, or hyperparameter selections.

## Registered reference

At 64 graph branches and 95% credible program mass, target nonzero retention was 0.917,
credible-set retention was 0.917, and empty posteriors were 0.000.
However, transition-set exact match was only 0.051, with 2.083
excess outcomes per query and a median of 64.5 credible programs.

## Evaluation curve

| Graph branches | Program mass | Exact | Target nonzero | Target credible | Empty | Excess | Missing |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.50 | 0.244 | 0.250 | 0.167 | 0.333 | 1.455 | 0.199 |
| 1 | 0.80 | 0.192 | 0.250 | 0.250 | 0.333 | 1.577 | 0.090 |
| 1 | 0.95 | 0.192 | 0.250 | 0.250 | 0.333 | 1.577 | 0.090 |
| 1 | 1.00 | 0.192 | 0.250 | 0.250 | 0.333 | 1.577 | 0.090 |
| 4 | 0.50 | 0.282 | 0.583 | 0.250 | 0.083 | 1.019 | 0.192 |
| 4 | 0.80 | 0.231 | 0.583 | 0.417 | 0.083 | 1.224 | 0.051 |
| 4 | 0.95 | 0.199 | 0.583 | 0.583 | 0.083 | 1.359 | 0.051 |
| 4 | 1.00 | 0.192 | 0.583 | 0.583 | 0.083 | 1.372 | 0.051 |
| 16 | 0.50 | 0.205 | 0.917 | 0.583 | 0.000 | 1.173 | 0.058 |
| 16 | 0.80 | 0.083 | 0.917 | 0.750 | 0.000 | 1.692 | 0.006 |
| 16 | 0.95 | 0.058 | 0.917 | 0.750 | 0.000 | 1.929 | 0.000 |
| 16 | 1.00 | 0.051 | 0.917 | 0.917 | 0.000 | 2.090 | 0.000 |
| 64 | 0.50 | 0.109 | 0.917 | 0.583 | 0.000 | 1.590 | 0.038 |
| 64 | 0.80 | 0.077 | 0.917 | 0.750 | 0.000 | 1.910 | 0.006 |
| 64 | 0.95 | 0.051 | 0.917 | 0.917 | 0.000 | 2.083 | 0.000 |
| 64 | 1.00 | 0.051 | 0.917 | 0.917 | 0.000 | 2.147 | 0.000 |

## Interpretation

Uncertainty propagation repairs catastrophic pruning but not identification. As graph coverage
rises, many incorrect programs acquire nonzero likelihood; credible unions then return nearly
the full outcome vocabulary. No registered cell combines high target retention with precise
answers. This is an anti-widening failure, not a probabilistic repair.

The next development direction is a candidate-conditioned frozen cross-encoder: compare each
evidence clause directly with a small set of atom and truth hypotheses, with explicit ordered
relation arguments. V22r2's top-3 matcher can be used only as a recall-preserving proposal stage.
A fresh surface benchmark is required after that interface is fixed because all V22r2 splits
are exposed. LoRA, a final suite, grammar expansion, and a neural challenger remain unauthorized.

Post-result integrity audit: `pass`.
