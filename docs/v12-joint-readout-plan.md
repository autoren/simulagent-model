# V12 preregistration: frozen joint-hypothesis relation readout

## Question

V11 showed that frozen scale repairs determinant/evidence matching but does not make current-state polarity transferable through the independent three-way NLI head. V12 asks one narrower question: is the current value recoverable when the active and inactive hypotheses for the same gold evidence are compared jointly?

This is a readout diagnostic, not an end-to-end evaluation. It reuses the exact frozen V10/V11 features and the exact 24 V10 folds. It performs no model inference, layer search, adapter training, final-mechanic evaluation, or test-set access.

## Examples and labels

Only the gold-matched evidence pair for determinants with gold `CURRENT` temporal status is eligible. The input is the ordered pair of V10 NLI-final vectors `(h_active, h_inactive)`, and the binary target is the gold current value (`inactive=0`, `active=1`). Training and evaluation masks come directly from each locked V10 fold. No predicted span or temporal label enters this diagnostic.

## Locked heads

The primary head is a balanced logistic regression with `C=1`, `lbfgs`, and at most 3,000 iterations on the signed comparison `h_active - h_inactive`. This explicitly tests whether relative relation information is linearly accessible while enforcing the intended hypothesis ordering.

If and only if none of the three frozen backbones passes the primary gates, V12 runs one conditional nonlinear diagnostic. It is a single-hidden-layer MLP with 32 ReLU units over `concat((h_active + h_inactive)/2, h_active - h_inactive)`, Adam, alpha `0.001`, batch size `256`, learning rate `0.001`, 200 maximum iterations, tolerance `1e-4`, and 20 iterations without improvement. It uses no early stopping or hyperparameter selection. Both heads use seed zero plus the locked fold index.

The three backbones run in increasing-size order: Qwen3.5 0.8B at V10 layer 6/24, then the pinned V11 4B and 9B models at layer 8/32. All feature and metadata hashes are frozen before fitting.

## Metrics and gates

Each head reports accuracy, balanced accuracy, ROC AUC, and swap-complement accuracy overall and by state-lexicon surface for all 24 folds. The locked pass rule, inherited from the V10 oracle-polarity requirement, is:

- accuracy at least `0.70` in every fold; and
- accuracy at least `0.65` for every non-empty fold-by-surface cell.

No average can compensate for a failed construction family. Balanced accuracy and ROC AUC are diagnostic because every eligible fold is exactly class-balanced. Swap-complement accuracy is diagnostic and not a gate.

## Decisions

- If a primary head passes, pairwise polarity is linearly accessible. The next experiment must integrate the smallest passing frozen backbone's fixed comparator with a separately repaired temporal head before another full symbolic pipeline evaluation.
- Otherwise, if a conditional MLP passes, pairwise polarity is nonlinearly accessible. The next experiment must lock that smallest passing readout and separately repair temporal transfer.
- If neither head passes, final-token hypothesis pairs are insufficient. The next permitted representation change is a single locked token/span interaction extraction; LoRA remains unauthorized.

V12 cannot authorize a final mechanic or final-mechanic access. Its result cannot be used to tune either locked head.

## Firewall

Access remains zero for V3 test records, prior holdouts, V7 Tone Drift, V7 model results, the untouched V8 mechanic, and the final V9 mechanic. Adapter runs, new frozen extractions, alternate layers, and final-mechanic evaluations are all forbidden.
