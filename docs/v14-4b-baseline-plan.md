# V14 preregistration: operator-supported 4B token-mean baseline

## Authorization and question

The locked V14 corpus and pre-model audit pass every structural, shortcut, and semantic-operator support gate. This model experiment asks whether V13's strongest frozen representation transfers across unseen surface forms once each held-out surface retains two training realizations of the same logical operator.

## Locked examples and overlap correction

The polarity-only corpus contains 11,070 record-weighted current determinants but only 756 unique local NLI hypothesis pairs (1,512 unique prompts). Repeated pairs always have the same target.

The local-prompt audit found exact overlap in the old context fold: all 702 evaluation pairs also occur in context training. Context is therefore retained only as a clearly labeled repeated-prompt control. It is not a generalization gate.

All other 26 primary transfer folds—six mechanics, nine surface families, three lexicons, two transition operators, and six operator×lexicon combinations—have zero exact pair overlap. V14 fits and scores each unique pair once so repeated source records cannot inflate metrics. The three zero-shot semantic-operator diagnostics also have zero overlap and remain non-gating.

## Locked model and readout

V14 uses only `mlx-community/Qwen3.5-4B-4bit` revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, frozen layer 8/32. It renders the exact V10 NLI interface and extracts the float32 mean over tokenizer tokens overlapping each terminal current-state hypothesis.

The sole classifier is the V13 hypothesis-mean signed comparison: `mean(h_active) - mean(h_inactive)`, followed by balanced logistic regression with `C=1`, `lbfgs`, 3,000 maximum iterations, and seed zero plus fold index. There is no alternate layer, token, pooling rule, nonlinear head, or hyperparameter search.

## Gates and decisions

The primary baseline passes only if accuracy is at least 0.70 in every one of the 26 clean transfer folds and at least 0.65 in every non-empty transfer-fold × state-lexicon cell. No average can compensate for a failed surface.

The three semantic-operator holdouts and repeated-prompt context control are reported but cannot fail or tune the primary protocol.

- If all supported transfer gates pass, freeze this polarity comparator and design a separate temporal-operator repair before another full symbolic pipeline evaluation.
- If any supported transfer gate fails, stop before LoRA and determine whether the remaining failure is a corpus support defect or a genuine frozen-representation limitation.

No outcome opens the final mechanic. V14 baseline permits one feature extraction, 30 linear fits (27 primary including context plus three diagnostics), zero adapters, and zero protected access.
