# V15 preregistration: operator-supported frozen full pipeline

## Objective

V14 establishes transferable current-state polarity across all supported surface, mechanic, lexicon, and transition-operator folds. V15 recomposes the frozen language-grounding pipeline with the exact symbolic evaluator:

1. 4B layer-8 evidence-span features select evidence for each determinant;
2. the same evidence-span features classify temporal status;
3. V14's fixed hypothesis-token-mean signed comparator assigns active/inactive polarity;
4. the locked symbolic evaluator derives allowed transitions and identifiability.

No model weight, prompt, layer, readout family, threshold, or hyperparameter is tuned in V15.

## Extraction and deduplication

The exact V14 records deterministically produce 5,022 unique base evidence prompts and 10,044 unique NLI prompts over all 94,500 determinant/evidence candidates. Base prompts extract an evidence-span mean; NLI prompts extract the hypothesis-token mean. The 1,512 already locked V14 polarity prompts are reused only after exact string, model, layer, and feature-hash verification; all remaining features are extracted once from the same pinned 4B checkpoint.

Every unique base prompt has one consistent match target and, when matched, one consistent temporal target. Heads fit each unique prompt once, then map predictions back to records for exact ledger and symbolic evaluation. This prevents replicated source records from acting as independent training weight.

The context fold remains a non-gating repeated-local-prompt control. The 26 clean transfer folds are gated. Non-current evidence language is intentionally shared across current-surface folds and therefore represents supported temporal language, while each held-out current surface retains two training realizations of its semantic operator.

## Fixed evaluation

V15 reuses the four V10 oracle ablations, state-lexicon cells, complete allowed ledgers, label-flip pairs, complete intervention groups, and deterministic symbolic metrics. The V10 thresholds are applied to every one of the 26 transfer folds and every non-empty transfer fold × lexicon cell. Context and the three zero-shot semantic-operator diagnostics are reported but non-gating.

If all transfer gates pass, V15 authorizes design of a separately locked final-mechanic evaluation with the same frozen architecture. It does not itself access that mechanic. If a gate fails, the result must identify span, temporal, polarity, or composition as the failing component before any new experiment.

LoRA, alternate representations, alternate layers, final-mechanic access, and protected data remain forbidden in V15.
