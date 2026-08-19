# V211 deterministic residual baselines and decision-impact plan

## Question

Is V210's 180-record paraphrase/opaque development residual genuinely model-relevant, or can deterministic learning from a group-disjoint calibration half resolve it safely?

This distinction matters because V210's surface grammar has a fixed development lexicon. If a deterministic learner can recover that lexicon, adding an LLM would test avoidable pattern matching rather than open-world language understanding.

## Prospective split and firewall

The 90 counterfactual groups are ranked by the hash of split seed `21101` and group ID. The first 45 groups form calibration and the remaining 45 evaluation. Each contributes exactly its paraphrase and opaque records, yielding 90 calibration and 90 evaluation records. Split identities and four hashes were frozen without reading surface or truth records.

Calibration surface and truth may be used to learn deterministic token-label associations. Evaluation prediction may read only record ID, context ID, and utterance. It cannot read group ID, truth, hidden regime/state, source probability, history, action, or stage. Predictions for all four baselines are frozen before evaluation truth is joined. Protected artifacts remain closed even to scoring.

## Baselines

`RAW_LEXICAL` retains sufficiently frequent calibration tokens associated with exactly one semantic observation and predicts only when all retained tokens in an evaluation utterance agree.

`COMPOSITIONAL_RESPONSE_SPAN` learns context-span tokens as those observed with multiple labels inside a context,
removes that span, and maps only globally label-pure response tokens. It composes context removal with response
interpretation and never matches a complete sentence template.

`ABSTENTION_FIRST_CONSENSUS` accepts only an identical non-abstaining prediction from both views. `ABSTAIN_ALWAYS` is the safety/value control.

No baseline reads the V210 template or lexicon configuration directly. Learning comes only from the frozen calibration half.
An automatic firewall phase partitions development artifacts. Prediction then runs in a separate worker that receives
calibration surface/truth and evaluation surface paths only. It is not given evaluation truth, group IDs, or any
protected path. The prediction file is closed and hashed before the scoring phase may open sealed evaluation truth.

## Decision impact

Semantic exactness alone is insufficient. A consensus observation selects the corresponding continuation of the frozen V209r1 exact policy; abstention selects safe defer. If a projection is wrong, its selected continuation is evaluated under the posterior induced by the true observation and preceding context history. Records are weighted by the frozen regime/state priors and source likelihood, normalized within each evaluation context and counterfactual type, then macro-averaged.

All accepting baselines must have perfect accepted accuracy and zero false acceptance. Consensus counterfactual disagreement and normalized decision regret must be zero; abstain-always must have positive regret, proving that the evaluation contains useful decisions.

## Branch rule

If consensus resolves every evaluation record safely, model eligibility is exactly zero. The correct successor is a new identifiable open-class population, not an LLM run on this trivial residual.

If a residual remains, it is model-eligible only if it contains 18–72 records, covers every regime/state/context/observation, and retains positive normalized decision value. Anything else is frozen as ineligible. A positive nontrivial residual authorizes only a separate local-model design; it does not open protected artifacts or run a model.
