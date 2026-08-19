# V211/V211r1 deterministic residual-baseline result

## Verdict

V211r1 establishes deterministic closure of the V210 development residual. The 90 evaluation records were completely and safely resolved by both deterministic views and their abstention-first consensus. Model eligibility is exactly zero.

V211 itself produced a failed result because the compositional predictions were emitted under the obsolete key `CONTEXT_CONTRAST` while the scorer expected the preregistered `COMPOSITIONAL_RESPONSE_SPAN`. V211r1 changed only that key. All 360 prediction values, the split, learned lexicon, truth firewall, metrics, gates, and decision rule remained unchanged.

## Firewall and split

The 180-record V210 paraphrase/opaque residual was split by group ID hash into 45 calibration and 45 evaluation groups, with 90 records each and no group overlap. The split hashes reconstructed exactly.

An automatic firewall process wrote calibration surface/truth and evaluation surface/sealed-truth artifacts. Prediction ran in a separate process that received only calibration surface/truth and evaluation record ID, context ID, and utterance. It received no evaluation-truth path, `group_id`, hidden factor, source probability, history, action, stage, or protected path. Predictions were closed and hashed before sealed evaluation truth was opened.

## Results

`RAW_LEXICAL`, `COMPOSITIONAL_RESPONSE_SPAN`, and `ABSTENTION_FIRST_CONSENSUS` each achieved:

- 90 predictions;
- 90 accepted;
- coverage `1.0`;
- accepted accuracy `1.0`;
- zero false acceptances;
- zero counterfactual disagreement;
- zero residual records; and
- macro normalized V209 decision regret `6.38e-19`, numerical zero.

The abstain-always control had zero coverage and macro normalized decision regret `0.0300079094`. Thus the evaluation cases had positive decision value, but the fixed controlled lexicon made them fully learnable from the group-disjoint calibration half.

The learned raw and compositional mappings each contained nine label-pure tokens. Their complete agreement was not obtained from evaluation truth or protected data.

## Repair audit

The original artifact contained 90 rows under `CONTEXT_CONTRAST`. V211r1 contains 90 corresponding rows under `COMPOSITIONAL_RESPONSE_SPAN`. After normalizing that key, the artifacts match exactly; changed prediction-value count is zero. The repaired prediction SHA-256 is `6c1db797e715b8a77a5ad12b2eff516ef3639f545cbdc8b608712bc2b1fbcd59`.

## Interpretation and next direction

The V210 residual was not an appropriate LLM benchmark. Its difficulty came from withholding a fixed lexicon from the first projector, and ordinary deterministic calibration recovered that lexicon perfectly. Running Qwen here would add cost without answering an unresolved scientific question.

The next population must introduce **identifiable open-class variation**. It should separate:

- semantically interpretable held-out descriptions that can be inferred compositionally;
- arbitrary renamed symbols that are accompanied by in-episode reference evidence;
- genuinely insufficient or contradictory descriptions that require abstention; and
- surface variation whose target cannot be recovered from a small fixed token dictionary.

Identifiability must be proved at the utterance-and-context level before any model is run. Protected V210 artifacts remain sealed, and no local/API model, training, ontology mutation, service, side effect, action, or execution was used or authorized.
