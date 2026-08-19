# V210 controlled-language population and projection result

## Verdict

V210 is a positive population/projection-feasibility result. The locked generator produced exactly 540 controlled-language records split evenly between development and protected roles. Every integrity, population, projection, and access gate passed.

The result establishes that the V209r1 finite probabilistic language channel can be represented as a fresh, auditable language population with separate surface/truth artifacts and a truth-blind deterministic residual. It does not establish natural-language understanding or model performance.

## Population

Each role contains 270 records in 90 groups. Every group holds three matched realizations of one fixed latent case and semantic observation:

- `DIRECT`;
- `MATCHED_PARAPHRASE`; and
- `OPAQUE_RENAMING`.

Each role covers all three semantic regimes, both task states, five clarification contexts, and all three semantic observations. Every regime/state/context/observation cell has exactly three records. Full factor keys and record identifiers are unique.

Development and protected roles have zero record-ID overlap, group-ID overlap, normalized surface-string overlap, template-skeleton overlap, or lexical-label overlap. Surface artifacts contain no hidden regime, task state, semantic-observation field, source-probability field, or history field.

All matched groups have identical truth and source probability across their three realizations. Opaque mappings have zero mismatch. All 180 role/counterfactual probability groups normalize exactly. Truth/surface metadata round-trips at 100%, and a second in-memory generation matches byte for byte.

Frozen artifact hashes are:

- development surfaces: `c26c9234416200f7eb1864e2debe830a045a5676847b828b08ea50ae5b687bac`;
- development truth: `67244f5d21c3828e2a01bdd9ab442296492f23e4ec8592a165ec71231619284c`;
- protected surfaces: `552f4c316bddf963888704eb090b514d4ae8770bc77e06b3442847d1d5135015`; and
- protected truth: `4f256ea1387d10253207a29045840463a213fa3f4b450bb7b47ef7a28ff2f835`.

Protected records were read only by automatic integrity auditing. Their text was not displayed, manually inspected, or read by the baseline.

## Deterministic projection

The locked projector read only development record IDs and utterances. It accepted the 90 explicit direct-marker records and abstained on all 180 paraphrase/opaque records.

- prediction count: `270`;
- accepted count: `90`;
- residual count: `180`;
- coverage: `0.3333333333`;
- accepted accuracy: `1.0`; and
- false acceptances: `0`.

Prediction read no truth. Residual membership was derived from predictions alone. The residual contains both non-direct counterfactual types and still spans all three regimes, both states, all five contexts, and all three observations.

This projector is intentionally only a safe interface control. It does not test whether lexical, structural, or semantic rules can recover the residual.

## Boundary and next gate

No external language, model response, local/API model, training, ontology registration, trusted-state mutation, service, side effect, action, or execution was used. Protected surface baseline reads and protected manual reads were zero.

V210 authorizes only a separately preregistered deterministic development-baseline design. That experiment should compare fixed lexical normalization, template-independent compositional parsing, and abstention-first consensus on the 180-record residual, with truth withheld until predictions are frozen. It must define any later model-eligible residual from predictions alone. Protected artifacts and all model use remain closed.
