# V17r2 results: one-shot final-mechanic evaluation

The one-shot final mechanic passes every preregistered gate.

This supports transfer to one unseen simulator action under supported state concepts, temporal language, semantic operators, and lexicons. It does not establish arbitrary-ontology transfer and does not authorize LoRA.

Decision: `final_mechanic_generalization_passes`.

## Locked gates

| Gate | Value | Minimum | Result |
|---|---:|---:|---|
| minimum_fold_span_accuracy | 1.000 | 0.650 | pass |
| minimum_surface_span_accuracy | 1.000 | 0.600 | pass |
| minimum_fold_temporal_accuracy | 1.000 | 0.700 | pass |
| minimum_surface_temporal_accuracy | 1.000 | 0.650 | pass |
| minimum_fold_oracle_polarity_accuracy | 1.000 | 0.700 | pass |
| minimum_surface_oracle_polarity_accuracy | 1.000 | 0.650 | pass |
| minimum_fold_nli_pair_consistency | 1.000 | 0.700 | pass |
| minimum_surface_nli_pair_consistency | 1.000 | 0.650 | pass |
| minimum_fold_allowed_values_accuracy | 1.000 | 0.650 | pass |
| minimum_surface_allowed_values_accuracy | 1.000 | 0.600 | pass |
| minimum_fold_symbolic_balanced_accuracy | 1.000 | 0.650 | pass |
| minimum_surface_symbolic_balanced_accuracy | 1.000 | 0.600 | pass |
| minimum_fold_complete_flip_pair_accuracy | 1.000 | 0.600 | pass |
| minimum_fold_complete_intervention_group_accuracy | 1.000 | 0.500 | pass |

## Overall final result

The final set contains 1,296 records in 216 complete intervention groups. It is exactly balanced: 648 ambiguous and 648 identifiable records.

- span accuracy: 1.000;
- predicted-span temporal accuracy: 1.000;
- oracle-span/oracle-temporal polarity accuracy: 1.000;
- fully predicted allowed-values accuracy: 1.000;
- fully predicted symbolic balanced accuracy: 1.000;
- complete label-flip-pair accuracy: 1.000; and
- complete six-record intervention-group accuracy: 1.000.

## Worst transfer cells

- worst template/lexicon span: `contrastive_correction/canonical` at 1.000;
- worst template/lexicon temporal: `contrastive_correction/canonical` at 1.000;
- worst template/lexicon polarity: `contrastive_correction/canonical` at 1.000;
- worst template/lexicon allowed values: `contrastive_correction/canonical` at 1.000;
- worst template/lexicon symbolic balanced accuracy: `contrastive_correction/canonical` at 1.000;

## Reproducibility and firewall

The initial V17 constructor aborted before writing data because its normalized transition identity collapsed a read-only action. V17r2 was freshly locked before any record existed and adds the visible returned action observation to that identity; no model result informed the correction.

- construction lock: `3803971e400ecfbe3cede32139ca0c0dbec0207fbba0b79437709d5a2c3d67b2`;
- sealed dataset: `52c5421032aee2eefb1a22f9719e73c287deb9b23e951ea2f831961f2afc974a`;
- evaluation lock: `28e399af297bea25e8343900213241b1f3cf5ba09e2987441e3ce7aa1de3dd33`;
- final feature artifact: `bdf14d6a6b3ade4f094c225bc2f5537991c73f97e8ad02f0aedb42594ccb20ce`;
- deployment heads: `78b43b62d1a7f45dfe5efabf83ccfcd00c8c370c49f90f275dd32747f5a74278`;
- result: `0fe4a1ea916dd1b53223b1d5dafacea0ed22f3240e8f41585023f8842128368a`;
- unique final base/NLI prompts and forward passes: 1,161 / 2,322 / 3,483;
- development-only linear fits / final evaluations / adapter runs: 3 / 1 / 0; and
- Tone Drift, V3 test records, prior holdouts, untouched V8 mechanics, V7 outputs, alternate models, alternate layers, alternate representations, threshold changes, and final retries: zero.

V17 is permanently exposed after this result. No subsequent score on these records is a final-holdout evaluation.
