# Cross-track evidence synthesis through V224/V224r2

## Bottom line

The project has a coherent uncertainty-aware decision architecture at the mechanism level, but it still lacks one empirical input: an independently grounded, prospective record-level semantic observation channel. Exact planning, typed clarification, reversible sandboxing, certificates, robust evidence gathering, explicit outside-semantics hypotheses, and safe deferral compose. Local-model proposal studies, retrospective Mondo reconstruction, and workflow/source censuses remain useful but do not supply that missing likelihood or gold standard.

No next experiment is authorized. Under the frozen gate, the unresolved premise is nonduplicative, but no admissible independent source is already obtainable: V224 populated none of the four required semantic strata. Further benchmark/model escalation would repeat the premise rather than test it.

## Audit coverage

- Frozen outcome locks audited: **198**.
- Canonical payload hashes valid: **198 / 198**.
- Non-sensitive referenced dependency pairs verified: **1353**.
- Non-sensitive dependency drifts: **8**.
- Expected append-only direction-document drifts (reported separately): **4**.
- Missing non-sensitive dependencies: **0**.
- Sensitive/nonlocal pairs deliberately not opened: **48**.
- Versions without a frozen outcome lock: **V58, V76, V77, V93, V99, V222**.
- Protected bodies, request language, and model/API runs during the audit: **0 / 0 / 0**.

A missing outcome version is a ledger gap, not evidence of failure. A valid outcome lock with a changed dependency remains valid as a historical payload, but its present-worktree reconstruction is no longer exact.

## Reproducibility drift preserved

- `configs/v78-clarification-outcome-lock.json` -> `python/verify_and_freeze_v78_clarification_outcome.py`: expected `14da46bfadbee3a8aacacd44a9929125bbd04abf0288ed467aafbfe4a4143f89`, found `5f9ca5f89bbcfadbcd96856b8e80bada7979a68b0394fd2c5a12308251cf650f`.
- `configs/v79-terminal-utility-outcome-lock.json` -> `python/verify_and_freeze_v79_terminal_utility_outcome.py`: expected `ad12eceada068fbd43e2a4bdc63b52cb79402c3c397384270a28f454881ae3e2`, found `abd1ce35419006d118dee42098cda065c7218f56efff9d86a5f91d6e79d61162`.
- `configs/v94-global-open-set-source-outcome-lock.json` -> `docs/v94-global-open-set-source-results.md`: expected `6650a3e7e0b4549e336e75a1119b2f2b707f33632befe28ca6a0c627ae69bb90`, found `cf79b33ab92551e04503bf600e31398d90d8b10e6e9a1747b13ccd5c15bba889`.
- `configs/v106-open-world-development-benchmark-technical-outcome-lock.json` -> `python/audit_and_freeze_v106_development_benchmark.py`: expected `d750e24ae7036957e6149915b7f6bc1c157763f7ee796b6a8df7db7978456f7d`, found `c0f5227a51887a2d0bea37f6b5dae2c341b148863b99dca8da99bbb4039c1ce0`.
- `configs/v118-evidence-identifiability-audit-outcome-lock.json` -> `docs/v118-evidence-identifiability-audit-results.md`: expected `9a1c37ffdd05bf1031c2008b64da9e1054fc404eeaada8ffc85d4a00ee2ceec5`, found `ad00a90ef82a4c914d9fe3711457625b126962e77a733a41eccb928c456a3cb5`.
- `configs/v119-asymmetric-adaptive-evidence-outcome-lock.json` -> `docs/v119-asymmetric-adaptive-evidence-results.md`: expected `4727bc16c7b1ac309be195826cd5a464daa27d0d183943a686296830130f7523`, found `7baab03d1967531c2b8ab436a7e87c67bb3684e44c7bc460a22a738facc12f52`.
- `configs/v120-selective-query-value-audit-outcome-lock.json` -> `docs/v120-selective-query-value-audit-results.md`: expected `ef975aa03573c1461335a5e704d3b0d323255ce703bccc5e0facbea80a331984`, found `cb3ce30602f9be3f0232b80e06bc6ce448f06e32f5e49196e8e49ba9b96ec0ee`.
- `configs/v121-prequery-value-selectivity-envelope-outcome-lock.json` -> `docs/v121-prequery-value-selectivity-envelope-results.md`: expected `53cc34691cee7420a5cdcd88f09e38d6dd362c604228ff8e4429e7d4366a6c45`, found `2d20e5326d83f53b6593394cb11bca0a0245a9f1c664518c534ee7484450ad06`.

These files were not rewritten. Silently updating expected hashes would erase provenance; restoration requires recovering the originally hashed dependency or freezing an explicit addendum.

## Family-level disposition

| Family | Versions | Evidence status | Disposition |
|---|---:|---|---|
| F01_FOUNDATIONS | V33-V62 | mixed_positive_mechanism_chain | stop_repeating_foundational_reskins |
| F02_EXTERNAL_POMDP | V63-V75 | positive_with_sensor_semantics_boundary | closed_positive_mechanism |
| F03_LOCAL_LLM_INTERFACE | V76-V92 | negative_model_branch_with_positive_deterministic_interfaces | closed_for_unstructured_model_retries |
| F04_OPEN_SET_SOURCE | V93-V107 | mixed_source_success_model_failure | superseded_by_interface_forensics |
| F05_INTERFACE_FORENSICS | V108-V115 | diagnostic_mixed_negative_transfer | closed_after_nontransfer |
| F06_CLARIFICATION_CAUSALITY | V116-V122 | boundary_mechanisms_positive_realization_negative | mechanism_retained_realization_deferred |
| F07_SGD_CLARIFICATION | V123-V134 | mixed_with_capability_confound | superseded_by_identifiability_protocol |
| F08_THINKING_CERTIFICATES | V135-V151 | positive_controller_mechanics_negative_local_realization | close_reasoning_effort_and_local_recovery_cycles |
| F09_DETERMINISTIC_ROUTING | V152-V160 | mixed_positive_deterministic_mechanisms | only_transfer_not_reskin |
| F10_MASSIVE_TRANSFER_RESIDUAL | V161-V164 | positive_population_negative_novel_model_residual | closed_model_residual |
| F11_ONTOLOGY_SANDBOX | V165-V182 | strong_positive_mechanism_chain | closed_positive_mechanism |
| F12_TYPED_CHANNEL | V183-V190 | mixed_binary_boundary_and_menu_compression | retain_finite_channel_only |
| F13_LANGUAGE_TO_MENU | V191-V203 | narrow_positive_non_authoritative_model_result_with_robustness_boundary | development_claim_frozen_no_scaling |
| F14_OPEN_WORLD_POMDP | V204-V211 | positive_terminal_mechanism_external_validation_negative | mechanism_retained_external_grounding_deferred |
| F15_REPRESENTATIONAL_DIAGNOSIS | V212-V214 | positive_deterministic_closure | reuse_protocol_do_not_repeat_same_operator |
| F16_EXTERNAL_ONTOLOGY | V215-V221 | retrospective_positive_after_source_failures | retrospective_track_closed |
| F17_ARCHIVED_ADJUDICATION | V223-V224 | workflow_positive_record_level_negative | deferred_until_external_state_changes |

Full claims, exact member outcome/design locks, repairs, protected boundaries, model involvement, and unresolved premises are in `outputs/cross-track-evidence-audit-through-v224/family-ledger.json`.

## What composes into the architecture

1. **Represent hypotheses and equivalence safely.** V212-V214 provide exact expressibility/evidence states, equivalence collapse, contradiction handling, and a rule that model candidates remain provisional.
2. **Optionally propose, never authorize.** V195/V198 support a narrow finite-menu local proposal role; V201 and V202 require presentation-aware normalization and a trusted controller retaining the full hypothesis set.
3. **Maintain beliefs and an outside regime.** V50r1/V51r1 and V205/V209 supply history-dependent inference plus an explicit outside-semantics state. Model confidence is not substituted for a likelihood.
4. **Choose evidence for decision value.** V63-V75, V116-V122, and V183-V190 define when sensing/clarification can change later control, including binary/multiway channel limits and safe OTHER/defer outcomes.
5. **Sandbox provisional consequences.** V168/V171 retain reversible transaction semantics while V175/V177 and V180/V182 add certificate-aware and one-corruption-robust evidence policies.
6. **Act only after settlement; otherwise defer.** V205's terminally proper structure prevents horizon escape and makes calibration, inspection, state-specific repair, or safe deferral optimal on different histories.

This is conceptual composition across separately frozen studies. The ledger does not claim that every later lock directly references every earlier one, nor that the complete stack has been externally validated end to end.

## What does not compose as semantic evidence

- LLM rank, confidence, agreement, or longer reasoning traces cannot be used as posterior mass.
- Finite-menu proposal recall cannot be reported as unrestricted open-world recognition.
- A synthetic person/model answer cannot replace independent speaker or expert adjudication.
- Retrospective versioned ontology reconstruction cannot identify a new speaker's intended concept.
- Mondo's workflow-level human curation cannot replace the absent record-level four-way population.
- Transport, parser, metric, or verifier repairs cannot be counted as independent replications.

## Duplicate and post-hoc risk

The main multiplicity risk is the sequence of related local-model recovery/interface variants across V80-V92, V107-V115, and V132-V164. Those runs diagnose different interfaces but repeatedly encounter the same missing premise: independently observable open-world semantic boundaries. They should be synthesized as bounded diagnostics, not counted as many independent tests of model competence. The scientific replications that do count as fresh mechanism confirmation are the role-separated/procedural confirmations such as V70, V75, V170/V171, V177, and V182.

## External evidence boundary

V221 is a legitimate positive external retrospective result: immutable Mondo artifacts plus deterministic exact-family expansion solve catalog-version reconstruction. V223 is a legitimate positive workflow census. V224 then narrows both: the same source does not yield a usable prospective record-level known/new/insufficient/unsupported population under the frozen metadata gate. The combined result is not contradictory; it separates catalog reconstruction from speaker semantics.

## Canonical next state

Experimental escalation is frozen. Continue only documentation, reproducibility restoration, and architecture consolidation until an external-state change satisfies the reopening rule in `docs/research-stopping-rule-after-v224.md`.
