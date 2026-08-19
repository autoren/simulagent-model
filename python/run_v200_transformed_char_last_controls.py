#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v200_transformed_char_last_controls import audit_evaluation, evaluate_transformed_char_last
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v200-transformed-char-last-controls-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock): raise RuntimeError("invalid V200 design lock")
    keys = (
        "config", "parent_V199_outcome", "source_V194_outcome", "source_V193_outcome", "source_V194_lock",
        "development_language", "hidden_targets", "visible_menu_variants", "hidden_variant_maps",
        "canonical_hidden_option_map", "canonical_CHAR_LAST_predictions", "primary_prior", "plan", "protocol",
        "tests", "V194_protocol", "runner", "verifier", "auditor", "design_audit",
    )
    for key in keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V200 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v200-transformed-char-last-controls/evaluation"
    if output_root.exists(): raise RuntimeError("V200 output already exists")
    evaluation = evaluate_transformed_char_last(
        *[json.loads((PROJECT_ROOT / lock[key]).read_text()) for key in (
            "development_language", "hidden_targets", "visible_menu_variants", "hidden_variant_maps",
            "canonical_hidden_option_map", "canonical_CHAR_LAST_predictions", "primary_prior",
        )],
        lock["config_payload"],
    )
    audit = audit_evaluation(evaluation, lock["config_payload"])
    config = lock["config_payload"]
    decision = config["decisionRule"][
        "ifEveryIntegrityInvarianceSignalAndAccessGatePasses" if audit["passed"] else "otherwise"
    ]
    result = {
        "schema_version": "200-transformed-char-last-controls-result", "experiment": config["experiment"],
        "passed": audit["passed"], "decision": decision, "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"], "summary": audit["summary"],
        "authorization": {
            "preregister_separate_unchanged_local_model_development_robustness_only": bool(audit["passed"]),
            "immediate_model_run_or_protected_access": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "scored-records.json", evaluation["scored_records"])
    write_json(output_root / "summary.json", evaluation["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
