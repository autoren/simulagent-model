#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v196_protected_confirmation_role_binding import audit_binding, build_binding


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v196-protected-confirmation-role-binding-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V196 design lock")
    for key in [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V196 dependency drifted: {key}")
    auth = lock["authorization"]
    if not (
        auth["run_exact_text_free_binding_once"]
        and not auth["modify_source_selection_counts_or_freshness_rules"]
        and not auth["open_or_score_protected_language_or_run_model"]
        and not auth["run_API_training_registration_authority_action_or_execution"]
    ):
        raise RuntimeError("invalid V196 authorization")
    output_root = PROJECT_ROOT / "outputs/v196-protected-confirmation-role-binding/binding"
    if output_root.exists():
        raise RuntimeError("V196 binding output already exists")
    config = lock["config_payload"]
    binding = build_binding(
        json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text()),
        json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V183_hidden_identifiability"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V191_hidden_targets"]).read_text()),
        config,
    )
    audit = audit_binding(binding, config)
    decision = (
        config["decisionRule"]["ifEverySourceFreshnessPopulationAndAccessGatePasses"]
        if audit["passed"] else config["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "196-protected-confirmation-role-binding-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "summary": audit["summary"],
        "authorization": {
            "preregister_unchanged_V195_policy_confirmation_only": audit["passed"],
            "immediate_protected_language_read_or_model_run": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "remaining-source-census.json", binding["remaining_source_census"])
    write_json(output_root / "confirmation-identities.json", binding["public_identities"])
    write_json(output_root / "hidden-targets.json", binding["hidden_targets"])
    write_json(output_root / "binding-summary.json", binding["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
