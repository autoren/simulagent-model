#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v194_deterministic_language_menu_rankers import audit_evaluation, evaluate_rankers
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v194-deterministic-language-menu-rankers-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V194 design lock")
    for key in (
        "config",
        "parent_V193_outcome",
        "source_V192_outcome",
        "source_V192_extraction_lock",
        "development_language",
        "hidden_targets",
        "visible_menu",
        "hidden_option_map",
        "primary_prior",
        "fixed_hierarchy_target_costs",
        "protocol",
        "runner",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V194 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v194-deterministic-language-menu-rankers/evaluation"
    if output_root.exists():
        raise RuntimeError("V194 evaluation output already exists")
    evaluation = evaluate_rankers(
        json.loads((PROJECT_ROOT / lock["development_language"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text()),
        json.loads((PROJECT_ROOT / lock["visible_menu"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_option_map"]).read_text()),
        json.loads((PROJECT_ROOT / lock["primary_prior"]).read_text()),
        json.loads((PROJECT_ROOT / lock["fixed_hierarchy_target_costs"]).read_text()),
        lock["config_payload"],
    )
    audit = audit_evaluation(evaluation, lock["config_payload"])
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEveryIntegritySafetyAndMinimumSignalGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "194-deterministic-language-menu-rankers-result",
        "experiment": lock["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "summary": audit["summary"],
        "authorization": {
            "preregister_one_bounded_local_model_shadow_comparator": bool(audit["passed"]),
            "immediate_model_run_or_API_fallback": False,
            "protected_access_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "ranker-results.json", evaluation["ranker_results"])
    write_json(output_root / "shadow-predictions.json", evaluation["predictions"])
    write_json(output_root / "evaluation-summary.json", evaluation["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
