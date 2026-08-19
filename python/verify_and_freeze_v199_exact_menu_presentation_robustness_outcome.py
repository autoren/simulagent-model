#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v199_exact_menu_presentation_robustness import audit_transformation_family, build_transformation_family
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v199-exact-menu-presentation-robustness/census"
    audit_path = PROJECT_ROOT / "outputs/v199-exact-menu-presentation-robustness/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v199-exact-menu-presentation-robustness-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V199 outcome already verified or frozen")
    keys = (
        "config", "parent_V198_outcome", "source_V195_outcome", "source_V193_outcome", "source_V191_outcome",
        "development_identities", "hidden_targets", "canonical_visible_menu", "canonical_hidden_option_map",
        "roadmap", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )
    rebuilt = build_transformation_family(
        json.loads((PROJECT_ROOT / lock["development_identities"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text()),
        json.loads((PROJECT_ROOT / lock["canonical_visible_menu"]).read_text()),
        json.loads((PROJECT_ROOT / lock["canonical_hidden_option_map"]).read_text()),
        lock["config_payload"],
    )
    expected = {
        "visible-menu-variants.json": rebuilt["visible_variants"],
        "hidden-variant-maps.json": rebuilt["hidden_variant_maps"],
        "transformation-audit-rows.json": rebuilt["audit_rows"],
        "summary.json": rebuilt["summary"],
    }
    artifacts_exact = all(
        (output_root / name).is_file() and json.loads((output_root / name).read_text()) == value
        for name, value in expected.items()
    )
    audit = audit_transformation_family(rebuilt, lock["config_payload"])
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    expected_decision = (
        lock["config_payload"]["decisionRule"]["ifEveryExactFeasibilityAndAccessGatePasses"]
        if audit["passed"] else lock["config_payload"]["decisionRule"]["otherwise"]
    )
    result_exact = bool(
        result["passed"] == audit["passed"] and result["checks"] == audit["checks"]
        and result["summary"] == audit["summary"] and result["decision"] == expected_decision
        and result["future_paired_development_gates"] == lock["config_payload"]["futurePairedDevelopmentGates"]
    )
    summary = audit["summary"]
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "transformation_artifacts_reconstruct_exactly": artifacts_exact,
        "result_reconstructs_exactly": result_exact,
        "all_exact_feasibility_gates_pass": bool(audit["passed"] and result["passed"]),
        "results_document_exists": results_path.is_file(),
        "language_model_protected_authority_and_execution_access_remain_zero": bool(
            summary["utterance_or_dialogue_language_read_count"] == 0
            and summary["deterministic_language_score_count"] == 0
            and summary["model_load_count"] == 0 and summary["model_generation_count"] == 0
            and summary["protected_language_read_count"] == 0 and summary["API_call_count"] == 0
            and summary["ontology_registration_count"] == 0 and summary["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "199-exact-menu-presentation-robustness-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "feasibility_gates_passed": bool(audit["passed"]),
        "decision": "freeze_verified_V199_exact_transformation_family" if passed else "freeze_failed_V199_verification",
        "checks": checks,
        "summary": summary,
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "experiment_lock": lock_path, "audit": audit_path,
        "visible_menu_variants": output_root / "visible-menu-variants.json",
        "hidden_variant_maps": output_root / "hidden-variant-maps.json",
        "transformation_audit_rows": output_root / "transformation-audit-rows.json",
        "summary": output_root / "summary.json", "result": result_path,
        "results_document": results_path, "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "199-exact-menu-presentation-robustness-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "feasibility_gates_passed": True,
            "decision": "freeze_V199_exact_family_and_allow_separate_deterministic_development_evaluation_only",
            "summary": summary,
        },
        "authorization": {
            "preregister_separate_deterministic_development_evaluation_only": True,
            "immediate_language_scoring_or_model_run": False,
            "protected_API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
