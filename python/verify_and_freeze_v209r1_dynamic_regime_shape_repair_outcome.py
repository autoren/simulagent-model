#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v209r1_dynamic_regime_shape_repair import audit_oracle, evaluate_oracle, repair_diagnostics
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v209r1-controlled-language-observation-pomdp-shape-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v209r1-controlled-language-observation-pomdp-shape-repair/evaluation"
    audit_path = PROJECT_ROOT / "outputs/v209r1-controlled-language-observation-pomdp-shape-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v209r1-controlled-language-observation-pomdp-shape-repair-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v209r1-controlled-language-observation-pomdp-shape-repair-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V209r1 outcome already frozen")

    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
    parent_lock = json.loads((PROJECT_ROOT / lock["parent_V209_design_lock"]).read_text())
    config = parent_lock["config_payload"]
    rebuilt_repair = repair_diagnostics(config)
    rebuilt = evaluate_oracle(config)
    rebuilt_audit = audit_oracle(rebuilt, config)
    summary_exact = json.loads((output_root / "summary.json").read_text()) == rebuilt
    repair_exact = json.loads((output_root / "repair-diagnostics.json").read_text()) == rebuilt_repair
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    scientific_pass = rebuilt_audit["scientific_gates_passed"]
    expected_decision = config["decisionRule"]["ifEveryOracleIntegrityScientificAndAccessGatePasses" if scientific_pass else "otherwise"]
    result_exact = bool(
        result["passed"] == rebuilt_audit["access_gates_passed"]
        and result["scientific_oracle_passed"] == scientific_pass
        and result["checks"] == rebuilt_audit["checks"]
        and result["access_checks"] == rebuilt_audit["access_checks"]
        and result["summary"] == rebuilt
        and result["repair_diagnostics"] == rebuilt_repair
        and result["decision"] == expected_decision
    )
    checks = {
        "repair_lock_and_dependencies_are_exact": dependencies_exact,
        "repair_diagnostics_reconstruct_exactly": repair_exact,
        "summary_reconstructs_exactly": summary_exact,
        "result_reconstructs_exactly": result_exact,
        "access_audit_passes": rebuilt_audit["access_gates_passed"],
        "results_document_exists": results_path.is_file(),
        "repair_changed_no_scientific_design_elements": all(
            rebuilt_repair[key] == 0
            for key in ("changed_scientific_parameter_count", "changed_gate_count", "changed_comparator_count", "changed_decision_rule_count")
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "209r1-controlled-language-observation-POMDP-shape-repair-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "scientific_oracle_passed": scientific_pass,
        "decision": "freeze_verified_V209r1_repaired_oracle" if passed else "freeze_failed_V209r1_verification",
        "checks": checks,
        "summary": rebuilt,
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "repair_lock": lock_path,
        "audit": audit_path,
        "repair_diagnostics": output_root / "repair-diagnostics.json",
        "summary": output_root / "summary.json",
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "209r1-controlled-language-observation-POMDP-shape-repair-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {"passed": True, "scientific_oracle_passed": scientific_pass, "decision": expected_decision, "summary": rebuilt},
        "authorization": {
            "preregister_fresh_controlled_language_population_design_only": scientific_pass,
            "open_language_population_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
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
