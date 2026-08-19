#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v169_fresh_constraint_state_population import DEPENDENCY_KEYS, reconstruct
from v169r1_json_key_normalization_repair import json_normalize, only_class_coverage_key_type_mismatch


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v169r1-json-key-normalization-repair.json"
    plan_path = PROJECT_ROOT / "docs/v169r1-json-key-normalization-repair-plan.md"
    results_path = PROJECT_ROOT / "docs/v169-fresh-constraint-state-population-results.md"
    protocol_path = PROJECT_ROOT / "python/v169r1_json_key_normalization_repair.py"
    tests_path = PROJECT_ROOT / "python/test_v169r1_json_key_normalization_repair.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v169r1_json_key_normalization_repair.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v169r1_json_key_normalization_repair.py"
    audit_path = PROJECT_ROOT / "outputs/v169r1-json-key-normalization-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v169r1-json-key-normalization-repair-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v169r1-json-key-normalization-repair-outcome-lock.json"
    nominal = PROJECT_ROOT / "configs/v169-fresh-constraint-state-population-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or outcome_path.exists() or nominal.exists():
        raise RuntimeError("V169r1 already exists or nominal outcome is present")
    config = json.loads(config_path.read_text())
    parent = json.loads((PROJECT_ROOT / config["parentV169PopulationLock"]).read_text())
    result = json.loads((PROJECT_ROOT / config["v169Result"]).read_text())
    failed = json.loads((PROJECT_ROOT / config["failedV169OutcomeAudit"]).read_text())
    rebuilt = reconstruct(parent)
    false_checks = sorted(key for key, value in failed["checks"].items() if not value)
    summary_ok = only_class_coverage_key_type_mismatch(result["summary"], rebuilt["population"]["summary"])
    checks = {
        "parent_lock_and_dependencies_are_exact": bool(valid_lock(parent) and all(file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"] for key in DEPENDENCY_KEYS)),
        "failed_audit_has_only_expected_false_checks": bool(not failed["passed"] and false_checks == sorted(config["diagnosis"]["expectedFailedChecks"])),
        "sole_summary_mismatch_is_JSON_object_key_typing": summary_ok,
        "all_population_artifacts_metrics_gates_and_decision_normalize_exactly": bool(
            json_normalize(rebuilt["population"]["summary"]) == result["summary"]
            and rebuilt["audit"]["checks"] == result["gates"]
            and rebuilt["audit"]["passed"] == result["passed"]
            and result["passed"] and config["diagnosis"]["scientificPopulationPassed"]
        ),
        "repair_scope_and_zero_access_hold": bool(
            not config["repairBoundary"]["modifyOrRebuildV169"]
            and not config["repairBoundary"]["changePopulationMembershipEligibilityMetricsGatesOrDecision"]
            and not config["repairBoundary"]["createNominalV169Outcome"]
            and config["repairBoundary"]["freezeOnlyUnderV169r1"]
            and all(value == 0 for value in config["accessGates"].values())
        ),
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, results_path, protocol_path, tests_path, auditor_path, verifier_path)),
        "nominal_outcome_absent": not nominal.exists(),
    }
    passed = all(checks.values())
    audit = {"schema_version": "169r1-json-key-normalization-repair-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "false_original_checks": false_checks, "repair_access": {key: 0 for key in config["accessGates"]}}
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {
        "config": config_path, "parent_V169_population_lock": PROJECT_ROOT / config["parentV169PopulationLock"],
        "V169_result": PROJECT_ROOT / config["v169Result"], "failed_V169_outcome_audit": PROJECT_ROOT / config["failedV169OutcomeAudit"],
        "plan": plan_path, "results_document": results_path, "protocol": protocol_path, "tests": tests_path,
        "auditor": auditor_path, "verifier": verifier_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {"schema_version": "169r1-json-key-normalization-repair-lock", "experiment": config["experiment"], "config_payload": config}
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
