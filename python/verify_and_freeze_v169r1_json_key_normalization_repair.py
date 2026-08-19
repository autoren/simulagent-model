#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v169_fresh_constraint_state_population import reconstruct
from v169r1_json_key_normalization_repair import json_normalize


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v169r1-json-key-normalization-repair-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v169r1-json-key-normalization-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v169r1-json-key-normalization-repair-outcome-lock.json"
    nominal = PROJECT_ROOT / "configs/v169-fresh-constraint-state-population-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists() or nominal.exists(): raise RuntimeError("V169r1 already frozen or nominal outcome exists")
    repair = json.loads(lock_path.read_text()); config = repair["config_payload"]
    parent = json.loads((PROJECT_ROOT / repair["parent_V169_population_lock"]).read_text())
    result = json.loads((PROJECT_ROOT / repair["V169_result"]).read_text())
    failed = json.loads((PROJECT_ROOT / repair["failed_V169_outcome_audit"]).read_text())
    rebuilt = reconstruct(parent)
    repair_deps = [key for key in repair if not key.endswith("_sha256") and f"{key}_sha256" in repair]
    checks = {
        "repair_lock_and_dependencies_exact": bool(valid_lock(repair) and all(file_sha256(PROJECT_ROOT / repair[key]) == repair[f"{key}_sha256"] for key in repair_deps)),
        "JSON_normalized_population_reconstructs_exactly": bool(json_normalize(rebuilt["population"]["summary"]) == result["summary"] and rebuilt["audit"]["checks"] == result["gates"]),
        "original_scientific_population_passed": bool(result["passed"] and rebuilt["audit"]["passed"] and all(result["gates"].values())),
        "failed_nominal_audit_preserved": bool(not failed["passed"] and not nominal.exists()),
        "zero_policy_model_authority_and_execution_access": all(value == 0 for value in config["accessGates"].values()),
    }
    passed = all(checks.values()); decision = config["decisionRule"]["ifRepairReconstructsExactly"] if passed else config["decisionRule"]["otherwise"]
    audit = {"schema_version": "169r1-json-key-normalization-repair-outcome-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "technical_serialization_repair_only": True, "decision": decision, "normalized_summary": json_normalize(rebuilt["population"]["summary"]), "repair_access": {key: 0 for key in config["accessGates"]}}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"repair_lock": lock_path, "parent_V169_population_lock": PROJECT_ROOT / repair["parent_V169_population_lock"], "V169_result": PROJECT_ROOT / repair["V169_result"], "failed_V169_outcome_audit": PROJECT_ROOT / repair["failed_V169_outcome_audit"], "verifier": PROJECT_ROOT / repair["verifier"], "audit": audit_path, "results_document": PROJECT_ROOT / repair["results_document"]}
    for key, integrity in result["output_integrity"].items(): deps[key] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "169r1-json-key-normalization-repair-outcome-lock", "experiment": config["experiment"],
        "outcome": {"passed": True, "scientific_population_gates_passed": True, "technical_serialization_repair_only": True, "decision": decision, "summary": result["summary"]},
        "authorization": {"modify_or_rebuild_V169": False, "create_nominal_V169_outcome": False, "preregister_unchanged_V167_planner_on_all_eligible_states": True, "score_planner_without_separate_lock": False, "tune_planner_or_run_model_register_act_or_execute": False},
    }
    for key, path in deps.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
