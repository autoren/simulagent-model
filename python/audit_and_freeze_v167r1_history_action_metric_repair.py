#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v167_exact_evidence_gathering_planner import DEPENDENCY_KEYS, reconstruct
from v167r1_history_action_metric_repair import corrected_history_count, corrected_summary


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair.json"
    parent_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    plan_path = PROJECT_ROOT / "docs/v167r1-history-action-metric-repair-plan.md"
    results_doc_path = PROJECT_ROOT / "docs/v167-exact-evidence-gathering-planner-results.md"
    protocol_path = PROJECT_ROOT / "python/v167r1_history_action_metric_repair.py"
    tests_path = PROJECT_ROOT / "python/test_v167r1_history_action_metric_repair.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v167r1_history_action_metric_repair.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v167r1_history_action_metric_repair.py"
    audit_path = PROJECT_ROOT / "outputs/v167r1-history-action-metric-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair-outcome-lock.json"
    nominal_outcome = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or outcome_path.exists():
        raise RuntimeError("V167r1 is already preregistered or frozen")
    if nominal_outcome.exists():
        raise RuntimeError("nominal V167 outcome unexpectedly exists")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    result_path = PROJECT_ROOT / config["v167Result"]
    policy_path = PROJECT_ROOT / config["v167PolicyTrees"]
    result = json.loads(result_path.read_text())
    policy_artifact = json.loads(policy_path.read_text())
    reconstructed = reconstruct(parent)
    repaired = corrected_summary(reconstructed)
    diagnosis = config["diagnosis"]
    changed_fields = {key for key in repaired if repaired[key] != reconstructed["summary"][key]}
    parent_exact = valid_lock(parent) and all(
        file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"]
        for key in DEPENDENCY_KEYS
    )
    checks = {
        "parent_V167_lock_and_dependencies_are_exact": parent_exact,
        "original_result_and_policy_artifact_reconstruct_exactly": bool(
            result["summary"] == reconstructed["summary"]
            and policy_artifact == {"cases": reconstructed["cases"], "contains_language": False, "shadow_only": True}
            and file_sha256(policy_path) == result["output_integrity"]["case_policy_trees"]["sha256"]
        ),
        "diagnosed_overcount_and_action_only_projection_are_exact": bool(
            result["summary"][diagnosis["affectedMetric"]] == diagnosis["persistedValue"]
            and corrected_history_count(reconstructed["cases"]) == diagnosis["correctedValue"]
            and repaired[diagnosis["affectedMetric"]] == diagnosis["correctedValue"]
            and changed_fields == {diagnosis["affectedMetric"]}
        ),
        "corrected_metric_still_passes_frozen_gate": bool(
            diagnosis["correctedValue"] >= diagnosis["minimumFrozenGate"]
            and diagnosis["gateStillPasses"]
            and all(result["gates"].values())
        ),
        "repair_scope_preserves_original_run_science_and_authority_boundary": bool(
            not config["repairBoundary"]["modifyOriginalV167Artifacts"]
            and not config["repairBoundary"]["rerunFormalPlanner"]
            and not config["repairBoundary"]["changePriorQueriesCostsLossesHorizonPoliciesOrRisks"]
            and not config["repairBoundary"]["changeAnyOtherMetricGateDecisionOrClaimBoundary"]
            and not config["repairBoundary"]["createNominalV167Outcome"]
            and config["repairBoundary"]["repairOnlyActionProjectionAndFreezeUnderV167r1"]
            and all(value == 0 for value in config["accessGates"].values())
        ),
        "required_files_exist": all(path.is_file() for path in (
            config_path, parent_path, plan_path, results_doc_path, protocol_path,
            tests_path, auditor_path, verifier_path, result_path, policy_path,
        )),
        "nominal_V167_outcome_remains_absent": not nominal_outcome.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "167r1-history-action-metric-repair-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_verifier_only_metric_repair" if passed else "reject_V167r1_repair",
        "checks": checks,
        "diagnosis": {
            "persisted_history_count": result["summary"][diagnosis["affectedMetric"]],
            "corrected_history_count": corrected_history_count(reconstructed["cases"]),
            "changed_summary_fields": sorted(changed_fields),
        },
        "repair_access": {key: 0 for key in config["accessGates"]},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V167_planner_lock": parent_path,
        "V167_result": result_path,
        "V167_policy_trees": policy_path,
        "original_V167_verifier": PROJECT_ROOT / parent["verifier"],
        "plan": plan_path,
        "results_document": results_doc_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "167r1-history-action-metric-repair-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_or_rerun_original_V167": False,
            "repair_history_action_summary_projection_only": True,
            "freeze_corrected_outcome_under_V167r1": True,
            "create_nominal_V167_outcome": False,
            "run_model_register_mutate_trusted_state_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
