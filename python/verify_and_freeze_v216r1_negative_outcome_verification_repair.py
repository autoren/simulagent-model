#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v216_bounded_external_artifact_population import read_jsonl
from v10_protocol import file_sha256
from v216_bounded_external_artifact_population import audit_population, score_population
from v216r1_negative_outcome_verification_repair import failed_scientific_checks, negative_outcome_matches
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v216r1-negative-outcome-verification-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v216r1-negative-outcome-verification-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v216r1-negative-outcome-verification-repair-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V216r1 outcome is already audited or frozen")
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V216r1 design lock or dependency hash mismatch")
    if not lock["authorization"]["verify_and_freeze_exact_existing_V216_negative_once"]:
        raise RuntimeError("V216r1 verification is not authorized")
    config = lock["config_payload"]
    v216_lock = json.loads((PROJECT_ROOT / lock["parent_V216_design_lock"]).read_text())
    v216_config = v216_lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in v216_config["artifacts"].items()}
    retrieval_manifest = json.loads(artifacts["retrievalManifest"].read_text())
    parser_control = json.loads(artifacts["parserControl"].read_text())
    development_public = read_jsonl(artifacts["developmentPublic"])
    development_truth = read_jsonl(artifacts["developmentTruth"])
    protected_public = read_jsonl(artifacts["protectedPublic"])
    protected_truth = read_jsonl(artifacts["protectedTruth"])
    split = json.loads(artifacts["split"].read_text())
    population_manifest = json.loads(artifacts["populationManifest"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    metrics = score_population(
        retrieval_manifest,
        parser_control,
        development_public + protected_public,
        development_truth + protected_truth,
        split,
        population_manifest,
        v216_config,
        PROJECT_ROOT,
    )
    rebuilt_scientific_audit = audit_population(metrics, summary["access"], v216_config)
    invariant = config["repairInvariant"]
    raw_checks = []
    for payload in v216_config["payloads"]:
        path = PROJECT_ROOT / payload["rawPath"]
        row = next(row for row in retrieval_manifest["attempts"] if row["payload_id"] == payload["payloadId"])
        raw_checks.append(path.is_file() and file_sha256(path) == row["sha256"] and path.stat().st_size == payload["expectedByteCount"])
    checks = {
        "repair_lock_and_every_V216_dependency_hash_are_exact": dependency_hashes_exact(lock),
        "raw_payloads_remain_exact": all(raw_checks),
        "metrics_and_scientific_audit_reconstruct_exactly": bool(
            metrics == summary["metrics"] and rebuilt_scientific_audit == summary["audit"]
        ),
        "negative_outcome_and_only_failed_check_are_exact": bool(
            negative_outcome_matches(summary, result, invariant)
            and failed_scientific_checks(rebuilt_scientific_audit) == sorted(invariant["expectedFailedScientificChecks"])
        ),
        "repair_does_not_authorize_V217_or_change_V216": bool(
            not rebuilt_scientific_audit["passed"]
            and rebuilt_scientific_audit["branch"] == invariant["expectedV216Branch"]
            and not lock["authorization"]["retrieve_rebuild_change_gate_or_authorize_V217"]
            and not result["authorization"]["design_V217_deterministic_external_reconstruction_controls"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "216r1-negative-outcome-verification-repair-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "V216_scientific_passed": False,
        "V216_branch": rebuilt_scientific_audit["branch"],
        "V216_decision": rebuilt_scientific_audit["decision"],
        "decision": config["decisionRule"]["ifRepairIntegrityPasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies: dict[str, Path] = {
        "repair_design_lock": lock_path,
        "repair_audit": audit_path,
    }
    for key in [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]:
        dependencies[f"frozen_{key}"] = PROJECT_ROOT / lock[key]
    outcome: dict[str, Any] = {
        "schema_version": "216r1-negative-outcome-verification-repair-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "repair_passed": True,
            "V216_scientific_passed": False,
            "V216_branch": rebuilt_scientific_audit["branch"],
            "V216_decision": rebuilt_scientific_audit["decision"],
            "failed_scientific_checks": failed_scientific_checks(rebuilt_scientific_audit),
            "metrics": metrics,
        },
        "authorization": {
            "select_post_V216_negative_non_V217_roadmap": True,
            "design_V217_deterministic_external_reconstruction_controls": False,
            "rerun_retrieve_rebuild_change_gate_open_protected_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()

