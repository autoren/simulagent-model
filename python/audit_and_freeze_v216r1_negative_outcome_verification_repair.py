#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from retrieve_v216_bounded_external_artifacts import dependency_hashes_exact
from v10_protocol import file_sha256
from v216r1_negative_outcome_verification_repair import negative_outcome_matches
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v216r1-negative-outcome-verification-repair.json",
        "plan": PROJECT_ROOT / "docs/v216r1-negative-outcome-verification-repair-plan.md",
        "protocol": PROJECT_ROOT / "python/v216r1_negative_outcome_verification_repair.py",
        "tests": PROJECT_ROOT / "python/test_v216r1_negative_outcome_verification_repair.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v216r1_negative_outcome_verification_repair.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v216r1_negative_outcome_verification_repair.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v216r1-negative-outcome-verification-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v216r1-negative-outcome-verification-repair-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v216r1-negative-outcome-verification-repair-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, outcome_path)):
        raise RuntimeError("V216r1 is already audited, locked, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    v216_lock_path = PROJECT_ROOT / config["parentV216DesignLock"]
    v216_lock = json.loads(v216_lock_path.read_text())
    v216_config = v216_lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in v216_config["artifacts"].items()}
    raw = {payload["payloadId"]: PROJECT_ROOT / payload["rawPath"] for payload in v216_config["payloads"]}
    summary_path = PROJECT_ROOT / config["parentV216Summary"]
    result_path = PROJECT_ROOT / config["parentV216Result"]
    results_document = PROJECT_ROOT / config["parentV216ResultsDocument"]
    original_verifier = PROJECT_ROOT / config["originalVerifier"]
    summary = json.loads(summary_path.read_text())
    result = json.loads(result_path.read_text())
    invariant = config["repairInvariant"]
    original_text = original_verifier.read_text()
    checks = {
        "V216_design_and_all_dependencies_remain_exact": dependency_hashes_exact(v216_lock),
        "existing_V216_negative_outcome_matches_frozen_repair_invariant": negative_outcome_matches(summary, result, invariant),
        "original_verifier_positive_only_defect_is_exactly_identified": bool(
            original_text.count('rebuilt_audit["passed"]') == invariant["expectedOriginalVerifierPositiveOnlyPredicateCount"]
            and 'and rebuilt_audit["passed"]' in original_text
            and 'if not passed:' in original_text
        ),
        "all_V216_raw_and_derived_artifacts_exist_before_repair": bool(
            all(path.is_file() for path in artifacts.values())
            and all(path.is_file() for path in raw.values())
            and summary_path.is_file()
            and result_path.is_file()
            and results_document.is_file()
        ),
        "repair_scope_is_verification_only_and_cannot_authorize_V217": bool(
            invariant["requireExactMetricAuditResultReconstruction"]
            and invariant["requireAllV216ArtifactsAndRawHashesUnchanged"]
            and invariant["authorizeNegativeOutcomeFreezeOnly"]
            and not config["decisionRule"]["repairCanAuthorizeV217"]
            and not config["decisionRule"]["repairCanAlterScientificOutcome"]
            and all(value == 0 for value in config["preLockExposure"].values())
        ),
        "repair_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in paths.values())
            and not audit_path.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "216r1-negative-outcome-verification-repair-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_negative_outcome_verification" if passed else "reject_V216r1_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies: dict[str, Path] = {
        **paths,
        "parent_V216_design_lock": v216_lock_path,
        "parent_V216_summary": summary_path,
        "parent_V216_result": result_path,
        "parent_V216_results_document": results_document,
        "original_V216_verifier": original_verifier,
        "design_audit": audit_path,
    }
    for key, path in artifacts.items():
        dependencies[f"V216_{key}"] = path
    for payload_id, path in raw.items():
        dependencies[f"V216_raw_{payload_id.lower()}"] = path
    lock: dict[str, Any] = {
        "schema_version": "216r1-negative-outcome-verification-repair-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "verify_and_freeze_exact_existing_V216_negative_once": True,
            "retrieve_rebuild_change_gate_or_authorize_V217": False,
            "open_protected_run_model_register_mutate_act_execute": False,
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

