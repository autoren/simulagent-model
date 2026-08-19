from __future__ import annotations

import json
from pathlib import Path

from cross_track_evidence_audit import (
    CONFIG_PATH,
    ROOT,
    audit_repository,
    payload_hash,
    read_json,
    render_stopping_rule,
    render_synthesis,
    sha256_file,
    valid_lock,
    write_json,
)


OUTPUT_DIR = ROOT / "outputs/cross-track-evidence-audit-through-v224"
OUTCOME_LOCK = ROOT / "configs/cross-track-evidence-audit-through-v224-outcome-lock.json"


def verify_artifacts() -> dict:
    audit = audit_repository(ROOT, CONFIG_PATH)
    expected_json = {
        "experiment-ledger.json": audit["experiment_ledger"],
        "family-ledger.json": audit["family_ledger"],
        "reproducibility-audit.json": audit["reproducibility_audit"],
        "critical-chain.json": audit["critical_chain"],
        "claim-and-risk-matrix.json": audit["claim_and_risk_matrix"],
        "stopping-decision.json": audit["stopping_decision"],
    }
    for name, expected in expected_json.items():
        actual = read_json(OUTPUT_DIR / name)
        if actual != expected:
            raise AssertionError(f"Audit artifact does not reconstruct exactly: {name}")
    synthesis = ROOT / "docs/cross-track-evidence-synthesis-through-v224.md"
    stopping = ROOT / "docs/research-stopping-rule-after-v224.md"
    if synthesis.read_text(encoding="utf-8") != render_synthesis(audit):
        raise AssertionError("Synthesis document does not reconstruct exactly")
    if stopping.read_text(encoding="utf-8") != render_stopping_rule(audit):
        raise AssertionError("Stopping-rule document does not reconstruct exactly")

    manifest = read_json(OUTPUT_DIR / "manifest.json")
    for row in manifest["artifacts"]:
        path = ROOT / row["path"]
        if not path.exists() or sha256_file(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise AssertionError(f"Manifest mismatch: {row['path']}")

    reproducibility = audit["reproducibility_audit"]
    if reproducibility["payload_invalid"]:
        raise AssertionError("At least one outcome lock has an invalid payload hash")
    if audit["stopping_decision"]["authorized_next_experiment_count"] != 0:
        raise AssertionError("Unexpected experiment authorization")
    return audit


def main() -> None:
    audit = verify_artifacts()
    artifact_paths = [
        OUTPUT_DIR / "experiment-ledger.json",
        OUTPUT_DIR / "family-ledger.json",
        OUTPUT_DIR / "reproducibility-audit.json",
        OUTPUT_DIR / "critical-chain.json",
        OUTPUT_DIR / "claim-and-risk-matrix.json",
        OUTPUT_DIR / "stopping-decision.json",
        OUTPUT_DIR / "manifest.json",
        ROOT / "docs/cross-track-evidence-synthesis-through-v224.md",
        ROOT / "docs/research-stopping-rule-after-v224.md",
    ]
    lock = {
        "schema_version": "cross_track_evidence_audit_outcome_lock.v1",
        "experiment": "cross_track_evidence_audit_through_v224",
        "audit_config": "configs/cross-track-evidence-audit-through-v224.json",
        "audit_config_sha256": sha256_file(CONFIG_PATH),
        "artifacts": {path.resolve().relative_to(ROOT.resolve()).as_posix(): sha256_file(path) for path in artifact_paths},
        "implementation": "python/cross_track_evidence_audit.py",
        "implementation_sha256": sha256_file(ROOT / "python/cross_track_evidence_audit.py"),
        "runner": "python/run_cross_track_evidence_audit.py",
        "runner_sha256": sha256_file(ROOT / "python/run_cross_track_evidence_audit.py"),
        "tests": "python/test_cross_track_evidence_audit.py",
        "tests_sha256": sha256_file(ROOT / "python/test_cross_track_evidence_audit.py"),
        "verifier": "python/verify_and_freeze_cross_track_evidence_audit.py",
        "verifier_sha256": sha256_file(ROOT / "python/verify_and_freeze_cross_track_evidence_audit.py"),
        "outcome": {
            "outcome_lock_count": audit["reproducibility_audit"]["outcome_lock_count"],
            "payload_valid_count": audit["reproducibility_audit"]["payload_valid_count"],
            "non_sensitive_dependency_drift_count": audit["reproducibility_audit"]["drifted_non_sensitive_dependency_pair_count"],
            "expected_living_document_drift_count": audit["reproducibility_audit"]["expected_living_document_drift_count"],
            "missing_outcome_versions": audit["reproducibility_audit"]["versions_without_frozen_outcome"],
            "family_count": len(audit["family_ledger"]),
            "critical_chain_count": len(audit["critical_chain"]),
            "authorized_next_experiment_count": 0,
            "decision": "freeze_cross_track_synthesis_and_experimental_escalation_until_external_reopening_condition_changes",
            "protected_body_read_count": 0,
            "request_language_read_count": 0,
            "model_or_api_run_count": 0,
        },
        "authorization": {
            "new_experiment": False,
            "protected_or_request_language_access": False,
            "local_or_api_model_run": False,
            "training_or_tuning": False,
            "ontology_registration_or_trusted_state_mutation": False,
            "action_or_execution": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(OUTCOME_LOCK, lock)
    if not valid_lock(read_json(OUTCOME_LOCK)):
        raise AssertionError("Frozen audit outcome lock failed self-validation")
    print(json.dumps(lock["outcome"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
