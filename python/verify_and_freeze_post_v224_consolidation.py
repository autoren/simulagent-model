from __future__ import annotations

import json

from cross_track_evidence_audit import ROOT, payload_hash, read_json, sha256_file, valid_lock, write_json
from post_v224_consolidation import (
    ACCESS_PATH,
    AUDIT_PATH,
    CONFIG_PATH,
    MANIFEST_PATH,
    RESULT_PATH,
    RESULTS_DOCUMENT,
    build_consolidation,
    render_results,
)


OUTCOME_LOCK = ROOT / "configs/post-v224-consolidation-outcome-lock.json"


def verify() -> dict:
    bundle = build_consolidation()
    if read_json(RESULT_PATH) != bundle["result"]:
        raise AssertionError("Consolidation result does not reconstruct exactly")
    if read_json(AUDIT_PATH) != bundle["audit"]:
        raise AssertionError("Consolidation audit does not reconstruct exactly")
    if read_json(ACCESS_PATH) != bundle["access"]:
        raise AssertionError("Consolidation access record does not reconstruct exactly")
    if RESULTS_DOCUMENT.read_text(encoding="utf-8") != render_results(bundle):
        raise AssertionError("Consolidation results document does not reconstruct exactly")
    manifest = read_json(MANIFEST_PATH)
    for row in manifest["artifacts"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise AssertionError(f"Manifest mismatch: {row['path']}")
    if not bundle["result"]["passed"]:
        raise AssertionError("Post-V224 consolidation gates did not pass")
    return bundle


def main() -> None:
    bundle = verify()
    artifacts = [RESULT_PATH, AUDIT_PATH, ACCESS_PATH, MANIFEST_PATH, RESULTS_DOCUMENT]
    lock = {
        "schema_version": "post_v224_consolidation_outcome_lock.v1",
        "experiment": "none_post_v224_maintenance_consolidation",
        "consolidation_config": str(CONFIG_PATH.relative_to(ROOT)),
        "consolidation_config_sha256": sha256_file(CONFIG_PATH),
        "implementation": "python/post_v224_consolidation.py",
        "implementation_sha256": sha256_file(ROOT / "python/post_v224_consolidation.py"),
        "runner": "python/run_post_v224_consolidation.py",
        "runner_sha256": sha256_file(ROOT / "python/run_post_v224_consolidation.py"),
        "tests": "python/test_post_v224_consolidation.py",
        "tests_sha256": sha256_file(ROOT / "python/test_post_v224_consolidation.py"),
        "verifier": "python/verify_and_freeze_post_v224_consolidation.py",
        "verifier_sha256": sha256_file(ROOT / "python/verify_and_freeze_post_v224_consolidation.py"),
        "artifacts": {str(path.relative_to(ROOT)): sha256_file(path) for path in artifacts},
        "outcome": {
            "passed": bundle["result"]["passed"],
            "gate_count": len(bundle["result"]["gates"]),
            "dependency_drift_addendum_count": bundle["result"]["dependency_drift"]["finding_count"],
            "reference_architecture_component_count": len(bundle["result"]["reference_architecture"]["component_ids"]),
            "reference_architecture_integration_gate_count": bundle["result"]["reference_architecture"]["integration_gate_count"],
            "historical_roadmap_count": bundle["result"]["navigation"]["historical_roadmap_count"],
            "authorized_next_experiment_count": 0,
            "decision": bundle["result"]["decision"],
            "protected_body_read_count": 0,
            "request_language_read_count": 0,
            "model_or_api_run_count": 0,
            "actual_execution_count": 0,
        },
        "authorization": {
            "new_experiment": False,
            "protected_or_request_language_access": False,
            "model_or_api": False,
            "training_or_tuning": False,
            "ontology_registration_or_trusted_state_mutation": False,
            "service_action_or_execution": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(OUTCOME_LOCK, lock)
    if not valid_lock(read_json(OUTCOME_LOCK)):
        raise AssertionError("Frozen consolidation outcome lock failed self-validation")
    print(json.dumps(lock["outcome"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
