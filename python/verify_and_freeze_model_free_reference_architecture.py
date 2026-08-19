from __future__ import annotations

import json

from cross_track_evidence_audit import ROOT, payload_hash, read_json, sha256_file, valid_lock, write_json
from model_free_reference_architecture import (
    ACCESS_PATH,
    AUDIT_PATH,
    CONFIG_PATH,
    MANIFEST_PATH,
    RESULT_PATH,
    RESULTS_DOCUMENT,
    render_results,
    run_reference_architecture,
)


OUTCOME_LOCK = ROOT / "configs/model-free-reference-architecture-integration-outcome-lock.json"


def verify() -> dict:
    bundle = run_reference_architecture()
    if read_json(RESULT_PATH) != bundle["result"]:
        raise AssertionError("Reference architecture result does not reconstruct exactly")
    if read_json(AUDIT_PATH) != bundle["audit"]:
        raise AssertionError("Reference architecture audit does not reconstruct exactly")
    if read_json(ACCESS_PATH) != bundle["access"]:
        raise AssertionError("Reference architecture access record does not reconstruct exactly")
    if RESULTS_DOCUMENT.read_text(encoding="utf-8") != render_results(bundle):
        raise AssertionError("Reference architecture results document does not reconstruct exactly")
    manifest = read_json(MANIFEST_PATH)
    for row in manifest["artifacts"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise AssertionError(f"Manifest mismatch: {row['path']}")
    if not bundle["audit"]["passed"]:
        raise AssertionError("Reference architecture integration gates did not pass")
    return bundle


def main() -> None:
    bundle = verify()
    artifacts = [RESULT_PATH, AUDIT_PATH, ACCESS_PATH, MANIFEST_PATH, RESULTS_DOCUMENT]
    lock = {
        "schema_version": "model_free_reference_architecture_integration_outcome_lock.v1",
        "experiment": "none_software_consolidation_only",
        "architecture_config": str(CONFIG_PATH.relative_to(ROOT)),
        "architecture_config_sha256": sha256_file(CONFIG_PATH),
        "implementation": "python/model_free_reference_architecture.py",
        "implementation_sha256": sha256_file(ROOT / "python/model_free_reference_architecture.py"),
        "runner": "python/run_model_free_reference_architecture.py",
        "runner_sha256": sha256_file(ROOT / "python/run_model_free_reference_architecture.py"),
        "tests": "python/test_model_free_reference_architecture.py",
        "tests_sha256": sha256_file(ROOT / "python/test_model_free_reference_architecture.py"),
        "verifier": "python/verify_and_freeze_model_free_reference_architecture.py",
        "verifier_sha256": sha256_file(ROOT / "python/verify_and_freeze_model_free_reference_architecture.py"),
        "artifacts": {str(path.relative_to(ROOT)): sha256_file(path) for path in artifacts},
        "outcome": {
            "passed": bundle["audit"]["passed"],
            "gate_count": len(bundle["audit"]["gates"]),
            "source_outcome_lock_count": len(bundle["result"]["source_lock_integrity"]),
            "trusted_route": bundle["result"]["typed_version_space"]["routed_decision"],
            "other_route": bundle["result"]["other_defer"]["decision"],
            "semantic_root_action": bundle["result"]["outside_semantic_terminal_planner"]["root_action"],
            "decision": bundle["audit"]["decision"],
            "new_scientific_experiment_count": 0,
            "protected_body_read_count": 0,
            "request_language_read_count": 0,
            "model_or_api_run_count": 0,
            "actual_execution_count": 0,
        },
        "authorization": {
            "new_experiment": False,
            "external_language_claim": False,
            "model_or_api": False,
            "ontology_registration_or_trusted_state_mutation": False,
            "service_action_or_execution": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(OUTCOME_LOCK, lock)
    if not valid_lock(read_json(OUTCOME_LOCK)):
        raise AssertionError("Frozen architecture outcome lock failed self-validation")
    print(json.dumps(lock["outcome"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
