#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v201r1_elapsed_time_verification_repair import evaluate_repair
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v201r1-elapsed-time-verification-repair-lock.json"; lock = json.loads(lock_path.read_text())
    if not valid_lock(lock): raise RuntimeError("invalid V201r1 lock")
    for key in [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V201r1 dependency drifted: {key}")
    output_root = PROJECT_ROOT / "outputs/v201r1-elapsed-time-verification-repair/repair"
    if output_root.exists(): raise RuntimeError("V201r1 may run only once")
    config = lock["config_payload"]
    source_lock = json.loads((PROJECT_ROOT / lock["source_V201_lock"]).read_text())
    repair = evaluate_repair(
        json.loads((PROJECT_ROOT / lock["source_failed_outcome_audit"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_result"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_evaluation_summary"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_access"]).read_text()),
        source_lock["config_payload"], config,
    )
    decision = config["decisionRule"]["ifExactSingleFieldRepairAndAllOtherVerificationChecksPass" if repair["passed"] else "otherwise"]
    result = {"schema_version": "201r1-elapsed-time-verification-repair-result", "experiment": config["experiment"], "passed": repair["passed"], "decision": decision, "claim_boundary": config["claimBoundary"], **repair, "source_artifact_mutation_count": 0, "model_or_policy_rerun_count": 0, "raw_model_response_read_count": 0, "API_call_count": 0, "actual_execution_count": 0}
    write_json(output_root / "result.json", result); print(json.dumps(result, indent=2, sort_keys=True))
    if not repair["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
