#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v130_clarification_evidence_strength import run_audit


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v130-clarification-evidence-strength-lock.json"
    output_path = PROJECT_ROOT / "outputs/v130-clarification-evidence-strength/evaluation/result.json"
    if output_path.exists(): raise RuntimeError("V130 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V130 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V130 dependency drifted: {key}")
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text()); baseline = json.loads((PROJECT_ROOT / lock["baseline_config"]).read_text())
    summary = run_audit(catalog, baseline, lock["config_payload"])
    if run_audit(catalog, baseline, lock["config_payload"]) != summary: raise RuntimeError("V130 deterministic recomputation mismatch")
    result = {"schema_version": "130-clarification-evidence-strength-result", "experiment": lock["config_payload"]["experiment"], "passed": summary["feasibility_pass"], "decision": summary["decision"], "summary": summary, "deterministic_recomputation_exact": True, "access": {key: 0 for key in lock["config_payload"]["accessGates"]}, "claim_boundary": lock["config_payload"]["claimBoundary"]}
    write_json(output_path, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
