#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v129_complete_clarification_interface import run_audit


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v129-complete-clarification-interface-lock.json"
    output_path = PROJECT_ROOT / "outputs/v129-complete-clarification-interface/evaluation/result.json"
    if output_path.exists(): raise RuntimeError("V129 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V129 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V129 dependency drifted: {key}")
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    baseline = json.loads((PROJECT_ROOT / lock["baseline_config"]).read_text())
    v119 = json.loads((PROJECT_ROOT / lock["V119_config"]).read_text())
    summary = run_audit(catalog, baseline, v119, lock["config_payload"])
    if run_audit(catalog, baseline, v119, lock["config_payload"]) != summary: raise RuntimeError("V129 deterministic recomputation mismatch")
    access = {key.replace("maximum", "", 1)[0].lower() + key.replace("maximum", "", 1)[1:]: value for key, value in lock["config_payload"]["accessGates"].items()}
    result = {"schema_version": "129-complete-clarification-interface-result", "experiment": lock["config_payload"]["experiment"], "passed": summary["outcome_pass"], "decision": summary["decision"], "summary": summary, "deterministic_recomputation_exact": True, "access": access, "claim_boundary": lock["config_payload"]["claimBoundary"]}
    write_json(output_path, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
