#!/usr/bin/env python3
"""Freeze the deterministic V16 scope-correct gate replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256


def main() -> None:
    plan_path = Path("docs/v16-scope-correct-replay-plan.md")
    output_path = Path("configs/v16-scope-correct-replay-lock.json")
    v15_lock_path = Path("configs/v15-full-pipeline-lock.json")
    result_path = Path("outputs/v15-full-pipeline/evaluation/result.json")
    audit_path = Path("outputs/v15-full-pipeline/group-scope-audit.json")
    implementation = {
        path: file_sha256(Path(path)) for path in (
            "python/replay_v16_scope_correct_gates.py",
            "python/test_v16_scope_correct.py",
            "python/v10_protocol.py",
        )
    }
    source = json.loads(result_path.read_text())
    if source["decision"] != "full_pipeline_transfer_fails_decompose_before_any_adaptation":
        raise RuntimeError("V15 result does not require scope replay")
    lock = {
        "schema_version": 16,
        "experiment": "v16_locked_scope_correct_v15_gate_replay",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "minimum_complete_group_accuracy": 0.5,
        "source": {
            "v15_lock": str(v15_lock_path), "v15_lock_sha256": file_sha256(v15_lock_path),
            "v15_result": str(result_path), "v15_result_sha256": file_sha256(result_path),
            "scope_audit": str(audit_path), "scope_audit_sha256": file_sha256(audit_path),
        },
        "implementation": implementation,
        "limits": {
            "model_fits_permitted": 0, "model_forward_passes_permitted": 0,
            "prediction_recomputations_permitted": 0, "threshold_changes_permitted": 0,
            "final_mechanic_evaluations_permitted": 0,
        },
        "data_access": source["data_access"],
        "lock_payload_sha256": "",
    }
    payload = {**lock, "lock_payload_sha256": ""}
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V16 lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
