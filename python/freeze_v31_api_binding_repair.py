#!/usr/bin/env python3
"""Lock the narrow pre-evaluation V31 MLX call-binding repair."""

from __future__ import annotations

import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/run_v31_training_api_repair.py",
    "python/freeze_v31_api_binding_repair.py",
    "python/freeze_v31_trained_systems_repaired.py",
    "docs/v31-api-binding-repair.md",
)


def main() -> None:
    protocol_path = PROJECT_ROOT / "configs/v31-signed-fact-adaptation-lock.json"
    output_path = PROJECT_ROOT / "configs/v31-api-binding-repair-lock.json"
    failed_path = PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/failed-attempts/frozen-api-binding/attempt.json"
    failed_output = PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/failed-attempts/frozen-api-binding/empty-output-directory"
    if output_path.exists():
        raise RuntimeError("V31 API-binding amendment already exists")
    if not failed_path.exists() or not failed_output.is_dir() or any(failed_output.iterdir()):
        raise RuntimeError("V31 failed API-binding attempt was not preserved as an empty run")
    failed = json.loads(failed_path.read_text())
    if failed["status"] != "started" or failed["evaluation_records_read"] != 0:
        raise RuntimeError("V31 failed attempt ledger is inconsistent with the amendment")
    forbidden = (
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/frozen-readout",
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/lora-readout",
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/sealed-evaluation",
        PROJECT_ROOT / "configs/v31-trained-systems-lock.json",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("V31 repair must lock before any successful training or evaluation")
    protocol = json.loads(protocol_path.read_text())
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"Original V31 implementation changed: {path}")
    amendment = {
        "schema_version": "31r1", "experiment": "v31_mlx_api_binding_repair",
        "protocol_lock": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(protocol_path),
        "failure": {
            "attempt": str(failed_path.relative_to(PROJECT_ROOT)),
            "attempt_sha256": file_sha256(failed_path),
            "exception": "TypeError: make_loss.<locals>.loss_fn() missing 1 required positional argument: 'entity_mask'",
            "completed_seeds": 0, "optimizer_updates": 0, "parameter_artifacts": 0,
            "evaluation_records_read": 0, "evaluation_features_read": 0,
        },
        "repair": {
            "scope": "inject_captured_module_as_locked_closure_first_argument_only",
            "extra_training_runs": 0, "extra_seeds": 0, "method_changes": 0,
            "gate_changes": 0, "evaluation_code_changes": 0,
        },
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION},
    }
    amendment["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(amendment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(amendment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
