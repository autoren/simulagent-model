#!/usr/bin/env python3
"""Seal the constructed V17 artifact before model extraction or scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    construction_path = Path("configs/v17-final-construction-lock.json")
    manifest_path = Path("data/v17-final/manifest.json")
    output_path = Path("configs/v17-final-evaluation-lock.json")
    construction = json.loads(construction_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest["construction_lock_sha256"] != file_sha256(construction_path):
        raise RuntimeError("V17 manifest does not share the construction lock")
    if manifest["config_sha256"] != construction["config_sha256"]:
        raise RuntimeError("V17 manifest/config mismatch")
    if manifest["validation"]["errors"]:
        raise RuntimeError("V17 final corpus failed structural validation")
    expected = construction["expected"]
    checks = {
        "records": expected["records"],
        "contexts": expected["contexts"],
        "intervention_groups": expected["interventionGroups"],
        "transition_cases": expected["transitionCases"],
        "transition_codes": expected["transitionCodes"],
    }
    for key, value in checks.items():
        if manifest["validation"][key] != value:
            raise RuntimeError(f"V17 manifest topology differs at {key}")
    artifact_hashes = {}
    for relative, expected_hash in manifest["artifact_sha256"].items():
        path = manifest_path.parent / relative
        digest = file_sha256(path)
        if digest != expected_hash:
            raise RuntimeError(f"V17 final artifact changed before seal: {relative}")
        artifact_hashes[str(path)] = digest
    lock = {
        "schema_version": 17,
        "experiment": "v17_sealed_one_shot_final_mechanic_evaluation",
        "construction_lock": str(construction_path),
        "construction_lock_sha256": file_sha256(construction_path),
        "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"], "artifact_sha256": artifact_hashes,
        "model": construction["model"], "c_value": construction["c_value"],
        "seed": construction["seed"], "max_sequence_length": construction["max_sequence_length"],
        "gates": construction["gates"], "expected": construction["expected"],
        "source": construction["source"], "implementation": construction["implementation"],
        "limits": construction["limits"],
        "data_access_at_seal": {
            **construction["data_access_before_lock"],
            "final_v17_mechanic_records_created": manifest["validation"]["records"],
            "final_v17_mechanic_records_read": 0,
        },
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V17 evaluation lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
