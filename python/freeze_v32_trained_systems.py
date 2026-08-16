#!/usr/bin/env python3
"""Freeze the six trained V32 artifacts before opening evaluation strata."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", default="configs/v32-factorized-semantics-lock.json")
    parser.add_argument("--training-manifest", default="outputs/v32-factorized-semantics/training/manifest.json")
    parser.add_argument("--features", default="outputs/v32-factorized-semantics/fit-calibration-features/metadata.json")
    parser.add_argument("--output", default="configs/v32-trained-systems-lock.json")
    args = parser.parse_args()
    protocol_path, manifest_path, metadata_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.protocol_lock, args.training_manifest, args.features, args.output))
    if output_path.exists():
        raise RuntimeError("V32 trained-system lock already exists")
    if (PROJECT_ROOT / "outputs/v32-factorized-semantics/sealed-evaluation").exists() or (PROJECT_ROOT / "outputs/v32-factorized-semantics/evaluation-attempt.json").exists():
        raise RuntimeError("V32 evaluation was accessed before trained lock")
    protocol, manifest, metadata = map(lambda path: json.loads(path.read_text()), (protocol_path, manifest_path, metadata_path))
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V32 locked implementation changed: {path}")
    if manifest["protocol_lock_sha256"] != file_sha256(protocol_path) or metadata["protocol_lock_sha256"] != file_sha256(protocol_path):
        raise RuntimeError("V32 training artifacts do not share the protocol lock")
    if manifest["training_runs"] != 6 or set(manifest["systems"]) != {"monolithic", "joint_auxiliary"}:
        raise RuntimeError("V32 trained population differs from registration")
    feature_artifact = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_artifact) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V32 fit/calibration features changed")
    seeds = {str(seed) for seed in protocol["training"]["seeds"]}
    systems = {}
    for system, entry in manifest["systems"].items():
        if file_sha256(PROJECT_ROOT / entry["manifest"]) != entry["manifest_sha256"]:
            raise RuntimeError(f"V32 {system} manifest changed")
        if set(entry["seeds"]) != seeds:
            raise RuntimeError(f"V32 {system} seed population differs from registration")
        for seed in entry["seeds"].values():
            if file_sha256(PROJECT_ROOT / seed["parameters"]) != seed["parameters_sha256"] or file_sha256(PROJECT_ROOT / seed["ledger"]) != seed["ledger_sha256"]:
                raise RuntimeError(f"V32 {system} artifact changed")
            access = json.loads((PROJECT_ROOT / seed["ledger"]).read_text())["data_access"]
            if access["evaluation_records_read"] or access["evaluation_features_read"] or access["checkpoint_selections"] or access["hyperparameter_selections"]:
                raise RuntimeError(f"V32 {system} ledger violates the firewall")
        systems[system] = entry
    lock = {
        "schema_version": 32, "experiment": "v32_trained_system_lock", "protocol_lock": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(protocol_path), "training_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "training_manifest_sha256": file_sha256(manifest_path), "feature_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "feature_metadata_sha256": file_sha256(metadata_path), "systems": systems,
        "proof": {"trained_systems": 6, "evaluation_records_read": 0, "evaluation_features_read": 0, "evaluation_predictions": 0, "checkpoint_selections": 0, "hyperparameter_selections": 0},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__": main()
