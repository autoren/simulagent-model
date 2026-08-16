#!/usr/bin/env python3
"""Hash-lock all six V31 trained systems before opening sealed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def verified_manifest(path: Path, system: str, lock_hash: str, seeds: list[int]) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["system"] != system or manifest["protocol_lock_sha256"] != lock_hash:
        raise RuntimeError(f"V31 {system} manifest does not share the protocol lock")
    if manifest["training_runs"] != len(seeds) or set(manifest["seeds"]) != {str(seed) for seed in seeds}:
        raise RuntimeError(f"V31 {system} does not contain exactly the registered seeds")
    if manifest["evaluation_records_read"] or manifest["evaluation_features_read"]:
        raise RuntimeError(f"V31 {system} accessed evaluation before trained-system lock")
    for seed in seeds:
        entry = manifest["seeds"][str(seed)]
        ledger_path = PROJECT_ROOT / entry["ledger"]
        parameter_path = PROJECT_ROOT / entry["parameters"]
        if file_sha256(ledger_path) != entry["ledger_sha256"]:
            raise RuntimeError(f"V31 {system} seed {seed} ledger changed")
        if file_sha256(parameter_path) != entry["parameters_sha256"]:
            raise RuntimeError(f"V31 {system} seed {seed} parameters changed")
        ledger = json.loads(ledger_path.read_text())
        access = ledger["data_access"]
        if access["evaluation_records_read"] or access["evaluation_features_read"]:
            raise RuntimeError(f"V31 {system} seed {seed} ledger reports evaluation access")
        if access["checkpoint_selections"] or access["hyperparameter_selections"]:
            raise RuntimeError(f"V31 {system} seed {seed} performed forbidden selection")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", default="configs/v31-signed-fact-adaptation-lock.json")
    parser.add_argument("--frozen-manifest", default="outputs/v31-signed-fact-adaptation/frozen-readout/manifest.json")
    parser.add_argument("--lora-manifest", default="outputs/v31-signed-fact-adaptation/lora-readout/manifest.json")
    parser.add_argument("--output", default="configs/v31-trained-systems-lock.json")
    args = parser.parse_args()
    protocol_path = (PROJECT_ROOT / args.protocol_lock).resolve()
    frozen_path = (PROJECT_ROOT / args.frozen_manifest).resolve()
    lora_path = (PROJECT_ROOT / args.lora_manifest).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V31 trained-system lock already exists")
    forbidden = (
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/sealed-evaluation",
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/evaluation-attempt.json",
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/integration",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("V31 evaluation artifact exists before trained-system lock")
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = file_sha256(protocol_path)
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 locked implementation changed: {path}")
    seeds = protocol["training"]["seeds"]
    frozen = verified_manifest(frozen_path, "frozen_readout", protocol_hash, seeds)
    lora = verified_manifest(lora_path, "lora_readout", protocol_hash, seeds)
    feature_metadata_path = PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/fit-calibration-features/metadata.json"
    feature_metadata = json.loads(feature_metadata_path.read_text())
    if feature_metadata["protocol_lock_sha256"] != protocol_hash:
        raise RuntimeError("V31 frozen fit/calibration features do not share the lock")
    if feature_metadata["data_access"]["evaluation_records_read"] or feature_metadata["data_access"]["evaluation_features_read"]:
        raise RuntimeError("V31 frozen features report premature evaluation access")
    artifact_path = PROJECT_ROOT / feature_metadata["feature_artifact"]
    if file_sha256(artifact_path) != feature_metadata["feature_artifact_sha256"]:
        raise RuntimeError("V31 frozen fit/calibration feature artifact changed")
    trained = {
        "schema_version": 31,
        "experiment": "v31_trained_systems_pre_evaluation_lock",
        "protocol_lock": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": protocol_hash,
        "registered_seeds": seeds,
        "frozen_readout": {
            "manifest": str(frozen_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": file_sha256(frozen_path), "seeds": frozen["seeds"],
        },
        "lora_readout": {
            "manifest": str(lora_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": file_sha256(lora_path), "seeds": lora["seeds"],
        },
        "fit_calibration_features": {
            "metadata": str(feature_metadata_path.relative_to(PROJECT_ROOT)),
            "metadata_sha256": file_sha256(feature_metadata_path),
            "artifact": feature_metadata["feature_artifact"],
            "artifact_sha256": feature_metadata["feature_artifact_sha256"],
        },
        "proof": {
            "trained_systems": 6, "evaluation_records_read": 0,
            "evaluation_features_read": 0, "evaluation_predictions": 0,
            "checkpoint_selections": 0, "hyperparameter_selections": 0,
            "seed_selections": 0,
        },
    }
    trained["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(trained, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(trained, indent=2, sort_keys=True) + "\n")
    print(json.dumps(trained, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
