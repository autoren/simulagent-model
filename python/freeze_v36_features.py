#!/usr/bin/env python3
"""Freeze the one V36 feature artifact before opening confirmation metrics."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", default="configs/v36-confirmation-seal.json")
    parser.add_argument("--metadata", default="outputs/v36-independent-confirmation/features/metadata.json")
    parser.add_argument("--output", default="configs/v36-features-lock.json")
    args = parser.parse_args()
    seal_path, metadata_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.seal, args.metadata, args.output))
    if output_path.exists():
        raise RuntimeError("V36 feature lock already exists")
    seal, metadata = json.loads(seal_path.read_text()), json.loads(metadata_path.read_text())
    if metadata["confirmation_seal_sha256"] != file_sha256(seal_path) or metadata["backbone_forward_passes"] != 3510 or metadata["truncated_prompts"] != 0:
        raise RuntimeError("V36 feature metadata violates seal")
    artifact_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(artifact_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V36 feature artifact changed")
    lock = {
        "schema_version": 36, "experiment": "v36_features_lock",
        "confirmation_seal": str(seal_path.relative_to(PROJECT_ROOT)), "confirmation_seal_sha256": file_sha256(seal_path),
        "feature_metadata": str(metadata_path.relative_to(PROJECT_ROOT)), "feature_metadata_sha256": file_sha256(metadata_path),
        "feature_artifact": str(artifact_path.relative_to(PROJECT_ROOT)), "feature_artifact_sha256": file_sha256(artifact_path),
        "authorization": {"confirmation_evaluations": 1, "selection_runs": 0, "threshold_changes": 0, "model_forward_passes": 0, "reuse_v32_evaluation": False, "run_v28": False, "construct_final_suite": False},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
