#!/usr/bin/env python3
"""Freeze the single V37 feature artifact and authorize one evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/v37-semantic-invariance/features/metadata.json")
    parser.add_argument("--output", default="configs/v37-features-lock.json")
    args = parser.parse_args()
    metadata_path = (PROJECT_ROOT / args.metadata).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V37 features are already frozen")
    metadata = json.loads(metadata_path.read_text())
    if metadata["backbone_forward_passes"] != 6840 or metadata["truncated_prompts"] != 0:
        raise RuntimeError("V37 feature extraction violated the lock")
    artifact_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(artifact_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V37 feature artifact hash mismatch")
    seal_path = PROJECT_ROOT / metadata["corpus_seal"]
    if file_sha256(seal_path) != metadata["corpus_seal_sha256"]:
        raise RuntimeError("V37 feature metadata does not bind the corpus seal")
    lock = {
        "schema_version": 37,
        "experiment": "v37_features_lock",
        "corpus_seal": metadata["corpus_seal"],
        "corpus_seal_sha256": metadata["corpus_seal_sha256"],
        "feature_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "feature_metadata_sha256": file_sha256(metadata_path),
        "feature_artifact": metadata["feature_artifact"],
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "authorization": {
            "validation_evaluations": 1,
            "selection_source": "semantic_invariance_fit_only",
            "v32_evaluation": False,
            "v28": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
