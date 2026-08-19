#!/usr/bin/env python3
"""Acquire the four exact V90 MLX snapshots without loading or generating from them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot_manifest(path: Path) -> dict[str, Any]:
    files = []
    weight_bytes = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = str(item.relative_to(path))
        size = item.stat().st_size
        resolved = item.resolve()
        row = {
            "path": relative,
            "size": size,
            "cache_blob_id": resolved.name if item.is_symlink() else None,
        }
        if relative.endswith(".safetensors"):
            weight_bytes += size
        else:
            row["sha256"] = file_sha256(item)
        files.append(row)
    return {
        "snapshot_path": str(path),
        "file_count": len(files),
        "weight_bytes": weight_bytes,
        "files": files,
        "manifest_sha256": payload_hash({"files": files}),
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v90-capacity-generation-acquisition-lock.json"
    output_root = PROJECT_ROOT / "outputs/v90-capacity-generation/model-acquisition"
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V90 acquisition lock payload mismatch")
    for key in ("design_lock", "corpus_seal", "corpus", "protocol", "tests", "downloader", "runner", "acquisition_auditor", "implementation_auditor"):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V90 acquisition dependency drifted: {key}")
    if not lock["authorization"]["download_pinned_snapshots_with_resumable_transport"]:
        raise RuntimeError("V90 model acquisition is not authorized")

    output_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    completed_before = 0
    for condition in lock["config_payload"]["modelConditions"]:
        manifest_path = output_root / f"{condition['id']}.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            snapshot = Path(existing["snapshot_path"])
            rebuilt = snapshot_manifest(snapshot)
            if rebuilt["manifest_sha256"] != existing["manifest_sha256"]:
                raise RuntimeError(f"existing V90 snapshot manifest drifted: {condition['id']}")
            manifests[condition["id"]] = existing
            completed_before += 1
            continue
        snapshot = Path(snapshot_download(
            repo_id=condition["repository"],
            revision=condition["revision"],
            allow_patterns=["*.json", "*.jinja", "*.model", "*.safetensors"],
        ))
        if snapshot.name != condition["revision"]:
            raise RuntimeError(f"snapshot revision path mismatch for {condition['id']}")
        manifest = snapshot_manifest(snapshot)
        if manifest["weight_bytes"] != condition["weightBytes"]:
            raise RuntimeError(
                f"weight byte mismatch for {condition['id']}: {manifest['weight_bytes']} != {condition['weightBytes']}"
            )
        required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
        present = {item["path"] for item in manifest["files"]}
        if not required <= present or not any(path.endswith(".safetensors") for path in present):
            raise RuntimeError(f"required snapshot files missing for {condition['id']}")
        artifact = {
            "schema_version": "90-capacity-generation-model-snapshot-manifest",
            "condition_id": condition["id"],
            "repository": condition["repository"],
            "revision": condition["revision"],
            "quantization_bits": condition["quantizationBits"],
            "expected_weight_bytes": condition["weightBytes"],
            **manifest,
            "model_load_count": 0,
            "model_generation_count": 0,
        }
        manifest_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        manifests[condition["id"]] = artifact
        progress = {
            "completed_condition_ids": list(manifests),
            "completed_condition_count": len(manifests),
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
        }
        (output_root / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")

    result = {
        "schema_version": "90-capacity-generation-model-acquisition-result",
        "experiment": "v90_capacity_generation_model_acquisition",
        "passed": len(manifests) == len(lock["config_payload"]["modelConditions"]),
        "condition_manifests": {
            key: {
                "path": str((output_root / f"{key}.json").relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(output_root / f"{key}.json"),
                "snapshot_path": value["snapshot_path"],
                "manifest_sha256": value["manifest_sha256"],
                "weight_bytes": value["weight_bytes"],
            }
            for key, value in sorted(manifests.items())
        },
        "access": {
            "snapshot_manifest_count_completed_before_this_invocation": completed_before,
            "snapshot_manifest_count": len(manifests),
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "manual_utterance_inspection_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": "model file acquisition and integrity manifests only; no model was loaded or queried",
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": result["passed"],
        "condition_manifests": result["condition_manifests"],
        "access": result["access"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
