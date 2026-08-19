#!/usr/bin/env python3
"""Audit the prospective pilot Phase 2 implementation and write its one-shot lock."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from prospective_language_phase2 import validate_phase2_config
from prospective_language_pilot import load_study_config, sha256_json, verify_phase_1_bundle
from v10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "configs" / "prospective-language-pilot-v1-phase2.json"
STUDY_CONFIG_PATH = ROOT / "configs" / "prospective-language-pilot-v1.json"
PARTICIPANT_DIR = ROOT / "data" / "prospective-language-pilot" / "prospective-language-pilot-v1" / "P001"
LOCK_PATH = PARTICIPANT_DIR / "audit" / "phase2_run_lock.json"
OUTPUT_DIR = PARTICIPANT_DIR / "assistant" / "phase2_architecture"


def main() -> None:
    if LOCK_PATH.exists():
        raise RuntimeError("Phase 2 run lock already exists.")
    if OUTPUT_DIR.exists():
        raise RuntimeError("Phase 2 output exists before the run lock.")

    config = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    validate_phase2_config(config)
    phase1_report = verify_phase_1_bundle(load_study_config(STUDY_CONFIG_PATH), PARTICIPANT_DIR)
    if (
        phase1_report["verification"] != config["participant"]["phase1_required_verification"]
        or phase1_report["assistant_generation_count"]
        != config["participant"]["phase1_required_assistant_generation_count"]
    ):
        raise RuntimeError("Phase 1 acceptance gates are not satisfied.")

    dependencies = [
        "configs/prospective-language-pilot-v1-phase2.json",
        "python/prospective_language_phase2.py",
        "python/run_prospective_language_phase2_mlx.py",
        "python/test_prospective_language_phase2.py",
        "python/v154_adaptive_local_question_order.py",
        config["participant"]["public_requests"],
        config["participant"]["phase1_manifest"],
        config["model"]["model_manifest"],
    ]
    frozen_dependencies = [
        {"path": path, "sha256": file_sha256(ROOT / path)} for path in dependencies
    ]
    expected_hashes = {
        config["participant"]["public_requests"]: config["participant"]["public_requests_sha256"],
        config["participant"]["phase1_manifest"]: config["participant"]["phase1_manifest_sha256"],
        config["model"]["model_manifest"]: config["model"]["model_manifest_sha256"],
    }
    for dependency in frozen_dependencies:
        expected = expected_hashes.get(dependency["path"])
        if expected is not None and dependency["sha256"] != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {dependency['path']}")

    manifest = json.loads((ROOT / config["model"]["model_manifest"]).read_text(encoding="utf-8"))
    snapshot = Path(manifest["snapshot_path"])
    if (
        manifest["repository"] != config["model"]["repository"]
        or manifest["revision"] != config["model"]["revision"]
        or manifest["quantization_bits"] != config["model"]["quantization_bits"]
        or not snapshot.is_dir()
    ):
        raise RuntimeError("Pinned model manifest or local snapshot mismatch.")

    versions = {package: metadata.version(package) for package in ("mlx", "mlx-lm", "huggingface-hub")}
    if versions["mlx-lm"] != "0.31.3":
        raise RuntimeError("Phase 2 requires the previously qualified mlx-lm 0.31.3 runtime.")

    payload = {
        "schema_version": "prospective-language-pilot-v1-phase2-run-lock",
        "locked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": config["experiment"],
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "dependencies": frozen_dependencies,
        "phase1_verification": phase1_report,
        "model": {
            "repository": manifest["repository"],
            "revision": manifest["revision"],
            "snapshot_path": str(snapshot),
            "model_manifest_sha256": config["model"]["model_manifest_sha256"],
        },
        "runtime_versions": versions,
        "prelock_counts": {
            "model_load_count": 0,
            "model_generation_count": 0,
            "api_call_count": 0,
            "training_run_count": 0,
            "real_service_call_count": 0,
            "actual_execution_count": 0,
        },
        "authorization": {
            "run_exact_single_bounded_local_architecture_condition": True,
            "retry_reprompt_or_generate_an_alternate_condition": False,
            "use_private_scenario_cards_or_future_answers": False,
            "use_api_training_authority_action_or_execution": False,
        },
    }
    lock = {**payload, "lock_payload_sha256": sha256_json(payload)}
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "lock": str(LOCK_PATH.relative_to(ROOT)),
        "lock_payload_sha256": lock["lock_payload_sha256"],
        "dependency_count": len(frozen_dependencies),
        "model_generation_count": 0,
        "phase1_verification": phase1_report["verification"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
