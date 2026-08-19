#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from retrieve_v216_bounded_external_artifacts import dependency_hashes_exact
from v10_protocol import file_sha256
from v216_bounded_external_artifact_population import audit_population, score_population
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v216-bounded-external-artifact-population-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V216 design lock or dependency hash mismatch")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()) or any((PROJECT_ROOT / payload["rawPath"]).exists() for payload in config["payloads"]):
        raise RuntimeError("V216 formal retrieval, population, or result output already exists")
    subprocess.run([sys.executable, str(PROJECT_ROOT / lock["retrieval_worker"]), str(lock_path)], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, str(PROJECT_ROOT / lock["population_worker"]), str(lock_path)], cwd=PROJECT_ROOT, check=True)
    retrieval_manifest = json.loads(artifacts["retrievalManifest"].read_text())
    parser_control = json.loads(artifacts["parserControl"].read_text())
    development_public = read_jsonl(artifacts["developmentPublic"])
    development_truth = read_jsonl(artifacts["developmentTruth"])
    protected_public = read_jsonl(artifacts["protectedPublic"])
    protected_truth = read_jsonl(artifacts["protectedTruth"])
    split = json.loads(artifacts["split"].read_text())
    population_manifest = json.loads(artifacts["populationManifest"].read_text())
    metrics = score_population(
        retrieval_manifest,
        parser_control,
        development_public + protected_public,
        development_truth + protected_truth,
        split,
        population_manifest,
        config,
        PROJECT_ROOT,
    )
    access = {
        "bounded_retrieval_run_count": 1,
        "population_build_run_count": 1,
        "unlisted_network_request_count": retrieval_manifest["unlisted_network_request_count"],
        "payload_count": len(retrieval_manifest["attempts"]),
        "v213_protected_access_count": 0,
        "protected_downstream_method_evaluation_count": 0,
        "protected_manual_semantic_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "model_api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_action_count": 0,
        "external_side_effect_count_beyond_read_only_retrieval": 0,
        "actual_execution_count": 0,
    }
    audit = audit_population(metrics, access, config)
    raw_hashes = {
        payload["payloadId"]: file_sha256(PROJECT_ROOT / payload["rawPath"])
        for payload in config["payloads"]
    }
    summary = {
        "schema_version": "216-bounded-external-artifact-population-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "raw_payload_sha256": raw_hashes,
        "retrieval_manifest_sha256": file_sha256(artifacts["retrievalManifest"]),
        "population_manifest_sha256": file_sha256(artifacts["populationManifest"]),
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "216-bounded-external-artifact-population-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "authorization": {
            "design_V217_deterministic_external_reconstruction_controls": audit["passed"],
            "open_protected_for_method_evaluation_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

