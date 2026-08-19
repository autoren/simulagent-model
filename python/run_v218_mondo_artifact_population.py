#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from retrieve_v218_mondo_artifacts import dependency_hashes_exact
from v10_protocol import file_sha256
from v218_mondo_artifact_population import audit_population, score_population
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_v218_mondo_artifact_population.py LOCK")
    lock = json.loads(Path(sys.argv[1]).resolve().read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V218 design lock or dependency hash mismatch")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if artifacts["summary"].exists() or artifacts["result"].exists():
        raise RuntimeError("V218 result already exists")
    retrieval = json.loads(artifacts["retrievalManifest"].read_text())
    parser_control = json.loads(artifacts["parserControl"].read_text())
    development_public = read_jsonl(artifacts["developmentPublic"])
    development_truth = read_jsonl(artifacts["developmentTruth"])
    protected_public = read_jsonl(artifacts["protectedPublic"])
    protected_truth = read_jsonl(artifacts["protectedTruth"])
    public_records = development_public + protected_public
    truth_records = development_truth + protected_truth
    split = json.loads(artifacts["split"].read_text())
    population_manifest = json.loads(artifacts["populationManifest"].read_text())
    metrics = score_population(retrieval, parser_control, public_records, truth_records, split, population_manifest, config, PROJECT_ROOT)
    access = {
        "bounded_retrieval_run_count": 1,
        "population_build_run_count": 1,
        "unlisted_network_request_count": retrieval["unlisted_network_request_count"],
        "payload_count": len(retrieval["attempts"]),
        "remote_import_resolution_count": retrieval["remote_import_resolution_count"],
        "v216_protected_access_count": 0,
        "v213_protected_access_count": 0,
        "protected_downstream_method_evaluation_count": split["protected_downstream_method_evaluation_count"],
        "protected_manual_semantic_inspection_count": split["protected_manual_semantic_inspection_count"],
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
    summary = {
        "schema_version": "218-mondo-artifact-population-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "retrieval_manifest_sha256": file_sha256(artifacts["retrievalManifest"]),
        "parser_control_sha256": file_sha256(artifacts["parserControl"]),
        "population_manifest_sha256": file_sha256(artifacts["populationManifest"]),
        "development_public_sha256": file_sha256(artifacts["developmentPublic"]),
        "development_truth_sha256": file_sha256(artifacts["developmentTruth"]),
        "protected_public_sha256": file_sha256(artifacts["protectedPublic"]),
        "protected_truth_sha256": file_sha256(artifacts["protectedTruth"]),
        "split_sha256": file_sha256(artifacts["split"]),
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "218-mondo-artifact-population-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "authorization": {
            "design_V219_deterministic_controls": audit["passed"],
            "open_protected_or_run_model": False,
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
