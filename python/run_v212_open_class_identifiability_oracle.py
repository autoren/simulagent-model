#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v212_open_class_identifiability_oracle import (
    audit_metrics,
    materialize_cases,
    materialize_public_semantics,
    score_oracle,
)
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v212-open-class-identifiability-oracle-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V212 design lock or dependency hash mismatch")
    if not lock["authorization"]["run_one_model_free_oracle"]:
        raise RuntimeError("V212 formal oracle is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()):
        raise RuntimeError("V212 formal oracle output already exists")

    semantics = materialize_public_semantics(config)
    public_records, truth_records = materialize_cases(config, semantics)
    write_json(artifacts["publicSemantics"], semantics)
    write_jsonl(artifacts["publicCases"], public_records)
    write_jsonl(artifacts["sealedTruth"], truth_records)
    del truth_records

    worker_path = PROJECT_ROOT / lock["worker"]
    subprocess.run(
        [
            sys.executable,
            str(worker_path),
            "--semantics",
            str(artifacts["publicSemantics"]),
            "--public-cases",
            str(artifacts["publicCases"]),
            "--predictions",
            str(artifacts["predictions"]),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    worker_source = worker_path.read_text()
    prediction_freeze = {
        "schema_version": "212-prediction-freeze",
        "design_lock_sha256": file_sha256(lock_path),
        "public_semantics_sha256": file_sha256(artifacts["publicSemantics"]),
        "public_cases_sha256": file_sha256(artifacts["publicCases"]),
        "predictions_sha256": file_sha256(artifacts["predictions"]),
        "predictions_frozen_before_truth_join": True,
        "truth_join_opened_before_freeze": False,
        "oracle_worker_truth_path_count": worker_source.count("sealed-truth"),
        "oracle_worker_hidden_field_count": sum(
            worker_source.count(token) for token in ("expected_candidate", "concept_family")
        ),
    }
    write_json(artifacts["predictionFreeze"], prediction_freeze)

    sealed_truth = read_jsonl(artifacts["sealedTruth"])
    predictions = read_jsonl(artifacts["predictions"])
    metrics = score_oracle(public_records, sealed_truth, predictions, semantics, config)
    access = {
        "model_free_oracle_run_count": 1,
        "natural_language_surface_read_count": 0,
        "external_ontology_payload_read_count": 0,
        "protected_access_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    audit = audit_metrics(metrics, prediction_freeze, access, config)
    summary = {
        "schema_version": "212-representational-diagnosis-oracle-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "212-representational-diagnosis-oracle-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "authorization": {
            "design_V213_programmatic_concept_population": audit["passed"],
            "generate_population_or_read_external_payload": False,
            "run_local_or_API_model_or_training": False,
            "register_mutate_call_act_or_execute": False,
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
