#!/usr/bin/env python3
"""One-shot evaluator for the sealed V70 nine-model confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from evaluate_v68r2_development_screen import evaluate_record

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import validate_model
from v68_cassandra_pomdp import parse_cassandra_pomdp_file
from v68_multi_environment_exact import CommandChannelFamily
from v69_dominant_remapping import build_dominant_remapping_family
from v70_confirmatory_aggregation import aggregate_confirmatory_rows


def payload_hash(value: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_preflight(
    evaluator_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    lock = json.loads(evaluator_lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V70 evaluator lock payload hash mismatch")
    for path_key, hash_key, label in (
        ("evaluator", "evaluator_sha256", "evaluator"),
        ("aggregation", "aggregation_sha256", "aggregation"),
        ("family_implementation", "family_implementation_sha256", "family"),
        ("point_control_implementation", "point_control_implementation_sha256", "point controls"),
        ("unchanged_exact_record_evaluator", "unchanged_exact_record_evaluator_sha256", "record evaluator"),
        ("reporting_lock", "reporting_lock_sha256", "reporting lock"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V70 {label} hash mismatch")
    reporting = json.loads((PROJECT_ROOT / lock["reporting_lock"]).read_text())
    seal_path = PROJECT_ROOT / reporting["census_seal"]
    if file_sha256(seal_path) != reporting["census_seal_sha256"]:
        raise RuntimeError("V70 census seal hash mismatch")
    seal = json.loads(seal_path.read_text())
    census_path = PROJECT_ROOT / seal["census"]
    if file_sha256(census_path) != seal["census_sha256"]:
        raise RuntimeError("V70 sealed census hash mismatch")
    design_path = PROJECT_ROOT / seal["confirmatory_design_lock"]
    if file_sha256(design_path) != seal["confirmatory_design_lock_sha256"]:
        raise RuntimeError("V70 confirmatory design hash mismatch")
    config = json.loads(design_path.read_text())["config_payload"]
    records = read_jsonl(census_path)
    if len(records) != seal["record_count"] == lock["expected_records"]:
        raise RuntimeError("V70 census record count mismatch")
    models = {record["model_file"] for record in records}
    if len(models) != lock["expected_confirmatory_models"] == 9:
        raise RuntimeError("V70 confirmatory model count mismatch")
    if not lock["authorization"]["run_confirmatory_outcome_once"]:
        raise PermissionError("V70 evaluator lock does not authorize execution")
    if lock["authorization"]["rescore_development_models"]:
        raise PermissionError("V70 evaluator unexpectedly authorizes development rescoring")
    return lock, config, records, census_path


def run(evaluator_lock_path: Path) -> None:
    lock, config, records, census_path = load_preflight(evaluator_lock_path)
    output_dir = PROJECT_ROOT / "outputs/v70-confirmatory/evaluation"
    attempt_path = output_dir / "attempt.json"
    rows_path = output_dir / "record-results.jsonl"
    result_path = output_dir / "result.json"
    if output_dir.exists() or attempt_path.exists() or rows_path.exists() or result_path.exists():
        raise RuntimeError("V70 confirmatory evaluation has already been attempted")
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_attempt",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "attempt_number": 1,
        "expected_records": len(records),
        "expected_confirmatory_models": 9,
        "development_models_rescored": 0,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")

    source_dir = (
        PROJECT_ROOT
        / "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/POMDP"
    )
    primary_nodes = int(config["exactPlanning"]["primaryQuadratureNodes"])
    convergence_nodes = int(config["exactPlanning"]["convergenceQuadratureNodes"])
    horizon = int(config["exactPlanning"]["horizonActions"])
    tolerance = float(config["exactPlanning"]["tieTolerance"])
    low, high = map(float, config["unknownDynamicsFamily"]["thetaSupport"])
    model_specs = {row["file"]: row for row in config["confirmatoryModels"]}
    observed_models = {record["model_file"] for record in records}
    if observed_models != set(model_specs):
        raise RuntimeError("V70 sealed census/model assignment mismatch")
    families: dict[str, tuple[CommandChannelFamily, CommandChannelFamily]] = {}
    source_validation: dict[str, dict[str, bool]] = {}
    for model_file, spec in model_specs.items():
        model = parse_cassandra_pomdp_file(source_dir / model_file)
        checks = validate_model(model)
        if not all(checks.values()):
            raise RuntimeError(f"V70 source model no longer validates: {model_file}")
        source_validation[model_file] = checks
        families[model_file] = (
            build_dominant_remapping_family(
                model,
                spec["canonicalActionCycle"],
                quadrature_nodes=primary_nodes,
                theta_support=(low, high),
            ),
            build_dominant_remapping_family(
                model,
                spec["canonicalActionCycle"],
                quadrature_nodes=convergence_nodes,
                theta_support=(low, high),
            ),
        )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for model_file in model_specs:
        model_records = [
            record for record in records if record["model_file"] == model_file
        ]
        print(
            json.dumps(
                {
                    "event": "V70_model_start",
                    "model": model_file,
                    "records": len(model_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        primary, convergence = families[model_file]
        for record in model_records:
            row = evaluate_record(
                model_file,
                primary,
                convergence,
                record,
                horizon=horizon,
                tie_tolerance=tolerance,
                posterior_sampling_points=17,
                posterior_sampling_offset=1.0 / 34.0,
            )
            row["stratum"] = record["stratum"]
            rows.append(row)
        print(
            json.dumps(
                {"event": "V70_model_complete", "model": model_file},
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    aggregate = aggregate_confirmatory_rows(
        rows,
        config,
        expected_record_count=len(records),
        source_validation=source_validation,
        record_selection_or_rejection_count=0,
        development_models_rescored=0,
    )
    result = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_dominant_remapping_replication",
        **aggregate,
        "runtime_seconds": elapsed,
        "source_validation": source_validation,
        "records": len(rows),
        "record_results": str(rows_path.relative_to(PROJECT_ROOT)),
        "record_results_sha256": file_sha256(rows_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "access": {
            "confirmatory_records_evaluated": len(rows),
            "confirmatory_models_scored": len(families),
            "records_selected_rejected_or_replaced": 0,
            "development_models_rescored": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "V70_complete", "passed": result["passed"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator-lock", default="configs/v70-confirmatory-evaluator-lock.json"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    lock_path = Path(args.evaluator_lock)
    if not lock_path.is_absolute():
        lock_path = PROJECT_ROOT / lock_path
    if args.preflight:
        _, config, records, census_path = load_preflight(lock_path)
        print(
            json.dumps(
                {
                    "passed": True,
                    "records": len(records),
                    "confirmatory_models": len(config["confirmatoryModels"]),
                    "census": str(census_path.relative_to(PROJECT_ROOT)),
                    "development_models_rescored": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    run(lock_path)


if __name__ == "__main__":
    main()
