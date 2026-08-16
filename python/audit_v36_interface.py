#!/usr/bin/env python3
"""Audit the four fixed V36 fits before construction authorization."""

from __future__ import annotations

import argparse
import json

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v36_interface import COMPONENTS, unpack_component


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v36-implementation-lock.json")
    parser.add_argument("--ledger", default="outputs/v36-independent-confirmation/interface/fit-ledger.json")
    parser.add_argument("--output", default="outputs/v36-independent-confirmation/interface/audit.json")
    args = parser.parse_args()
    lock_path, ledger_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.implementation_lock, args.ledger, args.output))
    lock, ledger = json.loads(lock_path.read_text()), json.loads(ledger_path.read_text()); errors = []
    if ledger["implementation_lock_sha256"] != file_sha256(lock_path):
        errors.append("V36 fit ledger does not bind implementation lock")
    artifact_path = PROJECT_ROOT / ledger["parameter_artifact"]
    if file_sha256(artifact_path) != ledger["parameter_artifact_sha256"]:
        errors.append("V36 parameter artifact changed")
    artifact = np.load(artifact_path)
    shapes = {}
    for component in COMPONENTS:
        try:
            parameters = unpack_component(artifact, component)
            shapes[component] = {key: list(value.shape) for key, value in parameters.items()}
            if not all(np.all(np.isfinite(value)) for value in parameters.values()):
                errors.append(f"V36 {component} parameters are non-finite")
        except KeyError:
            errors.append(f"V36 parameter artifact lacks {component}")
    if "binding__projection" not in artifact.files or list(artifact["binding__projection"].shape) != [2560, 256]:
        errors.append("V36 binding projection missing or wrong shape")
    config = lock["config_payload"]
    expected_alphas = {
        "predicate": config["frozenInterface"]["predicate"]["alpha"],
        "binding": config["frozenInterface"]["binding"]["alpha"],
        "lexical_sign": config["frozenInterface"]["lexicalSign"]["alpha"],
        "outer_operation": config["frozenInterface"]["outerOperation"]["alpha"],
    }
    for component, alpha in expected_alphas.items():
        if ledger["components"][component]["alpha"] != alpha:
            errors.append(f"V36 {component} alpha differs from lock")
    access = ledger["data_access"]
    access_checks = {
        "exactly_four_fits": access["interface_fit_runs"] == 4,
        "no_selection": access["selection_runs"] == 0,
        "fit_only": access["fit_records_used"] == 1456,
        "no_legacy_calibration_targets": access["legacy_calibration_targets_read"] == 0,
        "no_legacy_evaluation": access["legacy_evaluation_records_read"] == 0,
        "no_confirmation": access["confirmation_records_read"] == 0,
        "no_model_forward": access["model_forward_passes"] == 0,
    }
    if not all(access_checks.values()):
        errors.append("V36 interface-fit access ledger violates design")
    result = {
        "schema_version": 36, "experiment": "v36_interface_audit", "passed": not errors,
        "decision": "authorize_v36_interface_lock" if not errors else "reject_v36_interface_fit",
        "errors": errors, "parameter_shapes": shapes, "expected_alphas": expected_alphas,
        "training_metrics": ledger["components"], "access_checks": access_checks,
        "source": {"implementation_lock_sha256": file_sha256(lock_path), "fit_ledger_sha256": file_sha256(ledger_path), "parameter_artifact_sha256": file_sha256(artifact_path)},
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
