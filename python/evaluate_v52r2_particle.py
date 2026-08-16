#!/usr/bin/env python3
"""Run the single V52r2 normalization-repair evaluation."""
from __future__ import annotations

import argparse
import json
import sys

import evaluate_v52_particle as base
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v52r2_particle import particle_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repair-implementation-lock",
        default="configs/v52r2-implementation-lock.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v52r2-joint-normalization-repair/evaluation",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.repair_implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_repair_evaluation_once"]:
        raise RuntimeError("V52r2 implementation lock does not authorize evaluation")
    for path, expected in lock["repair_implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V52r2 repair implementation changed: {path}")

    base.particle_inference = particle_inference
    sys.argv = [
        sys.argv[0],
        "--population-seal", lock["source_population_seal"],
        "--output-dir", args.output_dir,
    ]
    base.main()

    output = (PROJECT_ROOT / args.output_dir).resolve()
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    result.update({
        "experiment": "v52r2_joint_normalization_repair",
        "repair_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "repair_implementation_lock_sha256": file_sha256(lock_path),
        "repair_evaluation_run_number": 1,
    })
    result["data_access"].update({
        "particle_evaluation_runs": 2,
        "source_evaluation_runs": 1,
        "repair_evaluation_runs": 1,
        "repair_selected_from_frozen_source_outcome": True,
    })
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = output.parent / "evaluation-attempt.json"
    state = json.loads(attempt.read_text())
    state.update({
        "repair_evaluation_run": 1,
        "result_sha256": file_sha256(result_path),
    })
    attempt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
