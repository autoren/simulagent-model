#!/usr/bin/env python3
"""Freeze the audited V60 evaluator and authorize one candidate run."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v60-approximate-belief-decision-calibration/evaluation-implementation-audit.json"
    )
    parser.add_argument("--implementation-lock", default="configs/v60-implementation-lock.json")
    parser.add_argument("--output", default="configs/v60-evaluation-implementation-lock.json")
    args = parser.parse_args()
    audit_path, implementation_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.implementation_lock, args.output)
    )
    if output.exists():
        raise RuntimeError("V60 evaluation implementation is already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["implementation_lock_sha256"] != file_sha256(implementation_path)
        or audit["population_seal_sha256"] != file_sha256(PROJECT_ROOT / audit["population_seal"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in ("evaluation_files_sha256", "frozen_dependencies_sha256")
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V60 evaluation audit is not intact and bound")
    lock = {
        "schema_version": 60, "experiment": "v60_evaluation_implementation_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "population_seal": audit["population_seal"],
        "population_seal_sha256": audit["population_seal_sha256"],
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "evaluation_files_sha256": audit["evaluation_files_sha256"],
        "frozen_dependencies_sha256": audit["frozen_dependencies_sha256"],
        "authorization": {
            "run_one_v60_candidate_evaluation": True,
            "run_additional_v60_candidate_evaluation": False,
            "change_v60_evaluation_implementation": False,
            "access_v59_audit_truth": False,
            "construct_or_modify_population": False,
            "collect_human_language": False,
            "exact_long_horizon_optimality_claim": False,
            "formal_safety_claim": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
