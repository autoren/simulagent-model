#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/evaluation/result.json",
    )
    parser.add_argument(
        "--audit",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v55-results.md")
    parser.add_argument("--output", default="configs/v55-outcome-lock.json")
    args = parser.parse_args()
    result_path, audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V55 outcome already frozen")
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    passed = result["qualification"]["passed"] and audit["passed"]
    if (
        audit["result_sha256"] != file_sha256(result_path)
        or audit["qualification"] != result["qualification"]
    ):
        raise RuntimeError("V55 post-result audit is not bound to the result")
    lock = {
        "schema_version": 55,
        "experiment": "v55_outcome_lock",
        "decision": (
            "authorize_policy_verification_preregistration_only"
            if passed else "retain_v55_failure_and_localize_failed_gates"
        ),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "population_seal": result["population_seal"],
        "population_seal_sha256": result["population_seal_sha256"],
        "qualification": result["qualification"],
        "authorization": {
            "preregister_symbolic_and_probabilistic_policy_verification": passed,
            "run_formal_verification": False,
            "construct_new_planning_population": False,
            "run_additional_v55_evaluation": False,
            "long_horizon_claim": False,
            "approximate_or_learned_planning": False,
            "language_grounding": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
