#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v55r1-results.md")
    parser.add_argument("--output", default="configs/v55r1-outcome-lock.json")
    args = parser.parse_args()
    audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V55r1 outcome already frozen")
    audit = json.loads(audit_path.read_text())
    result_path = PROJECT_ROOT / audit["result"]
    seal_path = PROJECT_ROOT / audit["population_seal"]
    v55_path = PROJECT_ROOT / "configs/v55-outcome-lock.json"
    v55 = json.loads(v55_path.read_text())
    if (
        not audit["passed"]
        or audit["result_sha256"] != file_sha256(result_path)
        or audit["population_seal_sha256"] != file_sha256(seal_path)
        or v55["decision"] != "retain_v55_failure_and_localize_failed_gates"
        or v55["qualification"]["passed"]
    ):
        raise RuntimeError("V55r1 post-result audit or frozen V55 failure is invalid")
    v55_passing = sum(v55["qualification"]["checks"].values()) == 19
    v55r1_passed = audit["qualification"]["passed"]
    combined = v55_passing and v55r1_passed
    lock = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_outcome_lock",
        "v55_standalone_qualification_passed": False,
        "v55_passing_gate_count": 19,
        "v55_outcome_lock": str(v55_path.relative_to(PROJECT_ROOT)),
        "v55_outcome_lock_sha256": file_sha256(v55_path),
        "v55r1_qualification_passed": v55r1_passed,
        "combined_planning_layer_qualification_passed": combined,
        "result": audit["result"],
        "result_sha256": audit["result_sha256"],
        "population_seal": audit["population_seal"],
        "population_seal_sha256": audit["population_seal_sha256"],
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "qualification": audit["qualification"],
        "decision": (
            "authorize_symbolic_and_probabilistic_policy_verification_preregistration_only"
            if combined else
            "retain_v55_and_v55r1_failures_and_continue_localization"
        ),
        "authorization": {
            "preregister_symbolic_and_probabilistic_policy_verification": combined,
            "run_formal_verification": False,
            "approximate_or_learned_planning": False,
            "long_horizon_claim": False,
            "construct_additional_v55_or_v55r1_population": False,
            "run_additional_v55_or_v55r1_evaluation": False,
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
