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
        default="outputs/v56-symbolic-probabilistic-policy-verification/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v56-results.md")
    parser.add_argument("--output", default="configs/v56-outcome-lock.json")
    args = parser.parse_args()
    audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V56 outcome already frozen")
    audit = json.loads(audit_path.read_text())
    result_path = PROJECT_ROOT / audit["result"]
    seal_path = PROJECT_ROOT / audit["verification_bundle_seal"]
    v55_path = PROJECT_ROOT / "configs/v55-outcome-lock.json"
    v55r1_path = PROJECT_ROOT / "configs/v55r1-outcome-lock.json"
    v55 = json.loads(v55_path.read_text())
    v55r1 = json.loads(v55r1_path.read_text())
    if (
        not audit["passed"]
        or audit["result_sha256"] != file_sha256(result_path)
        or audit["verification_bundle_seal_sha256"] != file_sha256(seal_path)
        or not v55r1["combined_planning_layer_qualification_passed"]
        or v55["qualification"]["passed"]
    ):
        raise RuntimeError("V56 post-result audit or planning-layer boundary is invalid")
    qualified = audit["qualification"]["passed"]
    lock = {
        "schema_version": 56,
        "experiment": "v56_outcome_lock",
        "qualification_passed": qualified,
        "result": audit["result"],
        "result_sha256": audit["result_sha256"],
        "verification_bundle_seal": audit["verification_bundle_seal"],
        "verification_bundle_seal_sha256": audit[
            "verification_bundle_seal_sha256"
        ],
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "v55_outcome_lock": str(v55_path.relative_to(PROJECT_ROOT)),
        "v55_outcome_lock_sha256": file_sha256(v55_path),
        "v55r1_outcome_lock": str(v55r1_path.relative_to(PROJECT_ROOT)),
        "v55r1_outcome_lock_sha256": file_sha256(v55r1_path),
        "qualification": audit["qualification"],
        "decision": (
            "authorize_definition_transfer_and_human_authored_language_preregistration_only"
            if qualified else
            "retain_v56_failure_and_localize_failed_gates"
        ),
        "authorization": {
            "preregister_definition_transfer_track": qualified,
            "preregister_human_authored_language_track": qualified,
            "run_definition_transfer_evaluation": False,
            "run_human_authored_language_evaluation": False,
            "run_additional_v56_candidate_verification": False,
            "formal_safety_claim": False,
            "worst_case_safety_claim": False,
            "parameter_uniform_claim": False,
            "long_horizon_claim": False,
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
