#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v54-exact-one-step-eig/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v54-exact-one-step-eig/post-result-audit.json")
    parser.add_argument("--summary", default="docs/v54-results.md")
    parser.add_argument("--output", default="configs/v54-outcome-lock.json")
    args = parser.parse_args()
    result_path, audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V54 outcome already frozen")
    result, audit = json.loads(result_path.read_text()), json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["result_sha256"] != file_sha256(result_path)
        or audit["qualification_passed"] != result["qualification"]["passed"]
    ):
        raise RuntimeError("V54 post-result audit is not intact and bound")
    passed = result["qualification"]["passed"]
    lock = {
        "schema_version": 54,
        "experiment": "v54_outcome_lock",
        "qualification_passed": passed,
        "decision": result["decision"],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "authorization": {
            "preregister_short_horizon_exact_bayes_adaptive_planning": passed,
            "construct_planning_population": False,
            "verification": False,
            "language_grounding": False,
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
