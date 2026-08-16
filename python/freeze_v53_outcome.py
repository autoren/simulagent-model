#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v53r2-continuous-parameter-smc2/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v53r2-continuous-parameter-smc2/post-result-audit.json")
    parser.add_argument("--summary", default="docs/v53r2-results.md")
    parser.add_argument("--output", default="configs/v53r2-outcome-lock.json")
    args = parser.parse_args()
    result_path, audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V53r2 outcome already frozen")
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["result_sha256"] != file_sha256(result_path)
        or audit["qualification_passed"] != result["qualification"]["passed"]
    ):
        raise RuntimeError("V53r2 result audit is not bound or did not pass")
    passed = result["qualification"]["passed"]
    lock = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_outcome_lock",
        "qualification_passed": passed,
        "decision": result["decision"],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "authorization": {
            "preregister_exact_one_step_expected_information_gain": passed,
            "construct_active_population": False,
            "reward_or_planning": False,
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
