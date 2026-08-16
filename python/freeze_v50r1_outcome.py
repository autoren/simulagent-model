#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-lock", default="configs/v50r1-repair-lock.json")
    parser.add_argument("--result", default="outputs/v50r1-execution-repair/development/result.json")
    parser.add_argument("--audit", default="outputs/v50r1-execution-repair/post-result-audit.json")
    parser.add_argument("--output", default="configs/v50r1-outcome-lock.json")
    args = parser.parse_args()
    repair_path, result_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.repair_lock, args.result, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V50r1 outcome already frozen")
    repair = json.loads(repair_path.read_text())
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["result_sha256"] != file_sha256(result_path):
        raise RuntimeError("V50r1 post-result audit is not bound to result")
    lock = {
        "schema_version": "50r1",
        "experiment": "v50r1_outcome_lock",
        "repair_lock": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_lock_sha256": file_sha256(repair_path),
        "source_failed_attempt": repair["failed_attempt"],
        "source_failed_attempt_sha256": repair["failed_attempt_sha256"],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "scientific_decision": result["decision"],
        "qualification_passed": result["qualification"]["passed"],
        "gate_checks": result["qualification"]["checks"],
        "metrics": result["metrics"],
        "authorization": result["authorization"],
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
