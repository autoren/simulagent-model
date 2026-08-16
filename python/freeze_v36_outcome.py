#!/usr/bin/env python3
"""Freeze the audited V36 evidence-driven outcome."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v36-independent-confirmation/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v36-independent-confirmation/post-result-audit.json")
    parser.add_argument("--output", default="configs/v36-outcome-lock.json")
    args = parser.parse_args()
    result_path, audit_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.result, args.audit, args.output))
    if output_path.exists():
        raise RuntimeError("V36 outcome lock already exists")
    result, audit = json.loads(result_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["result_sha256"] != file_sha256(result_path):
        raise RuntimeError("V36 outcome cannot freeze an unaudited result")
    outcome = {
        "schema_version": 36, "experiment": "v36_outcome_lock",
        "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)), "post_result_audit_sha256": file_sha256(audit_path),
        "confirmation_passed": result["passed"], "scientific_decision": result["decision"], "confirmation_magnitude": result["confirmation_magnitude"],
        "metrics": result["metrics"], "gate_checks": result["gate_checks"],
        "authorization": result["authorization"],
    }
    outcome["lock_payload_sha256"] = hashlib.sha256(json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
