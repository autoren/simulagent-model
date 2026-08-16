#!/usr/bin/env python3
"""Freeze the audited V35 continuation decision."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", default="configs/v35-binding-assembly-lock.json")
    parser.add_argument("--result", default="outputs/v35-binding-assembly/result.json")
    parser.add_argument("--audit", default="outputs/v35-binding-assembly/post-result-audit.json")
    parser.add_argument("--output", default="configs/v35-binding-outcome-lock.json")
    args = parser.parse_args()
    protocol_path, result_path, audit_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.protocol_lock, args.result, args.audit, args.output))
    if output_path.exists():
        raise RuntimeError("V35 outcome lock already exists")
    result, audit = json.loads(result_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["result_sha256"] != file_sha256(result_path):
        raise RuntimeError("V35 outcome cannot freeze an unaudited result")
    outcome = {
        "schema_version": 35, "experiment": "v35_outcome_lock",
        "protocol_lock": str(protocol_path.relative_to(PROJECT_ROOT)), "protocol_lock_sha256": file_sha256(protocol_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)), "post_result_audit_sha256": file_sha256(audit_path),
        "fit_selected_predicate_method": result["fit_selected_predicate_method"], "fit_selected_binding_method": result["fit_selected_binding_method"],
        "qualification": result["qualification"], "decision": result["decision"], "authorization": result["authorization"],
    }
    outcome["lock_payload_sha256"] = hashlib.sha256(json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
