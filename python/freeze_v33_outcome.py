#!/usr/bin/env python3
"""Freeze the accepted V33 outcome and any development-qualified interface."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", default="configs/v33-development-adequacy-lock.json")
    parser.add_argument("--result", default="outputs/v33-development-adequacy/result.json")
    parser.add_argument("--audit", default="outputs/v33-development-adequacy/post-result-audit.json")
    parser.add_argument("--output", default="configs/v33-development-outcome-lock.json")
    args = parser.parse_args()
    protocol_path, result_path, audit_path, output_path = map(
        lambda value: (PROJECT_ROOT / value).resolve(), (args.protocol_lock, args.result, args.audit, args.output)
    )
    if output_path.exists(): raise RuntimeError("V33 outcome lock already exists")
    protocol, result, audit = (json.loads(path.read_text()) for path in (protocol_path, result_path, audit_path))
    if not audit["passed"] or audit["decision"] != "accept_v33_development_result" or audit["result_sha256"] != file_sha256(result_path):
        raise RuntimeError("V33 post-result audit does not authorize an outcome lock")
    if result["protocol_lock_sha256"] != file_sha256(protocol_path):
        raise RuntimeError("V33 result does not bind the protocol lock")
    root = result_path.parent
    for name, expected in {**result["parameter_artifacts"], **result["prediction_artifacts"]}.items():
        if file_sha256(root / name) != expected: raise RuntimeError(f"V33 outcome artifact changed: {name}")
    outcome = {
        "schema_version": 33, "experiment": "v33_development_outcome_lock",
        "protocol_lock": str(protocol_path.relative_to(PROJECT_ROOT)), "protocol_lock_sha256": file_sha256(protocol_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)), "post_result_audit_sha256": file_sha256(audit_path),
        "development_qualified": result["development_qualified"], "selected_system": result["selected_system"],
        "diagnosis": result["diagnosis"], "selected_search_configurations": result["selected_search_configurations"],
        "selected_parameter_artifacts": result["parameter_artifacts"] if result["development_qualified"] else {},
        "authorization": {
            "fresh_suite_preregistration": result["development_qualified"],
            "fresh_suite_construction": False,
            "v32_evaluation_reuse": False, "v28_replay": False,
        },
        "decision": "qualified_interface_may_enter_fresh_suite_preregistration" if result["development_qualified"] else "stop_before_fresh_suite_and_pivot_as_diagnosed",
    }
    outcome["lock_payload_sha256"] = hashlib.sha256(json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__": main()
