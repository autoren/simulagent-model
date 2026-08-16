#!/usr/bin/env python3
"""Freeze the audited V37 scientific outcome and narrow authorization."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v37-semantic-invariance/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v37-semantic-invariance/post-result-audit.json")
    parser.add_argument("--output", default="configs/v37-outcome-lock.json")
    args = parser.parse_args()
    result_path, audit_path, output_path = (
        (PROJECT_ROOT / value).resolve() for value in (args.result, args.audit, args.output)
    )
    if output_path.exists():
        raise RuntimeError("V37 outcome is already frozen")
    result, audit = json.loads(result_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["result_sha256"] != file_sha256(result_path):
        raise RuntimeError("V37 post-result audit did not pass")
    qualified = result["qualification"]["passed"]
    lock = {
        "schema_version": 37,
        "experiment": "v37_outcome_lock",
        "scientific_decision": result["decision"],
        "qualification_passed": qualified,
        "component_selection": result["component_selection"],
        "selected_validation": result["selected_validation"],
        "frozen_v36_baseline": result["frozen_v36_baseline"],
        "gate_checks": result["qualification"]["checks"],
        "compiled_truth_gain_over_frozen_v36": result["qualification"]["compiled_truth_gain_over_frozen_v36"],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "preregister_fresh_semantic_confirmation": qualified,
            "construct_fresh_semantic_confirmation": False,
            "end_to_end_relational_suite": False,
            "v32_evaluation": False,
            "v28": False,
            "adapter_training": False,
            "change_backbone": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
