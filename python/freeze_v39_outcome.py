#!/usr/bin/env python3
"""Freeze the audited V39 outcome and its narrow next-step authorization."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v39-declared-language-compiler/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v39-declared-language-compiler/post-result-audit.json")
    parser.add_argument("--output", default="configs/v39-outcome-lock.json")
    args = parser.parse_args()
    result_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.result, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V39 outcome already frozen")
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["result_sha256"] != file_sha256(result_path):
        raise RuntimeError("V39 post-result audit did not pass")
    passed = result["qualification"]["passed"]
    lock = {
        "schema_version": 39,
        "experiment": "v39_outcome_lock",
        "scientific_decision": result["decision"],
        "qualification_passed": passed,
        "supported": result["supported"],
        "safety": result["safety"],
        "novel_paraphrase_diagnostic": result["novel_paraphrase_diagnostic"],
        "gate_checks": result["qualification"]["checks"],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "preregister_fresh_supported_language_confirmation": passed,
            "construct_confirmation": False,
            "expand_to_open_paraphrase": False,
            "end_to_end_relational_suite": False,
            "v32_evaluation": False,
            "v28": False,
            "adapter_training": False,
            "change_backbone": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
