#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from generate_v51_sbc import build_replications, corpus_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v51_sbc.py",
    "python/generate_v51_sbc.py",
    "python/evaluate_v51_sbc.py",
    "python/test_v51_sbc.py",
    "python/audit_v51_corpus.py",
    "python/seal_v51_corpus.py",
    "python/audit_and_summarize_v51.py",
    "python/freeze_v51_outcome.py",
    "scripts/run-v51-simulation-based-calibration.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v51-design-lock.json")
    parser.add_argument(
        "--audit",
        default="outputs/v51-simulation-based-calibration/implementation-audit.json",
    )
    parser.add_argument("--output", default="configs/v51-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V51 implementation already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V51 implementation audit is not bound to current design")
    rows = build_replications(design["config_payload"])
    lock = {
        "schema_version": 51,
        "experiment": "v51_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": design["config_payload"],
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "expected_corpus_sha256": corpus_hash(rows),
        "authorization": {
            "construct_calibration_replications": True,
            "run_calibration": False,
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
