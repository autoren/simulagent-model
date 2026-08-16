#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v52r2_particle.py",
    "python/evaluate_v52r2_particle.py",
    "python/test_v52r2_repair.py",
    "python/audit_and_summarize_v52r2.py",
    "python/freeze_v52r2_outcome.py",
    "scripts/run-v52r2-joint-normalization-repair.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-lock", default="configs/v52r2-repair-lock.json")
    parser.add_argument(
        "--audit",
        default="outputs/v52r2-joint-normalization-repair/implementation-audit.json",
    )
    parser.add_argument("--output", default="configs/v52r2-implementation-lock.json")
    args = parser.parse_args()
    lock_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.repair_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V52r2 implementation already frozen")
    repair = json.loads(lock_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["repair_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V52r2 implementation audit is not bound to repair lock")
    lock = {
        "schema_version": 52,
        "revision": "r2",
        "experiment": "v52r2_implementation_lock",
        "repair_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "repair_lock_sha256": file_sha256(lock_path),
        "source_population_seal": repair["source_population_seal"],
        "source_population_seal_sha256": repair["source_population_seal_sha256"],
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "repair_implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "authorization": {
            "run_repair_evaluation_once": True,
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
