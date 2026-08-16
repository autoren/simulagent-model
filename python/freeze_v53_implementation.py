#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from generate_v53_smc2 import build_populations, population_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v53_smc2.py",
    "python/generate_v53_smc2.py",
    "python/evaluate_v53_smc2.py",
    "python/test_v53_smc2.py",
    "python/audit_v53_populations.py",
    "python/seal_v53_populations.py",
    "python/audit_and_summarize_v53.py",
    "python/freeze_v53_outcome.py",
    "scripts/run-v53r2-continuous-parameter-smc2.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v53r2-design-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v53r2-continuous-parameter-smc2/implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v53r2-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V53r2 implementation already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V53r2 implementation audit is not bound to current design")
    populations = build_populations(design["config_payload"])
    lock = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": design["config_payload"],
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "expected_population_sha256": population_hash(populations),
        "expected_population_counts": {
            name: len(rows) for name, rows in populations.items()
        },
        "authorization": {
            "construct_v53r2_populations": True,
            "run_v53r2_evaluation": False,
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
