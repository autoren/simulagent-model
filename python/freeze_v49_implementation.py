#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from generate_v49_partial import build_population, corpus_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v49_belief.py",
    "python/generate_v49_partial.py",
    "python/evaluate_v49_partial.py",
    "python/audit_v49_corpus.py",
    "python/seal_v49_corpus.py",
    "python/audit_and_summarize_v49.py",
    "python/freeze_v49_outcome.py",
    "scripts/run-v49-passive-partial-observation.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v49-design-lock.json")
    parser.add_argument("--audit", default="outputs/v49-passive-partial-observation/implementation-audit.json")
    parser.add_argument("--output", default="configs/v49-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.design_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V49 implementation already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V49 implementation audit is not bound to the current design")
    implementation = {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION}
    rows = build_population(design["config_payload"])
    lock = {
        "schema_version": 49,
        "experiment": "v49_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": design["config_payload"],
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation": implementation,
        "expected_corpus_sha256": corpus_hash(rows),
        "authorization": {
            "construct_development_population": True,
            "run_development": False,
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
