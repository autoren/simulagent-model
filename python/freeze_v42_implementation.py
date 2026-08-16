#!/usr/bin/env python3
"""Freeze V42 implementation and authorize population construction only."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


FILES = (
    "python/v42_stateful.py",
    "python/generate_v42_sequential.py",
    "python/audit_v42_implementation.py",
    "python/freeze_v42_implementation.py",
    "python/audit_v42_corpus.py",
    "python/seal_v42_corpus.py",
    "python/evaluate_v42_sequential.py",
    "python/audit_and_summarize_v42.py",
    "python/freeze_v42_outcome.py",
    "python/test_v42_stateful.py",
    "python/v22_relational.py",
    "python/v10_protocol.py",
    "python/v22r2_grounding.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v42-design-lock.json")
    parser.add_argument("--audit", default="outputs/v42-sequential-state-foundation/implementation-audit.json")
    parser.add_argument("--output", default="configs/v42-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.design_lock, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V42 implementation already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V42 implementation audit did not pass")
    for path in FILES:
        if not (PROJECT_ROOT / path).is_file():
            raise RuntimeError(f"V42 implementation incomplete: {path}")
    source_path = PROJECT_ROOT / design["source_v41_outcome_lock"]
    lock = {
        "schema_version": 42,
        "experiment": "v42_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "config_payload": design["config_payload"],
        "source_v41_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v41_outcome_lock_sha256": file_sha256(source_path),
        "expected_corpus_sha256": audit["dry_run"]["expected_corpus_sha256"],
        "expected_counts": {
            key: audit["dry_run"][key]
            for key in ("mechanics", "support_sequences", "query_sequences", "partial_queries", "causal_order_pairs")
        },
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in FILES},
        "authorization": {
            "construct_development_population": True,
            "run_oracle_development": False,
            "language_grounding": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
