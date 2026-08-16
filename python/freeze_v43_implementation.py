#!/usr/bin/env python3
"""Freeze V43 implementation and authorize paired corpus construction only."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


FILES = (
    "python/v43_language.py",
    "python/generate_v43_language.py",
    "python/audit_v43_implementation.py",
    "python/freeze_v43_implementation.py",
    "python/audit_v43_corpus.py",
    "python/seal_v43_corpus.py",
    "python/evaluate_v43_language.py",
    "python/audit_and_summarize_v43.py",
    "python/freeze_v43_outcome.py",
    "python/test_v43_language.py",
    "python/v42_stateful.py",
    "python/evaluate_v42_sequential.py",
    "python/v39_compiler.py",
    "python/v38_focus_parser.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
    "python/v10_protocol.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v43-design-lock.json")
    parser.add_argument("--audit", default="outputs/v43-sequential-language-grounding/implementation-audit.json")
    parser.add_argument("--output", default="configs/v43-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.design_lock, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V43 implementation already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V43 implementation audit did not pass")
    for path in FILES:
        if not (PROJECT_ROOT / path).is_file():
            raise RuntimeError(f"V43 implementation incomplete: {path}")
    source_seal_path = PROJECT_ROOT / design["source_v42_corpus_seal"]
    lock = {
        "schema_version": 43,
        "experiment": "v43_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "config_payload": design["config_payload"],
        "source_v42_corpus_seal": str(source_seal_path.relative_to(PROJECT_ROOT)),
        "source_v42_corpus_seal_sha256": file_sha256(source_seal_path),
        "expected_corpus_sha256": audit["dry_run"]["expected_corpus_sha256"],
        "expected_counts": {
            key: audit["dry_run"][key]
            for key in ("mechanics", "support_sequences", "query_sequences", "state_clauses", "action_commands", "safety_challenges", "causal_order_pairs")
        },
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in FILES},
        "authorization": {
            "construct_paired_language_corpus": True,
            "run_paired_development": False,
            "preregister_delayed_effects": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
