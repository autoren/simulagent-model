#!/usr/bin/env python3
"""Freeze V45 implementation and authorize paired corpus construction only."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


FILES = (
    "python/v45_language.py", "python/generate_v45_language.py", "python/evaluate_v45_language.py",
    "python/test_v45_language.py", "python/audit_v45_implementation.py", "python/freeze_v45_implementation.py",
    "python/audit_v45_corpus.py", "python/seal_v45_corpus.py", "python/audit_and_summarize_v45.py",
    "python/freeze_v45_outcome.py", "python/v44_delayed.py", "python/evaluate_v44_delayed.py",
    "python/v43_language.py", "python/v43r1_measurement.py", "python/v39_compiler.py",
    "python/v38_focus_parser.py", "python/v42_stateful.py", "python/v22_relational.py",
    "python/v22r2_grounding.py", "python/v10_protocol.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v45-design-lock.json")
    parser.add_argument("--audit", default="outputs/v45-delayed-language-grounding/implementation-audit.json")
    parser.add_argument("--output", default="configs/v45-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.design_lock, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V45 implementation already frozen")
    design, audit = json.loads(design_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V45 implementation audit did not pass")
    for path in FILES:
        if not (PROJECT_ROOT / path).is_file():
            raise RuntimeError(f"V45 implementation incomplete: {path}")
    source_seal_path = PROJECT_ROOT / design["source_v44_corpus_seal"]
    lock = {
        "schema_version": 45,
        "experiment": "v45_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "config_payload": design["config_payload"],
        "source_v44_corpus_seal": str(source_seal_path.relative_to(PROJECT_ROOT)),
        "source_v44_corpus_seal_sha256": file_sha256(source_seal_path),
        "expected_corpus_sha256": audit["dry_run"]["expected_corpus_sha256"],
        "expected_counts": {
            key: audit["dry_run"][key]
            for key in (
                "mechanics", "support_sequences", "query_sequences", "state_clauses", "action_commands",
                "bound_action_commands", "wait_commands", "safety_challenges", "wait_counterfactual_pairs",
            )
        },
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in FILES},
        "authorization": {"construct_paired_language_corpus": True, "run_paired_development": False, "preregister_stochastic_foundation": False, "model_access": False},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
