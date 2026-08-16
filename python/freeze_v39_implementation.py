#!/usr/bin/env python3
"""Freeze V39 implementation and authorize corpus construction only."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


FILES = (
    "python/v39_compiler.py",
    "python/generate_v39_compiler.py",
    "python/audit_v39_implementation.py",
    "python/freeze_v39_implementation.py",
    "python/audit_v39_corpus.py",
    "python/seal_v39_corpus.py",
    "python/evaluate_v39_compiler.py",
    "python/audit_and_summarize_v39.py",
    "python/freeze_v39_outcome.py",
    "python/test_v39_compiler.py",
    "python/v38_focus_parser.py",
    "python/generate_v38_focus_parser.py",
    "python/v32_language.py",
    "python/v10_protocol.py",
    "python/v22r2_grounding.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v39-declared-language-compiler-lock.json")
    parser.add_argument("--audit", default="outputs/v39-declared-language-compiler/implementation-audit.json")
    parser.add_argument("--output", default="configs/v39-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.design_lock, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V39 implementation already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V39 implementation audit did not pass")
    for path in FILES:
        if not (PROJECT_ROOT / path).is_file():
            raise RuntimeError(f"V39 implementation incomplete: {path}")
    v32_path = PROJECT_ROOT / "configs/v32-factorized-semantics.json"
    v38_path = PROJECT_ROOT / design["config_payload"]["sourceV38OutcomeLock"]
    lock = {
        "schema_version": 39,
        "experiment": "v39_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "config_payload": design["config_payload"],
        "v32_config_payload": json.loads(v32_path.read_text()),
        "v32_config_sha256": file_sha256(v32_path),
        "source_v38_outcome_lock": str(v38_path.relative_to(PROJECT_ROOT)),
        "source_v38_outcome_lock_sha256": file_sha256(v38_path),
        "expected_corpus_sha256": audit["dry_run"]["expected_corpus_sha256"],
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in FILES},
        "authorization": {
            "construct_evaluation": True,
            "score_evaluation": False,
            "model_access": False,
            "v32_evaluation": False,
            "v28": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
