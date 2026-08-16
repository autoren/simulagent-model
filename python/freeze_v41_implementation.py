#!/usr/bin/env python3
"""Freeze V41 construction and evaluation before confirmation construction."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


FILES = (
    "python/v41_interface.py", "python/generate_v41_confirmation.py",
    "python/audit_v41_implementation.py", "python/freeze_v41_implementation.py",
    "python/audit_v41_corpus.py", "python/seal_v41_corpus.py",
    "python/evaluate_v41_confirmation.py", "python/audit_and_summarize_v41.py",
    "python/freeze_v41_outcome.py", "python/test_v41_confirmation.py",
    "python/generate_v22_relational_development.py", "python/v22_relational.py",
    "python/v39_compiler.py", "python/v38_focus_parser.py", "python/v32_language.py",
    "python/v10_protocol.py", "python/v22r2_grounding.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v41-design-lock.json")
    parser.add_argument("--audit", default="outputs/v41-relational-mechanic-confirmation/implementation-audit.json")
    parser.add_argument("--output", default="configs/v41-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.design_lock, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V41 implementation already frozen")
    design = json.loads(design_path.read_text()); audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V41 implementation audit did not pass")
    for path in FILES:
        if not (PROJECT_ROOT / path).is_file():
            raise RuntimeError(f"V41 implementation incomplete: {path}")
    if file_sha256(PROJECT_ROOT / design["frozen_compiler"]) != design["frozen_compiler_sha256"] or file_sha256(PROJECT_ROOT / design["frozen_semantic_kernel"]) != design["frozen_semantic_kernel_sha256"]:
        raise RuntimeError("V41 frozen component changed")
    v22_path = PROJECT_ROOT / design["v22_config"]
    v32_path = PROJECT_ROOT / "configs/v32-factorized-semantics.json"
    lock = {
        "schema_version": 41, "experiment": "v41_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)), "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)), "implementation_audit_sha256": file_sha256(audit_path),
        "config_payload": design["config_payload"],
        "frozen_compiler": design["frozen_compiler"], "frozen_compiler_sha256": design["frozen_compiler_sha256"],
        "frozen_semantic_kernel": design["frozen_semantic_kernel"], "frozen_semantic_kernel_sha256": design["frozen_semantic_kernel_sha256"],
        "v22_config_payload": json.loads(v22_path.read_text()), "v22_config_sha256": file_sha256(v22_path),
        "v32_config_payload": json.loads(v32_path.read_text()), "v32_config_sha256": file_sha256(v32_path),
        "expected_corpus_sha256": audit["dry_run"]["expected_corpus_sha256"],
        "expected_counts": {key: audit["dry_run"][key] for key in ("mechanics", "support_scenes", "query_scenes", "language_clauses")},
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in FILES},
        "authorization": {"construct_confirmation": True, "score_confirmation": False, "model_access": False, "v22r2_evaluation": False, "v28": False},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n"); print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__": main()
