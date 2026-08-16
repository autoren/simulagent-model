#!/usr/bin/env python3
"""Freeze the tested V37 implementation and authorize corpus construction only."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v37_language.py",
    "python/generate_v37_semantic_invariance.py",
    "python/v37_semantic.py",
    "python/audit_v37_implementation.py",
    "python/freeze_v37_implementation.py",
    "python/audit_v37_corpus.py",
    "python/seal_v37_corpus.py",
    "python/extract_v37_features_mlx.py",
    "python/freeze_v37_features.py",
    "python/evaluate_v37_semantic_invariance.py",
    "python/audit_and_summarize_v37.py",
    "python/freeze_v37_outcome.py",
    "python/test_v37_language.py",
    "python/test_v37_semantic.py",
    "python/evaluate_v30_signed_fact_language_mlx.py",
    "python/extract_v10_features_mlx.py",
    "python/v10_protocol.py",
    "python/v22r2_grounding.py",
    "python/v30_language.py",
    "python/v32_language.py",
    "python/v34_operation.py",
    "python/v36_interface.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v37-semantic-invariance-lock.json")
    parser.add_argument("--audit", default="outputs/v37-semantic-invariance/implementation-audit.json")
    parser.add_argument("--output", default="configs/v37-implementation-lock.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V37 implementation is already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v37_implementation_lock":
        raise RuntimeError("V37 implementation audit did not pass")
    if audit["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V37 implementation audit does not bind the design lock")
    for path in IMPLEMENTATION:
        if not (PROJECT_ROOT / path).is_file():
            raise RuntimeError(f"V37 implementation is incomplete: {path}")
    config = design["config_payload"]
    v32_path = PROJECT_ROOT / "configs/v32-factorized-semantics.json"
    v34_path = PROJECT_ROOT / "configs/v34-operation-interface.json"
    training_sources = {
        source["source"]: {
            "path": source["corpus"],
            "sha256": file_sha256(PROJECT_ROOT / source["corpus"]),
        }
        for source in config["allowedTrainingSources"]
    }
    v36_interface_path = PROJECT_ROOT / "configs/v36-interface-lock.json"
    v36_interface = json.loads(v36_interface_path.read_text())
    lock = {
        "schema_version": 37,
        "experiment": "v37_semantic_invariance_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "config_payload": config,
        "v32_config_payload": json.loads(v32_path.read_text()),
        "v32_config_sha256": file_sha256(v32_path),
        "v34_config_payload": json.loads(v34_path.read_text()),
        "v34_config_sha256": file_sha256(v34_path),
        "training_sources": training_sources,
        "frozen_v36_interface": {
            "path": str(v36_interface_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(v36_interface_path),
            "parameter_artifact": v36_interface["parameter_artifact"],
            "parameter_artifact_sha256": v36_interface["parameter_artifact_sha256"],
        },
        "expected_corpora": {
            "fit_corpus_sha256": audit["dry_run"]["fit_corpus_sha256"],
            "validation_corpus_sha256": audit["dry_run"]["validation_corpus_sha256"],
        },
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "authorization": {
            "construct_corpus": True,
            "extract_features": False,
            "fit_interface": False,
            "score_validation": False,
            "v32_evaluation": False,
            "v28": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
