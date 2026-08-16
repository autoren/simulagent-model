#!/usr/bin/env python3
"""Seal V39 corpora and authorize one deterministic evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v39-implementation-lock.json")
    parser.add_argument("--audit", default="outputs/v39-declared-language-compiler/corpus-audit.json")
    parser.add_argument("--manifest", default="data/v39-declared-language-compiler/manifest.json")
    parser.add_argument("--output", default="configs/v39-corpus-seal.json")
    args = parser.parse_args()
    lock_path, audit_path, manifest_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.implementation_lock, args.audit, args.manifest, args.output))
    if output.exists():
        raise RuntimeError("V39 corpus already sealed")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["implementation_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V39 corpus audit did not pass")
    seal = {
        "schema_version": 39,
        "experiment": "v39_corpus_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "corpus_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "corpus_audit_sha256": file_sha256(audit_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "corpora": audit["artifacts"],
        "authorization": {
            "evaluation_attempts": 1,
            "model_forward_passes": 0,
            "v32_evaluation": False,
            "v28": False,
        },
    }
    seal["lock_payload_sha256"] = hashlib.sha256(json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
