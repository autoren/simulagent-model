#!/usr/bin/env python3
"""Seal V43 and authorize one paired development run."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v43-implementation-lock.json")
    parser.add_argument("--audit", default="outputs/v43-sequential-language-grounding/corpus-audit.json")
    parser.add_argument("--manifest", default="data/v43-sequential-language-grounding/manifest.json")
    parser.add_argument("--output", default="configs/v43-corpus-seal.json")
    args = parser.parse_args()
    lock_path, audit_path, manifest_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.implementation_lock, args.audit, args.manifest, args.output))
    if output.exists():
        raise RuntimeError("V43 corpus already sealed")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["implementation_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V43 corpus audit did not pass")
    seal = {
        "schema_version": 43,
        "experiment": "v43_corpus_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "corpus_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "corpus_audit_sha256": file_sha256(audit_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "corpora": audit["artifacts"],
        "authorization": {
            "paired_development_runs": 1,
            "language_model_forward_passes": 0,
            "adapter_training_runs": 0,
            "final_evaluations": 0,
        },
    }
    seal["lock_payload_sha256"] = hashlib.sha256(json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
