#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v50-implementation-lock.json")
    parser.add_argument("--audit", default="outputs/v50-history-dependent-belief-filtering/corpus-audit.json")
    parser.add_argument("--output", default="configs/v50-corpus-seal.json")
    args = parser.parse_args()
    lock_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.implementation_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V50 corpus already sealed")
    lock = json.loads(lock_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["implementation_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V50 corpus audit is not bound to implementation")
    data = PROJECT_ROOT / "data/v50-history-dependent-belief-filtering"
    corpora = {}
    for split in ("development_fit", "development_evaluation"):
        path = data / f"{split}.jsonl"
        corpora[split] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "records": len(path.read_text().splitlines()),
        }
    manifest = data / "manifest.json"
    seal = {
        "schema_version": 50,
        "experiment": "v50_corpus_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "corpus_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "corpus_audit_sha256": file_sha256(audit_path),
        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest),
        "corpus_sha256": audit["corpus_sha256"],
        "corpora": corpora,
        "authorization": {"run_development_once": True, "model_access": False},
    }
    seal["seal_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
