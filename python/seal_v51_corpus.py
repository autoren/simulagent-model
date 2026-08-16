#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v51-implementation-lock.json"
    )
    parser.add_argument(
        "--audit", default="outputs/v51-simulation-based-calibration/corpus-audit.json"
    )
    parser.add_argument("--output", default="configs/v51-corpus-seal.json")
    args = parser.parse_args()
    lock_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.implementation_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V51 corpus already sealed")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["implementation_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V51 corpus audit is not bound to implementation")
    corpus = PROJECT_ROOT / "data/v51-simulation-based-calibration/replications.jsonl"
    manifest = PROJECT_ROOT / "data/v51-simulation-based-calibration/manifest.json"
    seal = {
        "schema_version": 51,
        "experiment": "v51_corpus_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "corpus_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "corpus_audit_sha256": file_sha256(audit_path),
        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest),
        "corpus": {
            "path": str(corpus.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(corpus),
            "records": len(corpus.read_text().splitlines()),
        },
        "authorization": {
            "run_calibration_once": True,
            "model_access": False,
        },
    }
    seal["seal_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
