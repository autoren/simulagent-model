#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-lock", default="configs/v51r1-corpus-repair-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v51r1-corpus-audit-repair/corpus-audit.json"
    )
    parser.add_argument("--output", default="configs/v51r1-corpus-seal.json")
    args = parser.parse_args()
    repair_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.repair_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V51r1 corpus already sealed")
    repair = json.loads(repair_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["repair_lock_sha256"] != file_sha256(repair_path):
        raise RuntimeError("V51r1 corpus audit is not bound to repair lock")
    implementation = PROJECT_ROOT / repair["source_implementation_lock"]
    corpus = PROJECT_ROOT / repair["source_corpus"]
    manifest = PROJECT_ROOT / "data/v51-simulation-based-calibration/manifest.json"
    seal = {
        "schema_version": 51,
        "revision": "r1",
        "experiment": "v51r1_corpus_seal",
        "repair_lock": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_lock_sha256": file_sha256(repair_path),
        "implementation_lock": str(implementation.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation),
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
