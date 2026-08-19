#!/usr/bin/env python3
"""Materialize and seal the already locked V81 population without model access."""
from __future__ import annotations

import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_lock_path = PROJECT_ROOT / "configs/v81-factorized-local-candidate-design-lock.json"
    corpus_dir = PROJECT_ROOT / "data/v81-factorized-local-candidate"
    corpus_path = corpus_dir / "records.jsonl"
    seal_path = corpus_dir / "corpus-seal.json"
    if corpus_path.exists() or seal_path.exists():
        raise RuntimeError("V81 corpus is already materialized")
    lock = json.loads(design_lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V81 design lock payload drifted")
    if not lock["authorization"]["construct_and_seal_corpus"]:
        raise RuntimeError("V81 design lock does not authorize corpus construction")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records = lock["config_payload"]["records"]
    corpus_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    seal = {
        "schema_version": "81-factorized-local-candidate-corpus-seal",
        "experiment": "v81_factorized_local_candidate_corpus_seal",
        "design_lock": str(design_lock_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_lock_path),
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "record_count": len(records),
        "authorization": {
            "modify_corpus": False,
            "implement_and_audit_runner": True,
            "run_local_model": False,
            "run_API_model": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"corpus": str(corpus_path), "sha256": file_sha256(corpus_path)}, indent=2))
    print(json.dumps({"seal": str(seal_path), "sha256": file_sha256(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
