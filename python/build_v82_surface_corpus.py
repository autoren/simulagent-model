#!/usr/bin/env python3
"""Materialize and seal the already locked V82 code/style population."""
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
    design_path = PROJECT_ROOT / "configs/v82-local-clarification-surface-design-lock.json"
    corpus_dir = PROJECT_ROOT / "data/v82-local-clarification-surface"
    corpus_path = corpus_dir / "records.jsonl"
    seal_path = corpus_dir / "corpus-seal.json"
    if corpus_path.exists() or seal_path.exists():
        raise RuntimeError("V82 surface corpus is already materialized")
    lock = json.loads(design_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V82 design lock drifted")
    if not lock["authorization"]["construct_and_seal_corpus"]:
        raise RuntimeError("V82 corpus construction is not authorized")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records = lock["config_payload"]["records"]
    corpus_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    seal = {
        "schema_version": "82-local-clarification-surface-corpus-seal",
        "experiment": "v82_local_clarification_surface_corpus_seal",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "record_count": len(records),
        "authorization": {
            "modify_corpus": False,
            "implement_and_audit_runner": True,
            "run_local_model": False,
            "run_API_model": False
        }
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"corpus": str(corpus_path), "sha256": file_sha256(corpus_path)}, indent=2))
    print(json.dumps({"seal": str(seal_path), "sha256": file_sha256(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
