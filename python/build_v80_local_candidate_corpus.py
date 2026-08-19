#!/usr/bin/env python3
"""Materialize and seal the preregistered V80 candidate corpus."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v80-local-candidate-generation-design-lock.json"
    builder_path = PROJECT_ROOT / "python/build_v80_local_candidate_corpus.py"
    output_dir = PROJECT_ROOT / "data/v80-local-candidate-generation"
    corpus_path = output_dir / "records.jsonl"
    seal_path = output_dir / "corpus-seal.json"
    if output_dir.exists():
        raise RuntimeError("V80 corpus already exists")
    design = json.loads(design_path.read_text())
    payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != design["lock_payload_sha256"]:
        raise RuntimeError("V80 design lock payload drifted")
    if not design["authorization"]["construct_and_seal_corpus"]:
        raise RuntimeError("V80 design does not authorize corpus construction")
    output_dir.mkdir(parents=True)
    records = design["config_payload"]["records"]
    corpus_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    seal = {
        "schema_version": "80-local-candidate-corpus-seal",
        "experiment": "v80_local_candidate_generation_corpus_seal",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "builder": str(builder_path.relative_to(PROJECT_ROOT)),
        "builder_sha256": file_sha256(builder_path),
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "record_count": len(records),
        "model_forward_pass_count": 0,
        "API_call_count": 0,
        "authorization": {
            "modify_corpus": False,
            "implement_and_audit_runner": True,
            "run_local_model": False,
            "run_API_model": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
