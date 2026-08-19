#!/usr/bin/env python3
"""Materialize and seal the frozen V85 record population."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-design-lock.json"
    corpus_path = PROJECT_ROOT / "data/v85-local-adversarial-generator/records.jsonl"
    seal_path = PROJECT_ROOT / "data/v85-local-adversarial-generator/corpus-seal.json"
    if corpus_path.exists() or seal_path.exists():
        raise RuntimeError("V85 corpus is already materialized or sealed")
    design = json.loads(design_path.read_text())
    payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != design["lock_payload_sha256"]:
        raise RuntimeError("V85 design lock payload mismatch")
    if not design["authorization"]["construct_and_seal_corpus"]:
        raise RuntimeError("V85 design does not authorize corpus construction")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in design["config_payload"]["records"]
    ))
    seal = {
        "schema_version": "85-local-adversarial-generator-corpus-seal",
        "experiment": "v85_local_adversarial_generator_corpus_seal",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "record_count": len(design["config_payload"]["records"]),
        "authorization": {
            "modify_or_rebuild_corpus": False,
            "implement_and_audit_runner": True,
            "run_local_model": False,
            "run_API_model_or_train_adapter": False,
            "collect_human_or_original_user_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"corpus": str(corpus_path), "sha256": file_sha256(corpus_path), "seal": str(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
