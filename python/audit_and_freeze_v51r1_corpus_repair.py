#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


REPAIR_IMPLEMENTATION = (
    "python/audit_v51r1_corpus.py",
    "python/seal_v51r1_corpus.py",
    "scripts/run-v51r1-corpus-audit-repair.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v51r1-corpus-audit-repair.json")
    parser.add_argument("--plan", default="docs/v51r1-corpus-audit-repair-plan.md")
    parser.add_argument("--output", default="configs/v51r1-corpus-repair-lock.json")
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.config, args.plan, args.output)
    )
    if output.exists():
        raise RuntimeError("V51r1 corpus repair already frozen")
    config = json.loads(config_path.read_text())
    source_design = PROJECT_ROOT / config["sourceDesignLock"]
    source_implementation = PROJECT_ROOT / config["sourceImplementationLock"]
    source_audit = PROJECT_ROOT / config["sourceFailedCorpusAudit"]
    source_corpus = PROJECT_ROOT / config["sourceCorpus"]
    audit = json.loads(source_audit.read_text())
    errors = []
    if audit.get("passed") or audit.get("errors") != ["V51 structural case firewall failed"]:
        errors.append("V51r1 is not anchored to the expected single audit failure")
    identity_checks = {"fresh_cases", "support_query_disjoint", "unique_replication_cases"}
    if not all(
        value for key, value in audit.get("checks", {}).items() if key not in identity_checks
    ):
        errors.append("V51 source audit has a non-identity failure")
    invariants = config["invariants"]
    if any(invariants.values()):
        errors.append("V51r1 invariants must prohibit all listed changes and accesses")
    if config["singleAuthorizedChange"]["newIdentity"] != [
        "entities", "initial_state", "actions", "masks"
    ]:
        errors.append("V51r1 repair identity is not the full observation design")
    missing = [path for path in REPAIR_IMPLEMENTATION if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V51r1 repair implementation missing: {missing}")
    if errors:
        raise RuntimeError("; ".join(errors))
    lock = {
        "schema_version": 51,
        "revision": "r1",
        "experiment": "v51r1_corpus_repair_lock",
        "repair_config": str(config_path.relative_to(PROJECT_ROOT)),
        "repair_config_sha256": file_sha256(config_path),
        "repair_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "repair_plan_sha256": file_sha256(plan_path),
        "source_design_lock": str(source_design.relative_to(PROJECT_ROOT)),
        "source_design_lock_sha256": file_sha256(source_design),
        "source_implementation_lock": str(source_implementation.relative_to(PROJECT_ROOT)),
        "source_implementation_lock_sha256": file_sha256(source_implementation),
        "source_failed_corpus_audit": str(source_audit.relative_to(PROJECT_ROOT)),
        "source_failed_corpus_audit_sha256": file_sha256(source_audit),
        "source_corpus": str(source_corpus.relative_to(PROJECT_ROOT)),
        "source_corpus_sha256": file_sha256(source_corpus),
        "single_authorized_change": config["singleAuthorizedChange"],
        "repair_implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in REPAIR_IMPLEMENTATION
        },
        "authorization": {
            "reaudit_unchanged_corpus_once": True,
            "seal_if_repaired_audit_passes": True,
            "run_calibration": False,
            "regenerate_corpus": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
