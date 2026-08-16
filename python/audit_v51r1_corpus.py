#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def observation_design_key(row):
    return sha256_text(canonical_json({
        key: row[key] for key in ("entities", "initial_state", "actions", "masks")
    }))


def v50_observation_design_keys():
    keys = set()
    root = PROJECT_ROOT / "data/v50-history-dependent-belief-filtering"
    for split in ("development_fit", "development_evaluation"):
        for record in read(root / f"{split}.jsonl"):
            keys.update(
                observation_design_key(row)
                for row in record["agent_input"]["support_interventions"]
            )
            keys.update(
                observation_design_key(row)
                for row in record["agent_input"]["queries"]
            )
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-lock", default="configs/v51r1-corpus-repair-lock.json")
    parser.add_argument(
        "--output", default="outputs/v51r1-corpus-audit-repair/corpus-audit.json"
    )
    args = parser.parse_args()
    repair_path = (PROJECT_ROOT / args.repair_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    repair = json.loads(repair_path.read_text())
    source_lock_path = PROJECT_ROOT / repair["source_implementation_lock"]
    source_lock = json.loads(source_lock_path.read_text())
    source_audit_path = PROJECT_ROOT / repair["source_failed_corpus_audit"]
    source_audit = json.loads(source_audit_path.read_text())
    corpus_path = PROJECT_ROOT / repair["source_corpus"]
    errors = []
    if file_sha256(source_lock_path) != repair["source_implementation_lock_sha256"]:
        errors.append("source implementation lock changed")
    if file_sha256(source_audit_path) != repair["source_failed_corpus_audit_sha256"]:
        errors.append("source failed audit changed")
    if file_sha256(corpus_path) != repair["source_corpus_sha256"]:
        errors.append("frozen V51 corpus bytes changed")
    for path, expected in repair["repair_implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"repair implementation changed: {path}")
    for path, expected in source_lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"V51 implementation changed: {path}")

    identity_checks = {
        "fresh_cases", "support_query_disjoint", "unique_replication_cases"
    }
    source_non_identity = {
        key: value for key, value in source_audit["checks"].items()
        if key not in identity_checks
    }
    source_only_identity_error = source_audit["errors"] == [
        "V51 structural case firewall failed"
    ]
    source_non_identity_passed = all(source_non_identity.values())
    if not source_only_identity_error or not source_non_identity_passed:
        errors.append("source audit contains a non-identity failure")

    rows = read(corpus_path)
    support = [observation_design_key(row) for record in rows for row in record["supports"]]
    query = [observation_design_key(record["query"]) for record in rows]
    previous = v50_observation_design_keys()
    support_unique = len(set(support)) == len(support)
    query_unique = len(set(query)) == len(query)
    support_query_disjoint = not bool(set(support) & set(query))
    v50_disjoint = not bool((set(support) | set(query)) & previous)
    if not all((support_unique, query_unique, support_query_disjoint, v50_disjoint)):
        errors.append("complete observation-design firewall failed")
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v51r1-corpus-seal.json",
            "outputs/v51r1-corpus-audit-repair/calibration-attempt.json",
            "outputs/v51r1-corpus-audit-repair/calibration",
        )
    )
    if not downstream_absent:
        errors.append("V51r1 downstream calibration artifact already exists")
    audit = {
        "schema_version": 51,
        "revision": "r1",
        "experiment": "v51r1_corpus_identity_audit",
        "passed": not errors,
        "decision": "authorize_seal_of_unchanged_v51_corpus" if not errors else "repair_v51_corpus",
        "errors": errors,
        "repair_lock": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_lock_sha256": file_sha256(repair_path),
        "implementation_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(source_lock_path),
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "checks": {
            "source_audit_only_identity_error": source_only_identity_error,
            "source_non_identity_checks_passed": source_non_identity_passed,
            "all_support_observation_designs_unique": support_unique,
            "all_query_observation_designs_unique": query_unique,
            "support_query_observation_designs_disjoint": support_query_disjoint,
            "zero_v50_observation_design_overlap": v50_disjoint,
            "corpus_unchanged": file_sha256(corpus_path) == repair["source_corpus_sha256"],
            "downstream_absent": downstream_absent,
        },
        "counts": {
            "support_designs": len(support),
            "query_designs": len(query),
            "v50_designs": len(previous),
        },
        "data_access": {
            "calibration_outcomes_accessed": 0,
            "calibration_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
