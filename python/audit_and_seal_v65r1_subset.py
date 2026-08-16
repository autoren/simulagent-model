#!/usr/bin/env python3
"""Independently reproduce, audit, and seal the public V65r1 subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def subset_digest(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"v65|subset|{seed}|{record_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/v65-smc2-eig-portability/manifest.json")
    parser.add_argument(
        "--audit", default="outputs/v65r1-nested-predictive-repair/subset-audit.json"
    )
    parser.add_argument("--output", default="configs/v65r1-subset-seal.json")
    args = parser.parse_args()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V65r1 subset already sealed")
    manifest = json.loads(manifest_path.read_text())
    lock_path = (PROJECT_ROOT / manifest["implementation_lock"]).resolve()
    lock = json.loads(lock_path.read_text())
    design_path = (PROJECT_ROOT / lock["design_lock"]).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    upstream_ok = bool(
        file_sha256(lock_path) == manifest["implementation_lock_sha256"]
        and lock["authorization"]["materialize_and_audit_subset"]
        and not lock["authorization"]["run_evaluation"]
        and file_sha256(PROJECT_ROOT / manifest["materializer"])
        == manifest["materializer_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in lock["source_sha256"].items()
        )
    )
    if not upstream_ok:
        errors.append("V65r1 implementation or materializer binding failed")

    source_path = (PROJECT_ROOT / manifest["source_public"]).resolve()
    source_hash_ok = bool(
        file_sha256(source_path) == manifest["source_public_sha256"]
        == config["pairedReuseBoundary"]["sourcePopulationSha256"]
        and manifest["source_records_loaded"] == 192
        and manifest["source_selection_audit_records_loaded"] == 0
        and manifest["source_evaluation_records_loaded"] == 0
    )
    if not source_hash_ok:
        errors.append("V64 public source hash or access accounting is invalid")

    files = {
        name: (PROJECT_ROOT / row["path"]).resolve()
        for name, row in manifest["files"].items()
    }
    file_hashes_ok = all(
        file_sha256(files[name]) == manifest["files"][name]["sha256"]
        for name in files
    )
    if not file_hashes_ok:
        errors.append("V65r1 subset file hash mismatch")
    source = read_jsonl(source_path)
    subset = read_jsonl(files["subset_public"])
    provenance = read_jsonl(files["subset_provenance"])
    counts_ok = bool(
        len(source) == 192
        and len(subset) == len(provenance) == config["subset"]["records"] == 48
        and manifest["counts"] == {"subset_public": 48, "subset_provenance": 48}
    )
    if not counts_ok:
        errors.append("V65r1 source or subset count differs from preregistration")

    allowed = set(config["subset"]["publicFieldsOnly"])
    public_firewall_ok = all(set(row) == allowed for row in source + subset)
    if not public_firewall_ok:
        errors.append("V65r1 public source or subset contains an undeclared field")

    seed = int(config["subset"]["selectionSeed"])
    independently_selected = []
    independent_provenance = []
    prefix_counts = {}
    source_by_id = {row["record_id"]: row for row in source}
    for prefix in config["subset"]["prefixLengths"]:
        stratum = [row for row in source if int(row["prefix_length"]) == int(prefix)]
        ranked = sorted(
            stratum,
            key=lambda row: (subset_digest(seed, str(row["record_id"])), str(row["record_id"])),
        )
        kept = ranked[: int(config["subset"]["recordsPerPrefixLength"])]
        independently_selected.extend(kept)
        independent_provenance.extend(
            {
                "record_id": row["record_id"],
                "prefix_length": prefix,
                "selection_digest": subset_digest(seed, str(row["record_id"])),
                "rank_within_prefix": rank,
            }
            for rank, row in enumerate(kept)
        )
        prefix_counts[str(prefix)] = len(kept)
    rule_reproduced = subset == independently_selected and provenance == independent_provenance
    if not rule_reproduced:
        errors.append("independent V65r1 hash selection does not reproduce sealed files")

    fidelity_ok = all(
        row == source_by_id.get(row["record_id"])
        and int(row["prefix_length"]) == len(row["actions"])
        and len(row["actions"]) == len(row["observations"])
        for row in subset
    )
    identity_ok = len({row["record_id"] for row in subset}) == 48
    quotas_ok = all(
        value == int(config["subset"]["recordsPerPrefixLength"])
        for value in prefix_counts.values()
    )
    if not fidelity_ok or not identity_ok or not quotas_ok:
        errors.append("V65r1 subset fidelity, uniqueness, history shape, or quota failed")

    selection_independence_ok = bool(
        config["pairedReuseBoundary"]["loadV64SelectionAudit"] is False
        and config["pairedReuseBoundary"]["loadV64EvaluationResultDuringSubsetSelection"] is False
        and config["pairedReuseBoundary"]["selectByExactEIGOrAction"] is False
        and set(provenance[0])
        == {"record_id", "prefix_length", "selection_digest", "rank_within_prefix"}
        and manifest["selection_rule"] == config["subset"]["selectionRule"]
    )
    if not selection_independence_ok:
        errors.append("V65r1 selection-independence boundary is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v65r1-evaluation-implementation-lock.json",
            "configs/v65r1-outcome-lock.json",
            "outputs/v65r1-nested-predictive-repair/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V65r1 evaluation artifact exists before subset seal")

    checks = {
        "implementation_and_materializer_bindings": upstream_ok,
        "V64_public_source_hash_and_access_accounting": source_hash_ok,
        "subset_file_hashes": file_hashes_ok,
        "preregistered_counts": counts_ok,
        "public_only_firewall": public_firewall_ok,
        "independent_hash_rule_reproduction": rule_reproduced,
        "record_fidelity_uniqueness_history_shape_and_prefix_quotas": (
            fidelity_ok and identity_ok and quotas_ok
        ),
        "selection_independent_of_truth_exact_EIG_and_action": selection_independence_ok,
        "evaluation_downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": "65r1",
        "experiment": "v65r1_subset_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "seal_v65r1_subset_and_authorize_evaluator_implementation_only"
            if not errors and all(checks.values())
            else "reject_v65r1_subset"
        ),
        "errors": errors,
        "checks": checks,
        "subset_summary": {
            "source_records": len(source),
            "selected_records": len(subset),
            "prefix_counts": prefix_counts,
            "selected_record_ids": [row["record_id"] for row in subset],
            "selection_seed": seed,
            "truth_or_exact_outcome_fields": 0,
        },
        "data_access": {
            "v64_selection_public_records_loaded": len(source),
            "v64_selection_audit_records_loaded": 0,
            "v64_evaluation_records_loaded": 0,
            "v65_subset_records_materialized": len(subset),
            "candidate_evaluation_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock_payload = {
        "schema_version": "65r1",
        "experiment": "v65r1_subset_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "materializer": manifest["materializer"],
        "materializer_sha256": manifest["materializer_sha256"],
        "seal_auditor": "python/audit_and_seal_v65r1_subset.py",
        "seal_auditor_sha256": file_sha256(Path(__file__).resolve()),
        "subset_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "subset_audit_sha256": file_sha256(audit_path),
        "source_public": manifest["source_public"],
        "source_public_sha256": manifest["source_public_sha256"],
        "files": manifest["files"],
        "counts": manifest["counts"],
        "prefix_counts": prefix_counts,
        "authorization": {
            "modify_or_rebuild_subset": False,
            "write_and_audit_evaluator": True,
            "run_evaluation": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock_payload["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock_payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
