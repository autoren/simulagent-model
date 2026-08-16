#!/usr/bin/env python3
"""Audit V63 population structure and freeze its immutable file hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v63-implementation-lock.json")
    parser.add_argument(
        "--population-root", default="data/v63-external-unknown-dynamics/sealed-populations"
    )
    parser.add_argument("--audit", default="outputs/v63-external-unknown-dynamics/population-audit.json")
    parser.add_argument("--output", default="configs/v63-population-seal.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    root = (PROJECT_ROOT / args.population_root).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63 populations already sealed")
    lock = json.loads(lock_path.read_text())
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    config = design["config_payload"]
    errors: list[str] = []
    lock_ok = bool(
        lock["authorization"]["construct_and_audit_sealed_populations"]
        and not lock["authorization"]["run_candidate_evaluation"]
        and file_sha256(PROJECT_ROOT / lock["design_lock"]) == lock["design_lock_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for path, digest in lock["source_sha256"].items()
        )
    )
    if not lock_ok:
        errors.append("V63 implementation lock or frozen source hashes are not intact")
    names = [
        "exact-public.jsonl", "exact-truth.jsonl", "sbc-public.jsonl",
        "sbc-truth.jsonl", "scale-public.jsonl", "scale-truth.jsonl", "construction.json",
    ]
    if not all((root / name).is_file() for name in names):
        errors.append("V63 population files are incomplete")
    populations = {
        name: read_jsonl(root / f"{name}-public.jsonl") for name in ("exact", "sbc", "scale")
    }
    truths = {
        name: read_jsonl(root / f"{name}-truth.jsonl") for name in ("exact", "sbc", "scale")
    }
    expected_counts = {
        "exact": int(config["exactBenchmark"]["records"]),
        "sbc": int(config["simulationBasedCalibration"]["replications"]),
        "scale": int(config["scaleStress"]["records"]),
    }
    count_ok = all(len(populations[name]) == expected_counts[name] for name in expected_counts)
    if not count_ok:
        errors.append("V63 population record counts differ from the frozen design")
    public_truth_ids_ok = all(
        [row["id"] for row in populations[name]] == [row["id"] for row in truths[name]]
        for name in populations
    )
    if not public_truth_ids_ok:
        errors.append("V63 public and truth sidecar identifiers do not align")
    forbidden_truth_fields = {"identity", "identity_name", "theta", "current_state", "states"}
    truth_firewall_ok = all(
        not (forbidden_truth_fields & set(record))
        and all(not (forbidden_truth_fields & set(episode)) for episode in record["episodes"])
        for rows in populations.values() for record in rows
    )
    if not truth_firewall_ok:
        errors.append("V63 public records expose private truth")
    low, high = config["unknownDynamicsFamily"]["continuousParameter"]["support"]
    truth_support_ok = all(
        row["identity"] in (0, 1)
        and low < float(row["theta"]) < high
        and row["current_state"] in (2, 3)
        for rows in truths.values() for row in rows
    )
    if not truth_support_ok:
        errors.append("V63 truth sidecar contains out-of-support values")
    observation_support_ok = all(
        all(value in (1, 2) for episode in record["episodes"] for value in episode["observations"])
        for rows in populations.values() for record in rows
    )
    if not observation_support_ok:
        errors.append("V63 public record contains an invalid report")
    exact_identity_counts = Counter(row["identity"] for row in truths["exact"])
    exact_balance_ok = exact_identity_counts == Counter({0: 16, 1: 16})
    if not exact_balance_ok:
        errors.append("V63 exact benchmark identity balance is wrong")
    exact_shape_ok = all(
        len(record["episodes"]) == 5
        and [episode["role"] for episode in record["episodes"]] == [
            "support", "support", "support", "support", "current"
        ]
        and sorted(len(episode["observations"]) for episode in record["episodes"][:4])
        == [8, 8, 12, 12]
        and len(record["episodes"][-1]["observations"]) in (6, 10)
        for record in populations["exact"]
    )
    sbc_shape_ok = all(
        len(record["episodes"]) == 5
        and sorted(len(episode["observations"]) for episode in record["episodes"][:4])
        == [8, 8, 12, 12]
        and len(record["episodes"][-1]["observations"]) in (6, 10)
        for record in populations["sbc"]
    )
    scale_shape_ok = all(
        len(record["episodes"]) in config["scaleStress"]["episodeCounts"]
        and len({len(episode["observations"]) for episode in record["episodes"]}) == 1
        and len(record["episodes"][0]["observations"])
        in config["scaleStress"]["sequenceLengths"]
        for record in populations["scale"]
    )
    if not (exact_shape_ok and sbc_shape_ok and scale_shape_ok):
        errors.append("V63 episode roles, counts, or lengths differ from the frozen design")
    all_ids = [record["id"] for rows in populations.values() for record in rows]
    unique_ids_ok = len(all_ids) == len(set(all_ids))
    if not unique_ids_ok:
        errors.append("V63 record identifiers collide across populations")
    construction = json.loads((root / "construction.json").read_text())
    access_ok = bool(
        construction["candidate_inference_runs"] == 0
        and construction["human_record_access_count"] == 0
        and construction["simulated_human_record_count"] == 0
        and construction["model_forward_pass_count"] == 0
    )
    if not access_ok:
        errors.append("V63 population construction crossed an access firewall")
    file_hashes = {name: file_sha256(root / name) for name in names}
    public_observation_hashes = [
        hashlib.sha256(
            json.dumps(record["episodes"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for rows in populations.values() for record in rows
    ]
    unique_observation_fraction = len(set(public_observation_hashes)) / len(public_observation_hashes)
    audit = {
        "schema_version": 63,
        "experiment": "v63_population_audit",
        "passed": not errors,
        "decision": "seal_v63_populations" if not errors else "reject_v63_populations",
        "errors": errors,
        "checks": {
            "implementation_lock_and_sources_intact": lock_ok,
            "population_counts": count_ok,
            "public_truth_id_alignment": public_truth_ids_ok,
            "public_truth_firewall": truth_firewall_ok,
            "truth_support": truth_support_ok,
            "observation_support": observation_support_ok,
            "exact_identity_balance": exact_balance_ok,
            "exact_episode_shape": exact_shape_ok,
            "sbc_episode_shape": sbc_shape_ok,
            "scale_episode_shape": scale_shape_ok,
            "globally_unique_record_ids": unique_ids_ok,
            "construction_access_firewall": access_ok,
        },
        "counts": {name: len(rows) for name, rows in populations.items()},
        "exact_identity_counts": dict(sorted(exact_identity_counts.items())),
        "unique_public_observation_fraction": unique_observation_fraction,
        "file_sha256": file_hashes,
        "data_access": {
            "public_records_audited": sum(len(rows) for rows in populations.values()),
            "truth_sidecars_audited": sum(len(rows) for rows in truths.values()),
            "candidate_inference_runs": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    manifest = {
        "schema_version": 63,
        "experiment": "v63_sealed_populations_manifest",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "population_root": str(root.relative_to(PROJECT_ROOT)),
        "file_sha256": file_hashes,
        "counts": audit["counts"],
        "candidate_inference_runs_before_seal": 0,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    seal = {
        "schema_version": 63,
        "experiment": "v63_population_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "population_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "population_audit_sha256": file_sha256(audit_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "file_sha256": file_hashes,
        "authorization": {
            "modify_v63_design_implementation_or_populations": False,
            "write_and_audit_evaluation_implementation": True,
            "run_one_candidate_evaluation": False,
            "active_intervention_selection": False,
            "reward_or_planning_evaluation": False,
            "access_human_v58_records": False,
            "simulate_human_records": False,
            "model_access": False,
        },
    }
    seal["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "seal": seal}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
