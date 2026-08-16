#!/usr/bin/env python3
"""Audit the independently split V59 population before sealing it."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from generate_v55r1_planning import prior_observation_design_keys
from generate_v59_planning import build_population, rows_hash
from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v59_planning import assert_search_payload_is_public


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population", default="data/v59-budgeted-root-sampled-planning"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v59-budgeted-root-sampled-planning/population-audit.json"
        ),
    )
    args = parser.parse_args()
    population = (PROJECT_ROOT / args.population).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    manifest_path = population / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lock_path = PROJECT_ROOT / manifest["implementation_lock"]
    lock = json.loads(lock_path.read_text())
    design_path = PROJECT_ROOT / lock["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    binding_ok = (
        lock["authorization"]["construct_v59_population"]
        and not lock["authorization"]["run_v59_evaluation"]
        and not lock["authorization"][
            "access_v59_audit_truth_during_candidate_evaluation"
        ]
        and manifest["implementation_lock_sha256"] == file_sha256(lock_path)
        and lock["design_lock_sha256"] == file_sha256(design_path)
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for section in (
                "implementation_files_sha256", "base_dependencies_sha256"
            )
            for path, digest in lock[section].items()
        )
    )
    if not binding_ok:
        errors.append("V59 population is not bound to its frozen implementation")

    public_path = PROJECT_ROOT / manifest["public_file"]["path"]
    audit_path = PROJECT_ROOT / manifest["audit_truth_file"]["path"]
    public_rows = read_jsonl(public_path)
    audit_rows = read_jsonl(audit_path)
    artifact_hashes_ok = (
        manifest["count"] == len(public_rows) == len(audit_rows)
        and file_sha256(public_path) == manifest["public_file"]["sha256"]
        and file_sha256(audit_path) == manifest["audit_truth_file"]["sha256"]
        and rows_hash(public_rows) == manifest["public_file"]["rows_sha256"]
        and rows_hash(audit_rows)
        == manifest["audit_truth_file"]["rows_sha256"]
    )
    if not artifact_hashes_ok:
        errors.append("V59 population artifact hashes or counts changed")

    regenerated_public, regenerated_audit = build_population(config)
    reproducible_ok = (
        public_rows == regenerated_public
        and audit_rows == regenerated_audit
        and rows_hash(regenerated_public) == manifest["public_file"]["rows_sha256"]
        and rows_hash(regenerated_audit)
        == manifest["audit_truth_file"]["rows_sha256"]
    )
    if not reproducible_ok:
        errors.append("V59 population is not byte-reproducible from the frozen generator")

    public_ids = [row["id"] for row in public_rows]
    audit_ids = [row["id"] for row in audit_rows]
    pairing_hash = sha256_text(canonical_json([
        {"public": public["id"], "audit": audit["id"]}
        for public, audit in zip(public_rows, audit_rows, strict=True)
    ]))
    split_firewall_ok = (
        public_ids == audit_ids
        and len(set(public_ids)) == len(public_ids)
        and pairing_hash == manifest["public_audit_pairing_sha256"]
        and manifest["audit_truth_file"]["candidate_access"] == "forbidden"
        and all(
            "truth" not in row
            and "target_program" not in canonical_json(row)
            and "target_theta" not in canonical_json(row)
            for row in public_rows
        )
        and all(set(row) == {"id", "schema_version", "record", "truth"} for row in audit_rows)
    )
    if not split_firewall_ok:
        errors.append("V59 public/audit pairing or truth firewall is invalid")

    schema_ok = True
    for row in public_rows:
        try:
            assert_search_payload_is_public(row)
        except Exception:
            schema_ok = False
            break
        schema_ok = schema_ok and (
            set(row) == {
                "id", "schema_version", "population", "record",
                "history_class", "horizon", "public",
            }
            and row["schema_version"] == 59
            and row["population"] == "budgeted_root_sampled_planning"
            and row["public"]["planning_horizon"] == row["horizon"]
            and len(row["public"]["supports"])
            == config["population"]["supportEpisodesPerTask"]
        )
    if not schema_ok:
        errors.append("V59 public schema or candidate-input firewall is invalid")

    horizons = Counter(row["horizon"] for row in public_rows)
    histories = Counter(row["history_class"] for row in public_rows)
    templates = Counter(
        row["truth"]["target_program_index"] for row in audit_rows
    )
    goals = Counter(
        (row["public"]["goal"]["atom"], row["public"]["goal"]["value"])
        for row in public_rows
    )
    census_ok = (
        len(public_rows) == config["population"]["tasks"] == 24
        and horizons == Counter({
            int(key): value
            for key, value in config["population"]["tasksPerHorizon"].items()
        })
        and histories == Counter(config["population"]["historyClasses"])
        and set(templates.values())
        == {config["population"]["tasksPerGeneratingTemplate"]}
        and len(templates) == 8
        and set(goals.values()) == {6}
        and len(goals) == 4
    )
    if not census_ok:
        errors.append("V59 horizon, history, template, or goal census is invalid")

    observation_keys = [
        case["observation_design_key"]
        for row in public_rows
        for case in row["public"]["supports"] + [row["public"]["query"]]
    ]
    prior_keys = prior_observation_design_keys()
    fresh_histories_ok = (
        len(observation_keys) == len(set(observation_keys))
        and not (set(observation_keys) & prior_keys)
    )
    if not fresh_histories_ok:
        errors.append("V59 observation designs are duplicated or overlap prior populations")

    downstream_paths = (
        "configs/v59-population-seal.json",
        "configs/v59-evaluation-implementation-lock.json",
        "configs/v59-outcome-lock.json",
        "outputs/v59-budgeted-root-sampled-planning/evaluation-attempt.json",
        "outputs/v59-budgeted-root-sampled-planning/evaluation",
        "outputs/v59-budgeted-root-sampled-planning/post-result-audit.json",
        "docs/v59-results.md",
    )
    integrity_ok = (
        not any((PROJECT_ROOT / path).exists() for path in downstream_paths)
        and not (PROJECT_ROOT / "data/v58-human-language").exists()
        and not (PROJECT_ROOT / "configs/v58-population-seal.json").exists()
    )
    if not integrity_ok:
        errors.append("V59 downstream or deferred V58 human-data firewall failed")

    audit = {
        "schema_version": 59,
        "experiment": "v59_population_audit",
        "passed": not errors,
        "decision": (
            "authorize_v59_population_seal" if not errors
            else "repair_v59_population"
        ),
        "errors": errors,
        "population": str(population.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "checks": {
            "implementation_authorization_and_binding": binding_ok,
            "artifact_hashes_counts_and_row_hashes": artifact_hashes_ok,
            "frozen_generator_byte_reproducibility": reproducible_ok,
            "public_audit_pairing_and_truth_firewall": split_firewall_ok,
            "public_schema_and_candidate_input_firewall": schema_ok,
            "horizon_history_template_and_goal_census": census_ok,
            "unique_fresh_observation_designs": fresh_histories_ok,
            "evaluation_downstream_and_v58_human_firewall": integrity_ok,
        },
        "metrics": {
            "records": len(public_rows),
            "horizons": dict(sorted(horizons.items())),
            "history_classes": dict(sorted(histories.items())),
            "generating_templates": len(templates),
            "goals": {f"{atom}={value}": count for (atom, value), count in sorted(goals.items())},
            "observation_designs": len(observation_keys),
            "prior_overlap_count": len(set(observation_keys) & prior_keys),
        },
        "data_access": {
            "v59_public_records_accessed_for_population_audit": len(public_rows),
            "v59_audit_truth_records_accessed_for_population_audit": len(audit_rows),
            "v59_candidate_records_accessed": 0,
            "v59_candidate_evaluation_runs": 0,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
