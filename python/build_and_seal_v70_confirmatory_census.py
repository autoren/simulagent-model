#!/usr/bin/env python3
"""Construct and seal the complete V70 nine-model confirmatory census."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import validate_model
from v68_cassandra_pomdp import parse_cassandra_pomdp_file
from v68_multi_environment_exact import (
    enumerate_public_prefixes,
    filter_action_observation_history,
)
from v69_dominant_remapping import build_dominant_remapping_family


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    resource_path = PROJECT_ROOT / "configs/v70-confirmatory-resource-lock.json"
    census_path = PROJECT_ROOT / "outputs/v70-confirmatory/census.jsonl"
    audit_path = PROJECT_ROOT / "outputs/v70-confirmatory/census-audit.json"
    seal_path = PROJECT_ROOT / "configs/v70-confirmatory-census-seal.json"
    if seal_path.exists() or census_path.exists():
        raise RuntimeError("V70 confirmatory census already exists")
    resource = json.loads(resource_path.read_text())
    resource_payload = {
        key: value for key, value in resource.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    resource_ok = bool(
        payload_hash(resource_payload) == resource["lock_payload_sha256"]
        and resource["authorization"]["construct_and_seal_complete_confirmatory_census"]
        and not resource["authorization"]["write_reporting_or_evaluator"]
        and not resource["authorization"]["run_confirmatory_outcomes"]
        and not resource["authorization"]["drop_or_replace_models"]
    )
    if not resource_ok:
        errors.append("V70 resource lock or census-only authorization failed")

    design_path = PROJECT_ROOT / resource["confirmatory_design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    bounds = {row["file"]: row for row in resource["model_bounds"]}
    source_dir = (
        PROJECT_ROOT
        / "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/POMDP"
    )
    nodes = int(config["exactPlanning"]["primaryQuadratureNodes"])
    low, high = map(float, config["unknownDynamicsFamily"]["thetaSupport"])
    maximum_depth = max(config["publicPrefixCensus"]["depths"])
    records: list[dict[str, Any]] = []
    model_counts: dict[str, int] = {}
    stratum_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    source_validation_passes = 0
    belief_checks = 0
    belief_passes = 0
    upper_bound_passes = 0
    for spec in config["confirmatoryModels"]:
        model = parse_cassandra_pomdp_file(source_dir / spec["file"])
        checks = validate_model(model)
        source_validation_passes += int(all(checks.values()))
        family = build_dominant_remapping_family(
            model,
            spec["canonicalActionCycle"],
            quadrature_nodes=nodes,
            theta_support=(low, high),
        )
        prefixes = enumerate_public_prefixes(family, maximum_depth=maximum_depth)
        model_counts[spec["file"]] = len(prefixes)
        upper_bound_passes += int(
            len(prefixes) <= bounds[spec["file"]]["census_record_upper_bound"]
        )
        for prefix in prefixes:
            action_names = [model.actions[action] for action in prefix.actions]
            observation_names = [
                model.observations[observation] for observation in prefix.observations
            ]
            recomputed, log_evidence = filter_action_observation_history(
                family, action_names, observation_names
            )
            belief_checks += 1
            belief_passes += int(
                abs(float(recomputed.sum()) - 1.0) <= 1e-12
                and abs(log_evidence - prefix.log_evidence) <= 1e-12
            )
            depth_counts[str(prefix.depth)] += 1
            stratum_counts[spec["stratum"]] += 1
            records.append(
                {
                    "record_id": prefix.record_id,
                    "model_file": spec["file"],
                    "model_name": model.name,
                    "stratum": spec["stratum"],
                    "prefix_depth": prefix.depth,
                    "actions": action_names,
                    "observations": observation_names,
                    "history_probability": prefix.probability,
                    "log_evidence": prefix.log_evidence,
                }
            )

    expected_models = {row["file"] for row in config["confirmatoryModels"]}
    unique_ok = len({record["record_id"] for record in records}) == len(records)
    census_ok = bool(
        set(model_counts) == expected_models
        and len(model_counts) == source_validation_passes == upper_bound_passes == 9
        and belief_checks == belief_passes == len(records)
        and unique_ok
        and depth_counts["0"] == 9
        and sum(model_counts.values()) == len(records)
        and len(records) <= resource["totals"]["total_census_record_upper_bound"]
    )
    if not census_ok:
        errors.append("complete V70 census, validation, bounds, or belief checks failed")

    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    audit = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_census_audit",
        "passed": not errors and resource_ok and census_ok,
        "decision": (
            "seal_complete_confirmatory_census_and_authorize_reporting_specification_only"
            if not errors
            else "reject_v70_confirmatory_census"
        ),
        "errors": errors,
        "checks": {
            "resource_lock_and_census_only_authorization": resource_ok,
            "all_nine_source_models_validate": source_validation_passes == 9,
            "all_recomputed_V70_beliefs_and_evidences_match": belief_checks == belief_passes,
            "all_model_counts_within_frozen_resource_bounds": upper_bound_passes == 9,
            "record_ids_unique": unique_ok,
            "all_nine_roots_retained": depth_counts["0"] == 9,
            "all_nine_models_retained": set(model_counts) == expected_models,
            "complete_census": census_ok,
        },
        "record_count": len(records),
        "model_counts": model_counts,
        "stratum_record_counts": dict(stratum_counts),
        "depth_counts": dict(depth_counts),
        "access": {
            "confirmatory_public_prefixes_constructed": len(records),
            "records_selected_rejected_or_replaced": 0,
            "confirmatory_planning_values_computed": 0,
            "confirmatory_EIG_values_computed": 0,
            "development_models_rescored": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    seal = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_census_seal",
        "resource_lock": str(resource_path.relative_to(PROJECT_ROOT)),
        "resource_lock_sha256": file_sha256(resource_path),
        "confirmatory_design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "confirmatory_design_lock_sha256": file_sha256(design_path),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "census_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "census_audit_sha256": file_sha256(audit_path),
        "record_count": len(records),
        "record_ids_sha256": hashlib.sha256(
            "\n".join(record["record_id"] for record in records).encode()
        ).hexdigest(),
        "model_counts": model_counts,
        "stratum_record_counts": dict(stratum_counts),
        "depth_counts": dict(depth_counts),
        "selection_rejection_or_replacement_count": 0,
        "development_models_rescored": 0,
        "authorization": {
            "modify_design_resource_or_census": False,
            "write_and_audit_confirmatory_reporting_specification": True,
            "write_confirmatory_evaluator_before_reporting_lock": False,
            "run_confirmatory_outcomes": False,
            "drop_or_replace_models": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"seal": str(seal_path), "sha256": file_sha256(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
