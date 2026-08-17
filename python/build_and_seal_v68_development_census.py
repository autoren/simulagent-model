#!/usr/bin/env python3
"""Construct and seal the complete V68 development public-prefix census."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import validate_model
from v68_cassandra_pomdp import parse_cassandra_pomdp_file
from v68_multi_environment_exact import (
    build_command_channel_family,
    enumerate_public_prefixes,
    filter_action_observation_history,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v68-development-implementation-lock.json"
    census_path = PROJECT_ROOT / "outputs/v68-development-screening/census.jsonl"
    audit_path = PROJECT_ROOT / "outputs/v68-development-screening/census-audit.json"
    seal_path = PROJECT_ROOT / "configs/v68-development-census-seal.json"
    if seal_path.exists() or census_path.exists():
        raise RuntimeError("V68 development census already exists")
    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value for key, value in implementation.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    implementation_ok = bool(
        payload_hash(implementation_payload) == implementation["lock_payload_sha256"]
        and implementation["authorization"]["construct_and_seal_complete_development_census"]
        and not implementation["authorization"]["run_development_screen"]
        and not implementation["authorization"]["score_confirmatory_models"]
    )
    if not implementation_ok:
        errors.append("V68 implementation lock or census-only authorization failed")

    design_path = PROJECT_ROOT / implementation["development_design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    source_dir = (
        PROJECT_ROOT
        / "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/POMDP"
    )
    nodes = int(config["exactPlanning"]["primaryQuadratureNodes"])
    low, high = map(float, config["unknownDynamicsFamily"]["thetaSupport"])
    maximum_depth = max(config["publicPrefixCensus"]["depths"])
    records: list[dict[str, Any]] = []
    model_counts: dict[str, int] = {}
    depth_counts: dict[str, int] = {}
    source_validation_passes = 0
    belief_checks = 0
    belief_passes = 0
    for spec in config["developmentModels"]:
        path = source_dir / spec["file"]
        model = parse_cassandra_pomdp_file(path)
        checks = validate_model(model)
        source_validation_passes += int(all(checks.values()))
        family = build_command_channel_family(
            model,
            spec["canonicalActionCycle"],
            quadrature_nodes=nodes,
            theta_support=(low, high),
        )
        prefixes = enumerate_public_prefixes(family, maximum_depth=maximum_depth)
        model_counts[spec["file"]] = len(prefixes)
        for prefix in prefixes:
            action_names = [model.actions[action] for action in prefix.actions]
            observation_names = [
                model.observations[observation] for observation in prefix.observations
            ]
            recomputed, log_evidence = filter_action_observation_history(
                family, action_names, observation_names
            )
            belief_checks += 1
            belief_ok = bool(
                abs(float(recomputed.sum()) - 1.0) <= 1e-12
                and abs(log_evidence - prefix.log_evidence) <= 1e-12
            )
            belief_passes += int(belief_ok)
            depth_counts[str(prefix.depth)] = depth_counts.get(str(prefix.depth), 0) + 1
            records.append(
                {
                    "record_id": prefix.record_id,
                    "model_file": spec["file"],
                    "model_name": model.name,
                    "prefix_depth": prefix.depth,
                    "actions": action_names,
                    "observations": observation_names,
                    "history_probability": prefix.probability,
                    "log_evidence": prefix.log_evidence,
                }
            )

    unique_ok = len({record["record_id"] for record in records}) == len(records)
    census_ok = bool(
        len(records) >= config["gates"]["minimumRetainedRecords"]
        and source_validation_passes == len(config["developmentModels"]) == 4
        and belief_checks == belief_passes == len(records)
        and unique_ok
        and depth_counts.get("0") == 4
        and set(model_counts) == {row["file"] for row in config["developmentModels"]}
    )
    if not census_ok:
        errors.append("complete development census, validation, or belief checks failed")

    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    audit = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_census_audit",
        "passed": not errors and implementation_ok and census_ok,
        "decision": (
            "seal_complete_development_census_and_authorize_evaluator_implementation"
            if not errors
            else "reject_v68_development_census"
        ),
        "errors": errors,
        "checks": {
            "implementation_lock_and_census_only_authorization": implementation_ok,
            "all_four_source_models_validate": source_validation_passes == 4,
            "all_recomputed_beliefs_and_evidences_match": belief_checks == belief_passes,
            "record_ids_unique": unique_ok,
            "all_four_roots_retained": depth_counts.get("0") == 4,
            "minimum_record_count": len(records) >= config["gates"]["minimumRetainedRecords"],
            "complete_census": census_ok,
        },
        "record_count": len(records),
        "model_counts": model_counts,
        "depth_counts": depth_counts,
        "access": {
            "development_public_prefixes_constructed": len(records),
            "records_selected_rejected_or_replaced": 0,
            "development_planning_values_computed": 0,
            "confirmatory_models_scored": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    seal = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_census_seal",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "development_design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "development_design_lock_sha256": file_sha256(design_path),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "census_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "census_audit_sha256": file_sha256(audit_path),
        "record_count": len(records),
        "record_ids_sha256": hashlib.sha256(
            "\n".join(record["record_id"] for record in records).encode()
        ).hexdigest(),
        "model_counts": model_counts,
        "depth_counts": depth_counts,
        "selection_rejection_or_replacement_count": 0,
        "confirmatory_models_scored": 0,
        "authorization": {
            "modify_design_implementation_or_census": False,
            "write_and_audit_durable_development_evaluator": True,
            "run_development_screen": False,
            "score_confirmatory_models": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"seal": str(seal_path), "sha256": file_sha256(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
