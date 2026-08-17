#!/usr/bin/env python3
"""Build and seal the complete public-prefix V71 development census."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v71_cassandra_pomdp import parse_cassandra_pomdp_file
from v71_sensor_codebook import enumerate_public_prefixes


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v71-sensor-codebook-source-lock.json"
    resource_lock_path = (
        PROJECT_ROOT
        / "configs/v71-sensor-codebook-development-resource-lock.json"
    )
    core_path = PROJECT_ROOT / "python/v71_sensor_codebook.py"
    core_tests_path = PROJECT_ROOT / "python/test_v71_sensor_codebook.py"
    builder_path = PROJECT_ROOT / "python/build_and_seal_v71_development_census.py"
    census_path = (
        PROJECT_ROOT / "outputs/v71-sensor-codebook/development-census.jsonl"
    )
    audit_path = PROJECT_ROOT / "outputs/v71-sensor-codebook/development-census-audit.json"
    seal_path = (
        PROJECT_ROOT
        / "configs/v71-sensor-codebook-development-census-seal.json"
    )
    if seal_path.exists():
        raise RuntimeError("V71 development census is already sealed")

    source_lock = json.loads(source_lock_path.read_text())
    resource_lock = json.loads(resource_lock_path.read_text())
    resource_payload = {
        key: value for key, value in resource_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(resource_payload) == resource_lock["lock_payload_sha256"]
        and resource_lock["source_lock_sha256"] == file_sha256(source_lock_path)
        and resource_lock["authorization"][
            "construct_and_seal_complete_development_census"
        ]
        and not resource_lock["authorization"]["write_development_evaluator"]
        and not resource_lock["authorization"]["run_development_outcomes"]
        and not resource_lock["authorization"][
            "read_protected_confirmation_histories_or_outcomes"
        ]
    )
    if not authorization_ok:
        errors.append("resource lock does not authorize census construction only")

    config = source_lock["config_payload"]
    development = config["prospectivePartition"]["developmentFresh"]
    protected = set(
        config["prospectivePartition"]["protectedConfirmationRelated"]
        + config["prospectivePartition"]["protectedConfirmationNovel"]
    )
    source_root = (
        PROJECT_ROOT
        / config["source"]["checkout"]
        / config["source"]["modelDirectory"]
    )
    reliability = float(config["unknownSensorFamily"]["reliability"])
    records: list[dict[str, Any]] = []
    per_model: dict[str, int] = {}
    for filename in development:
        parsed = parse_cassandra_pomdp_file(source_root / filename)
        prefixes = enumerate_public_prefixes(parsed, reliability=reliability)
        per_model[filename] = len(prefixes)
        for prefix in prefixes:
            action = (
                None
                if prefix.action_index is None
                else parsed.model.actions[prefix.action_index]
            )
            observation = (
                None
                if prefix.observation_index is None
                else parsed.model.observations[prefix.observation_index]
            )
            suffix = (
                "root"
                if prefix.depth == 0
                else f"a={action}::o={observation}"
            )
            records.append(
                {
                    "record_id": f"{filename}::{suffix}",
                    "model_file": filename,
                    "depth": prefix.depth,
                    "public_action": action,
                    "public_observation": observation,
                    "public_prefix_probability": prefix.probability,
                    "joint_belief_latent_by_state": prefix.joint_belief.tolist(),
                }
            )

    expected_bounds = {
        row["file"]: row["census_record_upper_bound"]
        for row in resource_lock["model_bounds"]
    }
    record_ids = [row["record_id"] for row in records]
    completeness_ok = bool(
        per_model == expected_bounds
        and len(records) == resource_lock["totals"]["total_census_record_upper_bound"]
        and len(record_ids) == len(set(record_ids))
        and all(row["model_file"] in development for row in records)
        and not any(row["model_file"] in protected for row in records)
        and all(row["depth"] in (0, 1) for row in records)
        and all(row["public_prefix_probability"] > 0.0 for row in records)
        and all(
            abs(
                sum(sum(latent) for latent in row["joint_belief_latent_by_state"])
                - 1.0
            )
            <= 1e-12
            for row in records
        )
    )
    if not completeness_ok:
        errors.append("development census is not the complete frozen positive-probability set")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/evaluate_v71_sensor_codebook_development.py",
            "configs/v71-sensor-codebook-development-evaluator-lock.json",
            "outputs/v71-sensor-codebook/development-evaluation",
            "outputs/v71-sensor-codebook/protected-confirmation-evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V71 evaluator or outcomes exist before census seal")

    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    checks = {
        "resource_lock_and_census_only_authorization": authorization_ok,
        "complete_21_record_three_model_positive_probability_census": completeness_ok,
        "zero_protected_confirmation_records": not any(
            row["model_file"] in protected for row in records
        ),
        "evaluator_and_outcomes_absent": downstream_absent,
        "zero_reward_policy_value_action_regret_or_EIG_access": True,
    }
    audit = {
        "schema_version": "71-sensor-codebook-development-census",
        "experiment": "v71_development_census_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "seal_complete_development_census_and_authorize_evaluator_implementation_only"
            if not errors
            else "reject_v71_development_census"
        ),
        "errors": errors,
        "checks": checks,
        "record_count": len(records),
        "per_model_record_counts": per_model,
        "access": {
            "development_histories_constructed": len(records),
            "protected_confirmation_histories_constructed": 0,
            "reward_arrays_read": 0,
            "policy_values_computed": 0,
            "optimal_actions_computed": 0,
            "regrets_computed": 0,
            "EIG_values_computed": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    seal = {
        "schema_version": "71-sensor-codebook-development-census",
        "experiment": "v71_sensor_codebook_development_census_seal",
        "source_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_lock_sha256": file_sha256(source_lock_path),
        "resource_lock": str(resource_lock_path.relative_to(PROJECT_ROOT)),
        "resource_lock_sha256": file_sha256(resource_lock_path),
        "belief_core": str(core_path.relative_to(PROJECT_ROOT)),
        "belief_core_sha256": file_sha256(core_path),
        "belief_core_tests": str(core_tests_path.relative_to(PROJECT_ROOT)),
        "belief_core_tests_sha256": file_sha256(core_tests_path),
        "census_builder": str(builder_path.relative_to(PROJECT_ROOT)),
        "census_builder_sha256": file_sha256(builder_path),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "census_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "census_audit_sha256": file_sha256(audit_path),
        "record_count": len(records),
        "per_model_record_counts": per_model,
        "authorization": {
            "modify_source_family_partition_resource_or_census": False,
            "write_and_audit_development_evaluator": True,
            "run_development_outcomes": False,
            "read_protected_confirmation_histories_or_outcomes": False,
            "select_filter_drop_or_replace_records_or_models": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"seal": str(seal_path), "sha256": file_sha256(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
