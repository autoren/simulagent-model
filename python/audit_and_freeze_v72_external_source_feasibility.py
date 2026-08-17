#!/usr/bin/env python3
"""Audit and freeze metadata-only V72 external active-sensing discovery."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def main() -> None:
    config_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-external-source-feasibility.json"
    )
    plan_path = (
        PROJECT_ROOT / "docs/v72-active-sensing-external-source-feasibility-plan.md"
    )
    oracle_lock_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-oracle-outcome-lock.json"
    )
    auditor_path = (
        PROJECT_ROOT / "python/audit_and_freeze_v72_external_source_feasibility.py"
    )
    inventory_path = (
        PROJECT_ROOT / "outputs/v72-active-sensing/external-source-inventory.json"
    )
    audit_path = PROJECT_ROOT / "outputs/v72-active-sensing/external-source-audit.json"
    lock_path = PROJECT_ROOT / "configs/v72-active-sensing-external-source-lock.json"
    if lock_path.exists():
        raise RuntimeError("V72 external source inventory is already frozen")

    config = json.loads(config_path.read_text())
    oracle_lock = json.loads(oracle_lock_path.read_text())
    oracle_payload = {
        key: value for key, value in oracle_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(oracle_payload) == oracle_lock["lock_payload_sha256"]
        and oracle_lock["outcome"]["passed_all_oracle_gates"]
        and not oracle_lock["outcome"]["scientific_evidence"]
        and oracle_lock["authorization"]["inspect_fresh_external_candidate_metadata"]
        and oracle_lock["authorization"][
            "freeze_external_source_inventory_and_structural_partition"
        ]
        and not oracle_lock["authorization"][
            "compute_external_candidate_policy_values_actions_regrets_or_EIG"
        ]
    )
    if not authorization_ok:
        errors.append("V72 oracle outcome does not authorize metadata-only discovery")

    checkout_root = PROJECT_ROOT / "data/v72-active-sensing/source-checkouts"
    repositories = {
        "sarsop": {
            "url": "https://github.com/AdaCompNUS/sarsop",
            "commit": "d9141104392fd0a7b35327fdf7d40ef4b71a13ca",
            "license": "mixed GPL-2.0/Apache-2.0/zlib as declared by license/License",
        },
        "SBO_AIPPMS": {
            "url": "https://github.com/sisl/SBO_AIPPMS",
            "commit": "24db0a618a98f7c0c9bed470d25cc68fe6dc9b50",
            "license": "MIT",
        },
        "BetaZero.jl": {
            "url": "https://github.com/sisl/BetaZero.jl",
            "commit": "386306ae7011cb35a7e3215daaaf7b88dc5b96d8",
            "license": "unresolved at repository root",
        },
        "POMDPModels.jl": {
            "url": "https://github.com/JuliaPOMDP/POMDPModels.jl",
            "commit": "d4edb2fb25ee880bf4d9326de17ad8dc38a88fdc",
            "license": "MIT Expat",
        },
        "RockSample.jl": {
            "url": "https://github.com/JuliaPOMDP/RockSample.jl",
            "commit": "c8b3566d30c5dd7be6c7790b4b9a54ebfcdeecde",
            "license": "MIT Expat",
        },
    }
    commit_ok = all(
        git_commit(checkout_root / name) == metadata["commit"]
        for name, metadata in repositories.items()
    )
    if not commit_ok:
        errors.append("One or more V72 source commits drifted")

    rock_root = checkout_root / "RockSample.jl"
    rock_actions = (rock_root / "src/actions.jl").read_text()
    rock_observations = (rock_root / "src/observations.jl").read_text()
    rock_rewards = (rock_root / "src/reward.jl").read_text()
    rock_states = (rock_root / "src/states.jl").read_text()
    rock_structure_ok = bool(
        "const N_BASIC_ACTIONS = 5" in rock_actions
        and "N_BASIC_ACTIONS+K" in rock_actions
        and "if a <= N_BASIC_ACTIONS" in rock_observations
        and "efficiency = 0.5*(1.0 + exp(" in rock_observations
        and "pomdp.good_rock_reward" in rock_rewards
        and "pomdp.bad_rock_penalty" in rock_rewards
        and "pomdp.exit_reward" in rock_rewards
        and "map_size[1]*pomdp.map_size[2]*2^length" in rock_states
        and (rock_root / "LICENSE.md").exists()
    )
    if not rock_structure_ok:
        errors.append("RockSample.jl no longer exposes the fixed finite active-sensing interface")

    sarsop_root = checkout_root / "sarsop"
    rocksample_path = sarsop_root / "examples/POMDP/RockSample_7_8.pomdp"
    with rocksample_path.open() as handle:
        rocksample_prefix = "".join(handle.readline() for _ in range(16))
    sarsop_structure_ok = bool(
        "ac0, ac1" in rocksample_prefix
        and "as: sample" in rocksample_prefix
        and "actions: 13" in rocksample_prefix
        and "observations: 2" in rocksample_prefix
        and "states: 12545" in rocksample_prefix
        and "rest of APPL is released under GNU General Public License V2"
        in (sarsop_root / "license/License").read_text()
    )
    if not sarsop_structure_ok:
        errors.append("SARSOP RockSample metadata or license declaration drifted")

    sbo_root = checkout_root / "SBO_AIPPMS"
    sbo_readme = (sbo_root / "README.md").read_text()
    isrs_source = (
        sbo_root
        / "InformationRockSample/AIPPMS/src/envs/InfSearchRockSample.jl"
    ).read_text()
    rover_observation = (sbo_root / "Rover/POMDP_Rover/observations.jl").read_text()
    sbo_structure_ok = bool(
        "multiple sensors, each with different sensing accuracy and energy costs" in sbo_readme
        and "ONLY ALLOW SENSING IF AT BEACON" in isrs_source
        and "sample_location_states" in isrs_source
        and "rand(rng)" in isrs_source
        and "Normal(0, pomdp.σ_spec)" in rover_observation
        and "MIT License" in (sbo_root / "LICENSE").read_text()
    )
    if not sbo_structure_ok:
        errors.append("SBO_AIPPMS structural or simulator metadata drifted")

    models_root = checkout_root / "POMDPModels.jl"
    mini = (models_root / "src/MiniHallway.jl").read_text()
    models_structure_ok = bool(
        "POMDPs.actions(m::MiniHallway) = 1:3" in mini
        and "function POMDPs.observation" in mini
        and "MIT \"Expat\" License" in (models_root / "LICENSE.md").read_text()
    )
    beta_root = checkout_root / "BetaZero.jl"
    beta_license_unresolved = not any(
        path.is_file() for path in beta_root.glob("[Ll][Ii][Cc][Ee][Nn][Ss][Ee]*")
    )
    if not models_structure_ok or not beta_license_unresolved:
        errors.append("POMDPModels or BetaZero classification metadata drifted")

    blueprint = config["selectedDevelopmentBlueprint"]
    blueprint_ok = bool(
        blueprint["repository"] == "https://github.com/JuliaPOMDP/RockSample.jl"
        and blueprint["repositoryRole"] == "development-only"
        and blueprint["mapSize"] == [2, 2]
        and blueprint["initialPosition"] == [2, 1]
        and blueprint["rocksInOrder"][0]["position"] == [1, 1]
        and blueprint["rocksInOrder"][1]["position"] == [2, 2]
        and blueprint["sensorCodebookWrapper"][
            "observationNoiseFloorMixtureWeight"
        ]
        == 0.2
        and blueprint["planningHorizonActions"] == 4
        and blueprint["expectedFiniteDimensionsBeforeLatentAugmentation"]
        == {"states": 17, "actions": 7, "observations": 3}
        and blueprint["sourceParameters"]["goodRockReward"]
        > blueprint["sourceParameters"]["exitReward"]
        > blueprint["sourceParameters"]["badRockPenalty"]
    )
    if not blueprint_ok:
        errors.append("V72 metadata-selected development blueprint drifted")

    candidate_records = [
        {
            "id": "sbo_information_rocksample",
            "repository": "SBO_AIPPMS",
            "status": "excluded",
            "reason_codes": ["no_finite_exact_representation", "development_exposed"],
            "structure": {
                "distinct_sensing_and_control_actions": True,
                "action_dependent_observation": True,
                "delayed_state_dependent_control_reward": True,
                "simulator_or_RNG_defined_observation_path": True,
            },
        },
        {
            "id": "sbo_rover",
            "repository": "SBO_AIPPMS",
            "status": "excluded",
            "reason_codes": ["no_finite_exact_representation", "development_exposed"],
            "structure": {
                "continuous_gaussian_observations": True,
                "drill_and_spectrometer_modalities": True,
                "delayed_state_dependent_control_reward": True,
            },
        },
        {
            "id": "sarsop_rocksample_7_8",
            "repository": "sarsop",
            "status": "resource_deferred",
            "reason_codes": ["eligible_active_sensing_structure", "resource_deferred", "development_exposed"],
            "structure": {
                "states": 12545,
                "actions": 13,
                "observations": 2,
                "finite_exact_model_file": True,
                "distinct_check_and_sample_actions": True,
            },
        },
        {
            "id": "betazero_examples",
            "repository": "BetaZero.jl",
            "status": "excluded",
            "reason_codes": ["license_unresolved", "development_exposed"],
            "structure": {"solver_repository_with_example_submodules": True},
        },
        {
            "id": "pomdpmodels_minihallway",
            "repository": "POMDPModels.jl",
            "status": "excluded",
            "reason_codes": ["no_distinct_sensing_action", "development_exposed"],
            "structure": {"states": 13, "actions": 3, "observations": 9},
        },
        {
            "id": "rocksample_jl_configurable_small",
            "repository": "RockSample.jl",
            "status": "selected_development",
            "reason_codes": ["eligible_active_sensing_structure", "development_exposed"],
            "structure": {
                "finite_deterministic_enumerator": True,
                "configurable_map_and_rock_count": True,
                "distinct_check_sample_move_and_exit_actions": True,
                "state_and_distance_dependent_check_observations": True,
                "state_dependent_sample_reward": True,
            },
            "blueprint": blueprint,
        },
    ]

    source_files = [
        "SBO_AIPPMS/LICENSE",
        "SBO_AIPPMS/README.md",
        "SBO_AIPPMS/InformationRockSample/AIPPMS/src/envs/InfSearchRockSample.jl",
        "SBO_AIPPMS/Rover/POMDP_Rover/observations.jl",
        "sarsop/license/License",
        "sarsop/examples/POMDP/RockSample_7_8.pomdp",
        "POMDPModels.jl/LICENSE.md",
        "POMDPModels.jl/src/MiniHallway.jl",
        "BetaZero.jl/README.md",
        "BetaZero.jl/Project.toml",
        "RockSample.jl/LICENSE.md",
        "RockSample.jl/src/RockSample.jl",
        "RockSample.jl/src/actions.jl",
        "RockSample.jl/src/observations.jl",
        "RockSample.jl/src/reward.jl",
        "RockSample.jl/src/states.jl",
        "RockSample.jl/src/transition.jl",
    ]
    inventory = {
        "schema_version": "72-active-sensing-external-source-feasibility",
        "experiment": "v72_metadata_only_external_active_sensing_inventory",
        "selection_basis": "structural source metadata only",
        "repositories": repositories,
        "candidate_records": candidate_records,
        "selected_development_candidate": "rocksample_jl_configurable_small",
        "protected_confirmation_candidates": [],
        "source_file_sha256": {
            path: file_sha256(checkout_root / path) for path in source_files
        },
        "access": {
            "repository_landing_pages_previewed_before_inventory": 3,
            "pinned_repositories_inspected": 5,
            "candidate_metadata_records": len(candidate_records),
            "candidate_model_files_or_source_families_inspected": len(candidate_records),
            "candidate_simulator_runs": 0,
            "candidate_policy_values_computed": 0,
            "candidate_optimal_actions_computed": 0,
            "candidate_regrets_computed": 0,
            "candidate_EIG_values_computed": 0,
            "V71_protected_access_count": 0,
            "human_record_access_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
    }
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    firewall = config["firewall"]
    firewall_ok = bool(
        not firewall["computeCandidatePolicyValues"]
        and not firewall["computeCandidateOptimalActions"]
        and not firewall["computeCandidateRegrets"]
        and not firewall["computeCandidateEIG"]
        and not firewall["runCandidateSimulator"]
        and inventory["access"]["candidate_simulator_runs"] == 0
        and inventory["access"]["candidate_policy_values_computed"] == 0
        and inventory["access"]["candidate_optimal_actions_computed"] == 0
        and inventory["access"]["candidate_regrets_computed"] == 0
        and inventory["access"]["candidate_EIG_values_computed"] == 0
        and inventory["access"]["V71_protected_access_count"] == 0
    )
    if not firewall_ok:
        errors.append("V72 source inventory violated the outcome firewall")

    checks = {
        "oracle_authorizes_metadata_only_discovery": authorization_ok,
        "five_repositories_pinned": commit_ok,
        "RockSample_jl_finite_active_sensing_structure": rock_structure_ok,
        "SARSOP_large_exact_RockSample_classification": sarsop_structure_ok,
        "SBO_AIPPMS_simulator_and_continuous_classification": sbo_structure_ok,
        "POMDPModels_and_BetaZero_exclusions": models_structure_ok
        and beta_license_unresolved,
        "fixed_small_development_blueprint": blueprint_ok,
        "zero_candidate_outcome_access": firewall_ok,
    }
    audit = {
        "schema_version": "72-active-sensing-external-source-feasibility",
        "experiment": "v72_external_source_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_metadata_inventory_and_authorize_exporter_implementation_only"
            if not errors
            else "reject_external_source_inventory"
        ),
        "errors": errors,
        "checks": checks,
        "access": inventory["access"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "72-active-sensing-external-source-feasibility",
        "experiment": "v72_external_active_sensing_source_lock",
        "oracle_outcome_lock": str(oracle_lock_path.relative_to(PROJECT_ROOT)),
        "oracle_outcome_lock_sha256": file_sha256(oracle_lock_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "repository_commits": {
            metadata["url"]: metadata["commit"] for metadata in repositories.values()
        },
        "selected_development_candidate": "rocksample_jl_configurable_small",
        "selected_development_blueprint": blueprint,
        "protected_confirmation_candidate_count": 0,
        "authorization": {
            "modify_or_rerun_V71_or_V72_oracle": False,
            "modify_source_inventory_selection_or_blueprint": False,
            "implement_and_test_deterministic_source_exporter": True,
            "run_structural_resource_census": True,
            "compute_candidate_policy_values_actions_regrets_or_EIG": False,
            "select_protected_confirmation_models": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
