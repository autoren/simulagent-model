#!/usr/bin/env python3
"""Audit and freeze source-only V71 sensor-codebook feasibility."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v71_cassandra_pomdp import parse_cassandra_pomdp_file, source_validation


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_output(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def maximum_row_error(array: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(array).sum(axis=-1) - 1.0)))


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v71-sensor-codebook-source-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v71-sensor-codebook-source-feasibility-plan.md"
    parser_path = PROJECT_ROOT / "python/v71_cassandra_pomdp.py"
    tests_path = PROJECT_ROOT / "python/test_v71_cassandra_pomdp.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v71_source_feasibility.py"
    inventory_path = (
        PROJECT_ROOT / "outputs/v71-sensor-codebook/source-feasibility-inventory.json"
    )
    audit_path = PROJECT_ROOT / "outputs/v71-sensor-codebook/source-feasibility-audit.json"
    lock_path = PROJECT_ROOT / "configs/v71-sensor-codebook-source-lock.json"
    if lock_path.exists():
        raise RuntimeError("V71 source feasibility is already frozen")

    config = json.loads(config_path.read_text())
    source = config["source"]
    checkout = PROJECT_ROOT / source["checkout"]
    model_directory = checkout / source["modelDirectory"]
    errors: list[str] = []

    directional_path = PROJECT_ROOT / config["directionalSourceLock"]
    directional = json.loads(directional_path.read_text())
    directional_payload = {
        key: value for key, value in directional.items() if key != "lock_payload_sha256"
    }
    directional_ok = bool(
        payload_hash(directional_payload) == directional["lock_payload_sha256"]
        and directional["authorization"][
            "new_family_only_after_new_preregistration_and_fresh_models"
        ]
        and not directional["authorization"]["modify_or_rerun_V69_or_V70"]
        and not directional["authorization"]["use_V70_as_development_data"]
    )
    if not directional_ok:
        errors.append("locked V68r2-V70 synthesis does not authorize a new boundary study")

    try:
        source_commit = git_output(checkout, "rev-parse", "HEAD")
        source_remote = git_output(checkout, "remote", "get-url", "origin")
        source_status = git_output(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        source_commit = ""
        source_remote = ""
        source_status = "unavailable"
        errors.append(f"cannot inspect pinned source checkout: {exc}")
    source_ok = bool(
        source_commit == source["commit"]
        and source_remote.rstrip("/").removesuffix(".git")
        == source["repository"].rstrip("/").removesuffix(".git")
        and source_status == ""
        and file_sha256(checkout / source["licenseFile"])
        == source["licenseSha256"]
        and source["license"] == "CC-BY-NC-4.0"
    )
    if not source_ok:
        errors.append("source commit, remote, cleanliness, or license binding failed")

    configured_rows = config["sourceInventory"]
    configured = {row["file"]: row for row in configured_rows}
    actual_files = sorted(path.name for path in model_directory.glob("*.POMDP"))
    inventory_ok = bool(
        len(configured_rows) == len(configured) == len(actual_files) == 19
        and set(configured) == set(actual_files)
        and all(
            file_sha256(model_directory / filename) == configured[filename]["sha256"]
            for filename in actual_files
        )
    )
    if not inventory_ok:
        errors.append("19-file source inventory or byte hashes drifted")

    partition = config["prospectivePartition"]
    role_counts = Counter(row["role"] for row in configured_rows)
    partition_sets = {key: set(value) for key, value in partition.items()}
    selected_names = set().union(*partition_sets.values())
    partition_ok = bool(
        role_counts
        == {
            "developmentFresh": 3,
            "protectedConfirmationRelated": 2,
            "protectedConfirmationNovel": 3,
            "priorProjectExposureExclusion": 5,
            "sourceDefectExclusion": 6,
        }
        and all(
            partition_sets[role]
            == {row["file"] for row in configured_rows if row["role"] == role}
            for role in partition_sets
        )
        and len(selected_names) == 8
        and not any(
            partition_sets[left] & partition_sets[right]
            for index, left in enumerate(partition_sets)
            for right in list(partition_sets)[index + 1 :]
        )
    )
    if not partition_ok:
        errors.append("prospective development/protected partition is incomplete or overlapping")

    parsed_selected: dict[str, Any] = {}
    inventory_records: list[dict[str, Any]] = []
    selected_valid = True
    for filename in sorted(selected_names):
        parsed = parse_cassandra_pomdp_file(model_directory / filename)
        parsed_selected[filename] = parsed
        checks = source_validation(parsed)
        selected_valid = selected_valid and all(checks.values())
        model = parsed.model
        inventory_records.append(
            {
                "file": filename,
                "role": configured[filename]["role"],
                "sha256": configured[filename]["sha256"],
                "states": len(model.states),
                "actions": len(model.actions),
                "observations": len(model.observations),
                "discount": model.discount,
                "value_type": parsed.value_type,
                "observation_dependent_source_reward": (
                    parsed.reward_observation_dependent
                ),
                "maximum_transition_row_sum_error": maximum_row_error(
                    model.transition
                ),
                "maximum_observation_row_sum_error": maximum_row_error(
                    model.observation
                ),
                "failed_validation_checks": sorted(
                    key for key, passed in checks.items() if not passed
                ),
            }
        )
    if not selected_valid:
        errors.append("one or more selected sources failed strict source validation")

    source_defects_ok = True
    for filename in ("1d.POMDP", "aloha-10max.POMDP"):
        parsed = parse_cassandra_pomdp_file(model_directory / filename)
        source_defects_ok = source_defects_ok and not source_validation(parsed)[
            "transition_normalized"
        ]
    for filename in ("ejs4.POMDP", "ejs5.POMDP", "ejs6.POMDP", "ejs7.POMDP"):
        try:
            parse_cassandra_pomdp_file(model_directory / filename)
        except ValueError as exc:
            source_defects_ok = source_defects_ok and "discount" in str(exc)
        else:
            source_defects_ok = False
    source_defects_ok = bool(
        source_defects_ok
        and "0.4 0.7" in (model_directory / "ejs7.POMDP").read_text()
        and "Missing 'discount:' specification."
        in (checkout / "src/mdp/parse_err.h").read_text()
    )
    if not source_defects_ok:
        errors.append("source-defect exclusions are not reproduced without repair")

    prior_exposure_ok = bool(
        configured["4x3.95.POMDP"]["sha256"]
        == "0fa62301931960d682b02961ffd38f4dd6b8e8835bc0203f4a12f849c267d6ff"
        and all(
            configured[name]["reason"]
            for name in (
                "4x3.95.POMDP",
                "cheese.95.POMDP",
                "shuttle.95.POMDP",
                "tiger.95.POMDP",
                "tiger.aaai.POMDP",
            )
        )
    )
    if not prior_exposure_ok:
        errors.append("prior-exposure exclusions are not prospectively documented")

    rho = float(config["unknownSensorFamily"]["reliability"])
    support_records: list[dict[str, Any]] = []
    shared_support_ok = 0.5 < rho < 1.0
    for filename, parsed in sorted(parsed_selected.items()):
        source_observation = parsed.model.observation
        reversed_source = source_observation[..., ::-1]
        canonical = rho * source_observation + (1.0 - rho) * reversed_source
        reversed_dominant = rho * reversed_source + (1.0 - rho) * source_observation
        canonical_error = maximum_row_error(canonical)
        reversed_error = maximum_row_error(reversed_dominant)
        masks_equal = bool(np.array_equal(canonical > 0.0, reversed_dominant > 0.0))
        model_ok = bool(
            masks_equal
            and canonical_error <= 1e-12
            and reversed_error <= 1e-12
            and np.isfinite(canonical).all()
            and np.isfinite(reversed_dominant).all()
        )
        shared_support_ok = shared_support_ok and model_ok
        support_records.append(
            {
                "file": filename,
                "identical_point_support": masks_equal,
                "canonical_maximum_row_sum_error": canonical_error,
                "reversed_maximum_row_sum_error": reversed_error,
                "support_cells": int(np.count_nonzero(canonical > 0.0)),
            }
        )
    if not shared_support_ok:
        errors.append("sensor-codebook point models do not have identical valid support")

    design = config["prospectiveDevelopmentDesign"]
    gates = config["prospectiveDevelopmentGates"]
    design_ok = bool(
        config["unknownSensorFamily"]["latentPrior"] == [0.5, 0.5]
        and rho == 0.85
        and design["publicPrefixDepths"] == [0, 1]
        and design["horizonActions"] == 3
        and design["selectionOrRejection"] is False
        and design["pointModelFallback"].startswith("forbidden")
        and design["materialNormalizedRegret"] == 0.005
        and gates["minimumDevelopmentModels"] == 3
        and gates["minimumModelsWithExactBAMAPRootActionDisagreement"] == 3
        and gates["minimumModelsWithMaterialMAPRegret"] == 2
        and gates["minimumModelsWithMaterialPosteriorSamplingRegret"] == 1
        and gates["minimumMaximumNormalizedMAPRegret"] == 0.01
        and gates["maximumProtectedConfirmationPolicyValueCount"] == 0
        and config["decisionRule"]["noThresholdRevision"]
        and config["decisionRule"]["noModelReplacement"]
    )
    if not design_ok:
        errors.append("prospective development design or noncompensatory gates drifted")

    boundary = config["claimBoundary"]
    authorization = config["stageAuthorization"]
    boundary_ok = bool(
        boundary["prospectiveNewUncertaintyMechanism"]
        and not boundary["externalUncertaintyFamilyClaim"]
        and not boundary["approximateInference"]
        and not boundary["SMC2"]
        and not boundary["humanData"]
        and not boundary["modelAccess"]
        and not boundary["adapterTraining"]
        and not boundary["commercialUseAuthorizedBySourceLicense"]
        and authorization["auditAndFreezeSourceFeasibility"]
        and not authorization["auditAndFreezeDevelopmentResourceFeasibility"]
        and not authorization["constructDevelopmentCensus"]
        and not authorization["writeDevelopmentEvaluator"]
        and not authorization["runDevelopmentOutcomes"]
        and not authorization["readProtectedConfirmationOutcomes"]
    )
    if not boundary_ok:
        errors.append("claim boundary, source license boundary, or stage firewall failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v71-sensor-codebook-development-resource-lock.json",
            "configs/v71-sensor-codebook-development-census-seal.json",
            "python/evaluate_v71_sensor_codebook_development.py",
            "configs/v71-sensor-codebook-development-evaluator-lock.json",
            "outputs/v71-sensor-codebook/development-evaluation",
            "outputs/v71-sensor-codebook/protected-confirmation-evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V71 census, evaluator, or outcome artifacts predate source lock")

    inventory = {
        "schema_version": "71-sensor-codebook-source-feasibility",
        "experiment": "v71_pinned_selected_source_inventory",
        "source_commit": source_commit,
        "selected_model_count": len(inventory_records),
        "role_counts": dict(sorted(role_counts.items())),
        "selected_models": inventory_records,
        "sensor_support": support_records,
        "access": {
            "source_metadata_records_read": 19,
            "source_arrays_parsed": 10,
            "policy_values_computed": 0,
            "optimal_actions_computed": 0,
            "regrets_computed": 0,
            "EIG_values_computed": 0,
            "histories_constructed": 0,
            "protected_confirmation_outcomes_read": 0,
        },
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    checks = {
        "directional_synthesis_lock_and_new_boundary_authorization": directional_ok,
        "pinned_clean_source_commit_remote_and_license": source_ok,
        "complete_19_file_byte_inventory": inventory_ok,
        "prospective_nonoverlapping_3_development_5_confirmation_partition": partition_ok,
        "all_8_selected_sources_strictly_valid": selected_valid,
        "all_6_source_defect_exclusions_reproduced_without_repair": source_defects_ok,
        "all_5_prior_exposure_exclusions_documented": prior_exposure_ok,
        "identical_normalized_sensor_point_model_support": shared_support_ok,
        "prospective_exact_development_design_and_gates": design_ok,
        "claim_license_and_stage_firewalls": boundary_ok,
        "downstream_census_evaluator_and_outcomes_absent": downstream_absent,
        "zero_policy_value_action_regret_EIG_or_history_access": True,
    }
    audit = {
        "schema_version": "71-sensor-codebook-source-feasibility",
        "experiment": "v71_source_feasibility_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_source_family_partition_and_authorize_development_resource_preflight_only"
            if not errors
            else "reject_v71_source_feasibility"
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
        "schema_version": "71-sensor-codebook-source-feasibility",
        "experiment": "v71_sensor_codebook_source_lock",
        "directional_source_lock": str(directional_path.relative_to(PROJECT_ROOT)),
        "directional_source_lock_sha256": file_sha256(directional_path),
        "source_repository": source["repository"],
        "source_commit": source_commit,
        "source_license": source["license"],
        "source_license_sha256": source["licenseSha256"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "parser": str(parser_path.relative_to(PROJECT_ROOT)),
        "parser_sha256": file_sha256(parser_path),
        "parser_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "parser_tests_sha256": file_sha256(tests_path),
        "source_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "source_auditor_sha256": file_sha256(auditor_path),
        "source_inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "source_inventory_sha256": file_sha256(inventory_path),
        "source_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "source_audit_sha256": file_sha256(audit_path),
        "prospective_role_census": dict(sorted(role_counts.items())),
        "authorization": {
            "modify_source_family_partition_or_development_gates": False,
            "audit_and_freeze_development_resource_feasibility": True,
            "construct_development_census": False,
            "write_development_evaluator": False,
            "run_development_outcomes": False,
            "read_protected_confirmation_outcomes": False,
            "drop_repair_renormalize_or_replace_models": False,
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
