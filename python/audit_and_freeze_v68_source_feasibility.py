#!/usr/bin/env python3
"""Audit and freeze the V68 pinned-source inventory and prospective split."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from test_v68_cassandra_pomdp import load_pobax_reference_parser
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import terminal_mask, validate_model
from v68_cassandra_pomdp import parse_cassandra_pomdp_file


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def syntax_kind(path: Path, prefix: str) -> str:
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith(prefix):
            lines.append(line)
    if any(
        len(line.split(":")) > 2
        or re.search(r"(?i)\b(identity|uniform)\b", line)
        for line in lines
    ):
        return "sparse_or_keyword"
    # Keywords appear on the line following a two-field directive.
    cleaned = [
        raw.split("#", 1)[0].strip()
        for raw in path.read_text().splitlines()
        if raw.split("#", 1)[0].strip()
    ]
    for index, line in enumerate(cleaned[:-1]):
        if line.startswith(prefix) and cleaned[index + 1] in {"identity", "uniform"}:
            return "sparse_or_keyword"
    return "full_matrix"


def role_map(config: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for role, files in config["prospectiveRoles"].items():
        if isinstance(files, list):
            for name in files:
                if name in roles:
                    raise ValueError(f"duplicate prospective role for {name}")
                roles[name] = role
    return roles


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v68-multi-environment-source-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v68-multi-environment-source-feasibility-plan.md"
    parser_path = PROJECT_ROOT / "python/v68_cassandra_pomdp.py"
    tests_path = PROJECT_ROOT / "python/test_v68_cassandra_pomdp.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v68_source_feasibility.py"
    output_dir = PROJECT_ROOT / "outputs/v68-multi-environment-source-feasibility"
    inventory_path = output_dir / "source-inventory.json"
    audit_path = output_dir / "source-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68-source-feasibility-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68 source feasibility is already frozen")

    config = json.loads(config_path.read_text())
    errors: list[str] = []
    source_lock_path = PROJECT_ROOT / config["sourceV67OutcomeLock"]
    source_lock = json.loads(source_lock_path.read_text())
    source_payload = {
        key: value for key, value in source_lock.items() if key != "lock_payload_sha256"
    }
    v67_ok = bool(
        payload_hash(source_payload) == source_lock["lock_payload_sha256"]
        and source_lock["decision"]
        == "authorize_preregistration_of_multi_environment_external_replication_only"
        and source_lock["authorization"]["preregister_multi_environment_external_replication"]
        and not source_lock["authorization"]["run_replication_before_preregistration_and_locks"]
    )
    if not v67_ok:
        errors.append("V67 outcome authorization or binding failed")

    checkout = PROJECT_ROOT / config["externalSource"]["checkout"]
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    expected_commit = config["externalSource"]["commit"]
    model_dir = PROJECT_ROOT / config["externalSource"]["modelDirectory"]
    source_paths = sorted(model_dir.glob("*.POMDP"))
    expected_hashes = config["externalSource"]["expectedFilesSha256"]
    source_hashes = {path.name: file_sha256(path) for path in source_paths}
    source_binding_ok = bool(
        commit == expected_commit
        and len(source_paths) == config["externalSource"]["expectedModelCount"]
        and source_hashes == expected_hashes
        and file_sha256(checkout / "LICENSE")
    )
    if not source_binding_ok:
        errors.append("pinned POBAX source commit, file census, or hashes differ")

    test_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v68_cassandra_pomdp.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tests_ok = test_run.returncode == 0 and "Ran 5 tests" in (test_run.stdout + test_run.stderr)
    if not tests_ok:
        errors.append("V68 parser tests failed")

    reference_parser = load_pobax_reference_parser()
    roles = role_map(config)
    inventory: list[dict[str, Any]] = []
    parity_passes = 0
    for source in source_paths:
        model = parse_cassandra_pomdp_file(source)
        reference = reference_parser(source)
        parity = bool(
            model.states == tuple(reference.states)
            and model.actions == tuple(reference.actions)
            and model.observations == tuple(reference.observations)
            and model.discount == reference.discount
            and np.array_equal(model.initial, reference.start)
            and np.array_equal(model.transition, reference.T)
            and np.array_equal(model.observation, reference.Z)
            and np.array_equal(model.reward, reference.R)
        )
        parity_passes += int(parity)
        validation = validate_model(model)
        failed_checks = sorted(key for key, passed in validation.items() if not passed)
        role = roles.get(source.name, "unassigned")
        inventory.append(
            {
                "file": source.name,
                "sha256": source_hashes[source.name],
                "role": role,
                "prior_outcome_exposure": role == "developmentPreviouslyExposed",
                "source_valid": not failed_checks,
                "failed_validation_checks": failed_checks,
                "parser_reference_parity": parity,
                "states": len(model.states),
                "actions": len(model.actions),
                "observations": len(model.observations),
                "action_labels": list(model.actions),
                "discount": model.discount,
                "transition_syntax": syntax_kind(source, "T:"),
                "observation_syntax": syntax_kind(source, "O:"),
                "absorbing_state_count": int(terminal_mask(model).sum()),
                "maximum_transition_row_sum_error": float(
                    np.max(np.abs(model.transition.sum(axis=2) - 1.0))
                ),
                "maximum_observation_row_sum_error": float(
                    np.max(np.abs(model.observation.sum(axis=2) - 1.0))
                ),
                "initial_sum_error": abs(float(model.initial.sum()) - 1.0),
                "reward_nonzero_count": int(np.count_nonzero(model.reward)),
                "reward_minimum": float(model.reward.min()),
                "reward_maximum": float(model.reward.max()),
            }
        )

    parser_parity_ok = parity_passes == len(source_paths) == 14
    if not parser_parity_ok:
        errors.append("candidate parser does not exactly match the pinned POBAX parser")
    failed = {
        row["file"]: row["failed_validation_checks"]
        for row in inventory
        if row["failed_validation_checks"]
    }
    source_defect_ok = failed == {"paint.POMDP": ["observation_normalized"]}
    if not source_defect_ok:
        errors.append("source validation failures differ from the preregistered paint defect")

    role_census = {
        role: sum(row["role"] == role for row in inventory)
        for role in (
            "developmentPreviouslyExposed",
            "confirmatoryStructurallyRelatedButOutcomeUntouched",
            "confirmatoryNovelAndOutcomeUntouched",
            "sourceDefectExclusion",
        )
    }
    role_ok = bool(
        set(roles) == set(expected_hashes)
        and all(row["role"] != "unassigned" for row in inventory)
        and role_census
        == {
            "developmentPreviouslyExposed": 4,
            "confirmatoryStructurallyRelatedButOutcomeUntouched": 3,
            "confirmatoryNovelAndOutcomeUntouched": 6,
            "sourceDefectExclusion": 1,
        }
        and all(
            row["source_valid"]
            for row in inventory
            if row["role"] != "sourceDefectExclusion"
        )
    )
    if not role_ok:
        errors.append("prospective role census, coverage, or validity failed")

    cheese = {
        row["file"]: row
        for row in inventory
        if row["file"] in config["replicationTiersToDesignAfterThisAudit"]["tierB"]["frozenPair"]
    }
    tier_b_ok = bool(
        len(cheese) == 2
        and all(row["source_valid"] for row in cheese.values())
        and cheese["cheese.95.POMDP"]["states"]
        == cheese["cheese.95_nonterminating.POMDP"]["states"]
        and cheese["cheese.95.POMDP"]["actions"]
        == cheese["cheese.95_nonterminating.POMDP"]["actions"]
        and cheese["cheese.95.POMDP"]["observations"]
        == cheese["cheese.95_nonterminating.POMDP"]["observations"]
        and cheese["cheese.95.POMDP"]["absorbing_state_count"] == 1
        and cheese["cheese.95_nonterminating.POMDP"]["absorbing_state_count"] == 0
    )
    if not tier_b_ok:
        errors.append("source-native cheese pair structural qualification failed")

    firewall = config["firewall"]
    stage = config["stageAuthorization"]
    boundary = config["claimBoundary"]
    boundary_ok = bool(
        set(firewall.values()) == {"forbidden"}
        and stage["auditAndFreezeSourceFeasibility"]
        and not any(value for key, value in stage.items() if key != "auditAndFreezeSourceFeasibility")
        and boundary["pinnedExternalSourceInventory"]
        and boundary["independentParserParity"]
        and not any(
            boundary[key]
            for key in (
                "policyEvaluation",
                "expectedInformationGainEvaluation",
                "rewardPlanningEvaluation",
                "SMC2Evaluation",
                "multiEnvironmentReplicationResult",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
    )
    if not boundary_ok:
        errors.append("source-only authorization, claim boundary, or firewall is invalid")

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_payload = {
        "schema_version": "68-source-feasibility",
        "experiment": "v68_pinned_source_inventory",
        "repository": config["externalSource"]["repository"],
        "commit": commit,
        "model_count": len(inventory),
        "role_census": role_census,
        "parser_reference_parity_rate": parity_passes / len(inventory),
        "valid_model_count": sum(row["source_valid"] for row in inventory),
        "excluded_source_defect_count": sum(not row["source_valid"] for row in inventory),
        "policy_value_eig_regret_smc2_results_computed": 0,
        "models": inventory,
    }
    inventory_path.write_text(json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n")

    checks = {
        "V67_replication_preregistration_authorization": v67_ok,
        "pinned_source_commit_census_and_hashes": source_binding_ok,
        "five_parser_tests": tests_ok,
        "fourteen_model_exact_POBAX_parser_parity": parser_parity_ok,
        "paint_only_preregistered_source_defect": source_defect_ok,
        "complete_provenance_based_role_census": role_ok,
        "external_cheese_pair_structural_qualification": tier_b_ok,
        "source_only_claim_boundary_and_firewall": boundary_ok,
    }
    audit = {
        "schema_version": "68-source-feasibility",
        "experiment": "v68_source_feasibility_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_source_inventory_and_authorize_development_only_exact_infrastructure"
            if not errors
            else "reject_v68_source_feasibility"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "source_models_parsed_and_structurally_validated": 14,
            "reference_parser_array_crosschecks": 14,
            "new_policy_values_computed": 0,
            "new_EIG_values_computed": 0,
            "new_reward_planning_values_computed": 0,
            "new_SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "68-source-feasibility",
        "experiment": "v68_source_feasibility_lock",
        "source_v67_outcome_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_v67_outcome_lock_sha256": file_sha256(source_lock_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "parser": str(parser_path.relative_to(PROJECT_ROOT)),
        "parser_sha256": file_sha256(parser_path),
        "parser_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "parser_tests_sha256": file_sha256(tests_path),
        "source_inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "source_inventory_sha256": file_sha256(inventory_path),
        "source_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "source_audit_sha256": file_sha256(audit_path),
        "source_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "source_auditor_sha256": file_sha256(auditor_path),
        "prospective_role_census": role_census,
        "valid_source_models": 13,
        "source_defect_exclusions": ["paint.POMDP"],
        "external_uncertainty_pair": [
            "cheese.95.POMDP",
            "cheese.95_nonterminating.POMDP",
        ],
        "authorization": {
            "modify_or_rerun_v62_through_v67": False,
            "modify_v68_source_inventory_or_roles": False,
            "write_and_audit_development_only_exact_infrastructure": True,
            "run_development_only_outcome_screening": False,
            "run_confirmatory_policy_EIG_regret_or_SMC2_outcomes": False,
            "repair_or_normalize_paint_source": False,
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
