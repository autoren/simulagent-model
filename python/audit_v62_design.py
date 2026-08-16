#!/usr/bin/env python3
"""Audit the V62 external classic-POMDP preregistration."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def checkout_source(repository: str, commit: str, destination: Path) -> None:
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", repository],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "-q", "--depth", "1", "origin", commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "-q", "--detach", "FETCH_HEAD"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v62-external-classic-pomdp-transfer.json"
    )
    parser.add_argument(
        "--plan", default="docs/v62-external-classic-pomdp-transfer-plan.md"
    )
    parser.add_argument(
        "--requirements", default="configs/v62-pobax-runtime-requirements.txt"
    )
    parser.add_argument(
        "--output", default="outputs/v62-external-pomdp-transfer/design-audit.json"
    )
    args = parser.parse_args()
    config_path, plan_path, requirements_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.requirements, args.output)
    )
    config = json.loads(config_path.read_text())
    errors: list[str] = []

    source_lock_path = PROJECT_ROOT / config["sourceV61OutcomeLock"]
    source_lock = json.loads(source_lock_path.read_text())
    source_ok = (
        source_lock["qualification_passed"]
        and source_lock["authorization"]["continue_to_next_preregistered_stage"]
        and not source_lock["authorization"]["treat_synthetic_v58_as_human"]
    )
    if not source_ok:
        errors.append("V62 requires the passing frozen V61 outcome")

    external = config["externalSource"]
    try:
        with tempfile.TemporaryDirectory(prefix="v62-source-audit-") as temp_dir:
            checkout = Path(temp_dir) / "pobax"
            checkout_source(external["repository"], external["commit"], checkout)
            resolved_commit = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            observed_hashes = {
                path: file_sha256(checkout / path) for path in external["files"]
            }
            source_binding_ok = (
                resolved_commit == external["commit"]
                and observed_hashes == external["files"]
                and file_sha256(checkout / "LICENSE") == external["licenseSha256"]
                and external["license"] == "Apache-2.0"
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        source_binding_ok = False
        observed_hashes = {}
        errors.append(f"could not retrieve pinned POBAX source: {exc}")
    if not source_binding_ok:
        errors.append("pinned POBAX commit, files, or license do not match")

    runtime = config["runtime"]
    expected_requirements = [
        f"{name}=={version}" for name, version in runtime["packages"].items()
    ]
    runtime_ok = (
        requirements_path.read_text().splitlines() == expected_requirements
        and runtime["python"] == "3.12.12"
        and runtime["jaxDoublePrecision"]
        and runtime["platform"] == "cpu"
    )
    if not runtime_ok:
        errors.append("isolated external runtime is not completely pinned")

    benchmark = config["benchmark"]
    benchmark_ok = (
        benchmark["taskCells"] == 6
        and benchmark["models"]
        == [
            {
                "id": "tiger-alt-start",
                "category": "noisy_sensor_and_information_gathering",
                "horizons": [1, 3, 5, 7],
            },
            {"id": "tmaze2", "category": "delayed_memory", "horizons": [4]},
            {
                "id": "tmaze5",
                "category": "longer_delayed_memory",
                "horizons": [7],
            },
        ]
        and benchmark["tieTolerance"] <= 1e-12
        and benchmark["tieBreak"].startswith("lowest_external_action_index")
    )
    if not benchmark_ok:
        errors.append("external benchmark cells or decision semantics changed")

    independent = config["independentReference"]
    audit = config["implementationAudit"]
    controls_ok = (
        len(independent["analyticFixtures"]) == 6
        and len(audit["mutants"]) == 12
        and audit["requiredMutantKillRate"] == 1.0
        and audit["requiredAnalyticFixturePassRate"] == 1.0
        and set(config["controls"])
        == {
            "observationOnly",
            "mapCollapse",
            "fullyObservedOracle",
            "uniformRandom",
            "interpretation",
        }
    )
    if not controls_ok:
        errors.append("independent references, fixtures, mutants, or controls are incomplete")

    rollout = config["externalRollout"]
    rollout_ok = (
        rollout["episodesPerTaskPolicy"] == 4096
        and len(rollout["policies"]) == 4
        and rollout["comparisons"] == 24
        and rollout["familywiseAlpha"] == 0.01
        and not rollout["candidateTruthAccess"]
    )
    if not rollout_ok:
        errors.append("external rollout census or simultaneous bound is not frozen")

    gates = config["gates"]
    gates_ok = (
        len(config["metrics"]) == 32
        and len(gates) == 32
        and gates["minimumCompletedTaskFraction"] == 1.0
        and gates["minimumExternalSourceBindingRate"] == 1.0
        and gates["minimumIndependentParserAgreementRate"] == 1.0
        and gates["maximumCandidateReferenceValueError"] <= 1e-10
        and gates["minimumCandidateReferenceOptimalSetMembershipRate"] == 1.0
        and gates["minimumOfficialRuntimeReturnWithinSimultaneousBoundRate"] == 1.0
        and gates["minimumTmazeExactHistoryMinusObservationOnlyValue"] == 0.9
        and gates["minimumTigerExactHistoryMinusMapCollapseValue"] == 1.0
        and gates["minimumImplementationMutantKillRate"] == 1.0
        and gates["maximumUnexpectedEvaluationAttemptCount"] == 0
        and gates["maximumHumanRecordAccessCount"] == 0
        and gates["maximumModelForwardPassCount"] == 0
    )
    if not gates_ok:
        errors.append("V62 must retain all 32 noncompensatory gates")

    boundary = config["claimBoundary"]
    firewall = config["firewall"]
    boundary_ok = (
        boundary["externalBenchmark"]
        and boundary["exactFiniteStateBelief"]
        and boundary["externalRuntimeExecution"]
        and not boundary["unknownProgramOrParameterInference"]
        and not boundary["smc2Portability"]
        and not boundary["formalSafety"]
        and not boundary["humanAuthoredLanguage"]
        and all(value == "forbidden" for value in firewall.values())
    )
    if not boundary_ok:
        errors.append("V62 claim boundary or firewall is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v62-design-lock.json",
            "configs/v62-implementation-lock.json",
            "configs/v62-external-bundle-seal.json",
            "configs/v62-evaluation-implementation-lock.json",
            "configs/v62-outcome-lock.json",
            "outputs/v62-external-pomdp-transfer/evaluation-attempt.json",
            "outputs/v62-external-pomdp-transfer/evaluation/result.json",
            "docs/v62-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V62 downstream artifacts already exist")

    result = {
        "schema_version": 62,
        "experiment": "v62_design_audit",
        "passed": not errors,
        "decision": "freeze_v62_design" if not errors else "repair_v62_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "runtime_requirements": str(requirements_path.relative_to(PROJECT_ROOT)),
        "runtime_requirements_sha256": file_sha256(requirements_path),
        "source_v61_outcome_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_v61_outcome_lock_sha256": file_sha256(source_lock_path),
        "external_commit": external["commit"],
        "external_files_sha256": observed_hashes,
        "external_license_sha256": external["licenseSha256"],
        "checks": {
            "passing_frozen_v61_source": source_ok,
            "pinned_external_source_and_license": source_binding_ok,
            "isolated_pinned_runtime": runtime_ok,
            "six_prospectively_fixed_task_cells": benchmark_ok,
            "independent_references_fixtures_mutants_and_controls": controls_ok,
            "frozen_external_rollout_and_simultaneous_bound": rollout_ok,
            "thirty_two_noncompensatory_gates": gates_ok,
            "claim_boundary_and_firewall": boundary_ok,
            "downstream_absence": downstream_absent,
        },
        "data_access": {
            "external_model_definition_files_read": 3,
            "external_candidate_evaluations": 0,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
