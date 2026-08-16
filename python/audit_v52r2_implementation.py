#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from decimal import Decimal, localcontext

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v51_sbc import distribution_tv, sequence_tv
from v52_particle import mechanic_registry, particle_inference as base_inference
from v52r2_particle import particle_inference as repaired_inference


REQUIRED = (
    "python/v52r2_particle.py",
    "python/evaluate_v52r2_particle.py",
    "python/test_v52r2_repair.py",
    "python/audit_and_summarize_v52r2.py",
    "python/freeze_v52r2_outcome.py",
    "scripts/run-v52r2-joint-normalization-repair.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-lock", default="configs/v52r2-repair-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v52r2-joint-normalization-repair/implementation-audit.json",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.repair_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    errors = []
    source_bound = (
        file_sha256(PROJECT_ROOT / lock["source_outcome_lock"])
        == lock["source_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / lock["source_implementation_lock"])
        == lock["source_implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / lock["source_population_seal"])
        == lock["source_population_seal_sha256"]
    )
    if not source_bound or not lock["authorization"]["write_and_audit_repair_implementation"]:
        errors.append("V52r2 repair lock is invalid or does not authorize implementation")
    missing = [path for path in REQUIRED if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V52r2 repair implementation files missing: {missing}")

    source_implementation = json.loads(
        (PROJECT_ROOT / lock["source_implementation_lock"]).read_text()
    )
    config = source_implementation["config_payload"]
    record = next(
        json.loads(line)
        for line in (
            PROJECT_ROOT / "data/v52-rao-blackwellized-particle-filtering/sbc.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["id"] == "sbc_00006"
    )
    registry = mechanic_registry()
    inference_args = (
        registry, record["supports"], record["query"],
        config["particleBudgets"]["primaryBudget"],
        config["population"]["particleSeed"], "sbc", record["id"],
        config["particleBudgets"]["primaryRepeatOnSbc"],
        config["algorithm"]["resamplingEssThresholdFraction"],
    )
    base = base_inference(*inference_args)
    repaired = repaired_inference(*inference_args)
    max_tv = max(
        sequence_tv(base["support_program"], repaired["support_program"]),
        sequence_tv(base["query_program"], repaired["query_program"]),
        distribution_tv(base["probability"], repaired["probability"]),
        distribution_tv(base["joint"], repaired["joint"]),
        distribution_tv(base["configuration"], repaired["configuration"]),
        distribution_tv(base["suffix"], repaired["suffix"]),
    )
    paths_identical = (
        base["support_log_evidence_by_program"]
        == repaired["support_log_evidence_by_program"]
        and base["query_log_weight_by_program"]
        == repaired["query_log_weight_by_program"]
        and base["record_log_evidence"] == repaired["record_log_evidence"]
        and base["support_diagnostics"] == repaired["support_diagnostics"]
        and base["query_diagnostics"] == repaired["query_diagnostics"]
    )
    with localcontext() as context:
        context.prec = 100
        residuals = {
            name: str(abs(sum(values, Decimal(0)) - 1))
            for name, values in (
                ("support_program", repaired["support_program"]),
                ("query_program", repaired["query_program"]),
                ("probability", repaired["probability"].values()),
                ("joint", repaired["joint"].values()),
                ("configuration", repaired["configuration"].values()),
                ("suffix", repaired["suffix"].values()),
            )
        }
        normalization_ok = all(
            Decimal(value) < Decimal("1e-80") for value in residuals.values()
        )
    tv_ok = max_tv <= float(lock["maximum_permitted_base_vs_repair_tv"])
    if not paths_identical:
        errors.append("V52r2 changes filtering likelihoods, paths, or resampling diagnostics")
    if not tv_ok:
        errors.append("V52r2 exceeds the preregistered base-versus-repair TV limit")
    if not normalization_ok:
        errors.append("V52r2 does not meet the frozen normalization tolerance")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v52r2-implementation-lock.json",
            "configs/v52r2-outcome-lock.json",
            "outputs/v52r2-joint-normalization-repair/evaluation-attempt.json",
            "outputs/v52r2-joint-normalization-repair/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V52r2 evaluation exists before implementation lock")
    audit = {
        "schema_version": 52,
        "revision": "r2",
        "experiment": "v52r2_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v52r2_implementation_lock" if not errors
            else "repair_v52r2_implementation"
        ),
        "errors": errors,
        "repair_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "repair_lock_sha256": file_sha256(lock_path),
        "checks": {
            "source_chain_bound": source_bound,
            "implementation_complete": not missing,
            "filtering_paths_and_diagnostics_identical": paths_identical,
            "substantive_tv_below_limit": tv_ok,
            "all_marginals_meet_frozen_normalization_tolerance": normalization_ok,
            "downstream_absent": downstream_absent,
        },
        "fixture": {
            "record": record["id"],
            "maximum_base_vs_repair_tv": max_tv,
            "normalization_residuals": residuals,
        },
        "data_access": {
            "post_failure_diagnostic_records": 1,
            "repair_evaluation_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
