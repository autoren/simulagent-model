#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal

from generate_v50_history import query_case
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import mechanic_registry as v49_registry
from v50_belief import mechanic_registry, query_predictive


REQUIRED = (
    "python/v50_belief.py",
    "python/generate_v50_history.py",
    "python/evaluate_v50_history.py",
    "python/audit_v50_corpus.py",
    "python/seal_v50_corpus.py",
    "python/audit_and_summarize_v50.py",
    "python/freeze_v50_outcome.py",
    "scripts/run-v50-history-dependent-belief-filtering.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v50-design-lock.json")
    parser.add_argument("--output", default="outputs/v50-history-dependent-belief-filtering/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    errors = []
    if not design["authorization"]["write_history_dependent_implementation"]:
        errors.append("V50 design does not authorize implementation")
    if design["config_sha256"] != file_sha256(PROJECT_ROOT / design["config"]):
        errors.append("V50 design config changed after lock")
    if design["preregistration_sha256"] != file_sha256(PROJECT_ROOT / design["preregistration"]):
        errors.append("V50 preregistration changed after lock")
    missing = [path for path in REQUIRED if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V50 implementation files missing: {missing}")

    registry = mechanic_registry()
    previous = {
        row["key"]
        for source in (v46_registry, v47_registry, v48_registry, v49_registry)
        for row in source()
    }
    keys = {row["key"] for row in registry}
    counts = Counter((row["family"], row["probability"]) for row in registry)
    registry_ok = len(registry) == 48 and len(keys) == 48 and not keys & previous and set(counts.values()) == {4}
    if not registry_ok:
        errors.append("V50 registry is not fresh, unique, and balanced")

    config = design["config_payload"]
    constructed = []
    for family in config["population"]["families"]:
        mechanic = next(row for row in registry if row["family"] == family)
        built = query_case(mechanic, 0, config, set())
        constructed.append((mechanic, built))
    history_constructed = all(
        built["oracle_tv"] >= config["historyDependenceContract"]["minimumOracleFullHistoryVsLatestOnlyTv"]
        and built["history_kl"] >= 0.05 and built["shuffled_kl"] >= 0.05
        for _, built in constructed
    )
    if not history_constructed:
        errors.append("V50 synthetic construction does not guarantee history or time dependence")

    mechanic, built = constructed[0]
    prediction, weights, _ = query_predictive(
        [mechanic], [Decimal(1)], built["case"]["entities"], built["case"]["initial_world"],
        built["case"]["actions"], built["evidence"], built["prefix_length"],
    )
    oracle_exact = (
        abs(sum(prediction.values(), Decimal(0)) - Decimal(1)) < Decimal("1e-80")
        and weights == [Decimal(1)]
    )
    if not oracle_exact:
        errors.append("V50 oracle-program filter is not exact")

    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in (
        "configs/v50-implementation-lock.json",
        "data/v50-history-dependent-belief-filtering",
        "outputs/v50-history-dependent-belief-filtering/development",
    ))
    if not downstream_absent:
        errors.append("V50 downstream artifact exists before implementation lock")
    audit = {
        "schema_version": 50,
        "experiment": "v50_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v50_implementation_lock" if not errors else "repair_v50_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "checks": {
            "design_bound": design["config_sha256"] == file_sha256(PROJECT_ROOT / design["config"]),
            "implementation_complete": not missing,
            "fresh_balanced_registry": registry_ok,
            "oracle_program_filter_exact": oracle_exact,
            "history_dependent_construction": history_constructed,
            "value_independent_schedule_contract": all(
                built["case"]["informative_step"] < built["prefix_length"] - 1
                for _, built in constructed
            ),
            "downstream_absent": downstream_absent,
        },
        "registry": {
            "mechanics": len(registry),
            "overlap_v46_through_v49": len(keys & previous),
            "family_probability_cells": {
                f"{family}|{probability}": count
                for (family, probability), count in sorted(counts.items())
            },
        },
        "data_access": {
            "synthetic_audit_queries": len(constructed),
            "development_mechanics_constructed": 0,
            "development_runs": 0,
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
