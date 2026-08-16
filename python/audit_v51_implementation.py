#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter

from evaluate_v51_sbc import evaluate_replication
from generate_v51_sbc import build_replications
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import mechanic_registry as v49_registry
from v50_belief import mechanic_registry as v50_registry
from v51_sbc import mechanic_registry


REQUIRED = (
    "python/v51_sbc.py",
    "python/generate_v51_sbc.py",
    "python/evaluate_v51_sbc.py",
    "python/test_v51_sbc.py",
    "python/audit_v51_corpus.py",
    "python/seal_v51_corpus.py",
    "python/audit_and_summarize_v51.py",
    "python/freeze_v51_outcome.py",
    "scripts/run-v51-simulation-based-calibration.sh",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v51-design-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v51-simulation-based-calibration/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    errors = []
    if not design["authorization"]["write_sbc_implementation"]:
        errors.append("V51 design does not authorize SBC implementation")
    if design["config_sha256"] != file_sha256(PROJECT_ROOT / design["config"]):
        errors.append("V51 design config changed after lock")
    if design["preregistration_sha256"] != file_sha256(
        PROJECT_ROOT / design["preregistration"]
    ):
        errors.append("V51 preregistration changed after lock")
    missing = [path for path in REQUIRED if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V51 implementation files missing: {missing}")

    registry = mechanic_registry()
    previous = {
        row["key"]
        for source in (v46_registry, v47_registry, v48_registry, v49_registry, v50_registry)
        for row in source()
    }
    keys = {row["key"] for row in registry}
    counts = Counter((row["family"], row["probability"]) for row in registry)
    registry_ok = (
        len(registry) == 48
        and len(keys) == 48
        and not keys & previous
        and set(counts.values()) == {4}
    )
    if not registry_ok:
        errors.append("V51 registry is not fresh, unique, and balanced")

    smoke_config = copy.deepcopy(design["config_payload"])
    smoke_config["simulation"]["replications"] = 6
    for name in (
        "generatorSeed", "priorSeed", "trajectorySeed", "posteriorDrawSeed", "tieBreakSeed"
    ):
        smoke_config["simulation"][name] += 1_000_000
    smoke = [
        evaluate_replication(record, registry, smoke_config)
        for record in build_replications(smoke_config)
    ]
    normalization_ok = all(row["normalization"] for row in smoke)
    exact_path_maximum = max(
        value for row in smoke for value in row["exact_agreement"].values()
    )
    exact_paths_ok = exact_path_maximum <= 1e-90
    if not normalization_ok:
        errors.append("V51 smoke-test probability normalization failed")
    if not exact_paths_ok:
        errors.append("V51 batch and independent exact paths disagree")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v51-implementation-lock.json",
            "configs/v51-corpus-seal.json",
            "data/v51-simulation-based-calibration",
            "outputs/v51-simulation-based-calibration/calibration-attempt.json",
            "outputs/v51-simulation-based-calibration/calibration",
        )
    )
    if not downstream_absent:
        errors.append("V51 downstream artifact exists before implementation lock")

    audit = {
        "schema_version": 51,
        "experiment": "v51_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v51_implementation_lock" if not errors else "repair_v51_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "checks": {
            "design_bound": design["config_sha256"]
            == file_sha256(PROJECT_ROOT / design["config"]),
            "implementation_complete": not missing,
            "fresh_balanced_registry": registry_ok,
            "smoke_normalization": normalization_ok,
            "independent_exact_path_agreement": exact_paths_ok,
            "downstream_absent": downstream_absent,
        },
        "smoke": {
            "replications": len(smoke),
            "maximum_exact_path_tv": exact_path_maximum,
        },
        "data_access": {
            "synthetic_smoke_replications": len(smoke),
            "calibration_replications_accessed": 0,
            "calibration_runs": 0,
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
