#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import exact_inference, mechanic_registry
from v54_eig import belief_atoms_from_exact
from v55_planning import evaluate_policy, plan_exact
from v55r1_planning import planning_registry
from v56_verification import (
    compile_policy_dtmc,
    direct_policy_statistics,
    finite_model,
    model_statistics,
    transition_rows_normalize,
    write_policy_bundle,
)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cohort_inputs(cohort: str):
    if cohort == "v55":
        seal_path = PROJECT_ROOT / "configs/v55-population-seal.json"
        outcome_path = PROJECT_ROOT / "configs/v55-outcome-lock.json"
        config_path = PROJECT_ROOT / "configs/v55-short-horizon-bayes-adaptive-planning.json"
        seal = json.loads(seal_path.read_text())
        population_path = PROJECT_ROOT / seal["population"]["path"]
        population_sha = seal["population"]["sha256"]
        registry = mechanic_registry(5303)
    elif cohort == "v55r1":
        seal_path = PROJECT_ROOT / "configs/v55r1-population-seal.json"
        outcome_path = PROJECT_ROOT / "configs/v55r1-outcome-lock.json"
        config_path = PROJECT_ROOT / "configs/v55r1-delayed-consequence-adequacy-confirmation.json"
        seal = json.loads(seal_path.read_text())
        population_path = PROJECT_ROOT / seal["population"]
        population_sha = seal["population_sha256"]
        config = json.loads(config_path.read_text())
        registry = planning_registry(config)
    else:
        raise ValueError(cohort)
    if file_sha256(population_path) != population_sha:
        raise RuntimeError(f"{cohort} sealed population changed")
    outcome = json.loads(outcome_path.read_text())
    result_path = PROJECT_ROOT / outcome["result"]
    if file_sha256(result_path) != outcome["result_sha256"]:
        raise RuntimeError(f"{cohort} source result changed")
    return {
        "cohort": cohort,
        "seal_path": seal_path,
        "outcome_path": outcome_path,
        "population_path": population_path,
        "result_path": result_path,
        "config": json.loads(config_path.read_text()),
        "registry": registry,
        "rows": read_jsonl(population_path),
        "result": json.loads(result_path.read_text()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v56-implementation-lock.json"
    )
    parser.add_argument(
        "--output", default="data/v56-symbolic-probabilistic-policy-verification"
    )
    args = parser.parse_args()
    implementation_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V56 verification bundle target already exists")
    implementation = json.loads(implementation_path.read_text())
    if (
        not implementation["authorization"][
            "construct_and_audit_v56_verification_bundle"
        ]
        or implementation["authorization"]["run_v56_candidate_formal_verification"]
        or file_sha256(PROJECT_ROOT / implementation["implementation"])
        != implementation["implementation_sha256"]
        or file_sha256(PROJECT_ROOT / implementation["implementation_audit"])
        != implementation["implementation_audit_sha256"]
    ):
        raise RuntimeError("V56 implementation lock is not intact or authorized")
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    v56_config = design["config_payload"]
    v53_config = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    cohort_inputs = [_cohort_inputs("v55"), _cohort_inputs("v55r1")]
    expected = {"v55": 32, "v55r1": 16}
    if {row["cohort"]: len(row["rows"]) for row in cohort_inputs} != expected:
        raise RuntimeError("V56 source population census changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v56-bundle-build-", dir=output.parent
    ) as temp:
        temp_root = Path(temp) / output.name
        temp_root.mkdir()
        manifest_rows = []
        started = time.time()
        completed = 0
        for cohort_data in cohort_inputs:
            cohort = cohort_data["cohort"]
            result_by_id = {
                row["id"]: row for row in cohort_data["result"]["records"]
            }
            for row in sorted(cohort_data["rows"], key=lambda value: value["record"]):
                if set(row) - {"id", "schema_version", "population", "record", "history_class", "public", "truth", "revision"}:
                    raise RuntimeError("unexpected source record field")
                public = row["public"]
                if "truth" in public:
                    raise PermissionError("truth leaked inside public source record")
                query, goal = public["query"], public["goal"]
                tick = query["prefix_length"]
                horizon = cohort_data["config"]["planningModel"]["horizonActions"]
                exact = exact_inference(
                    cohort_data["registry"],
                    {"supports": public["supports"], "query": query},
                    v53_config,
                )
                atoms = belief_atoms_from_exact(exact)
                policy = plan_exact(
                    atoms, cohort_data["registry"], query["entities"], goal,
                    horizon, tick, cohort_data["config"],
                )
                frozen = result_by_id[row["id"]]
                action_match = (
                    policy["selected_action_key"] == frozen["selected_action_key"]
                )
                value_error = abs(policy["value"] - frozen["root_value"])
                if not action_match or value_error > 1e-10:
                    raise RuntimeError(
                        f"{cohort}/{row['id']} does not reconstruct its frozen root"
                    )
                independent_value = evaluate_policy(
                    atoms, policy, cohort_data["registry"], query["entities"],
                    goal, horizon, tick, cohort_data["config"],
                )
                direct = direct_policy_statistics(
                    atoms, policy, cohort_data["registry"], query["entities"],
                    goal, horizon, tick, v56_config,
                )
                model = compile_policy_dtmc(
                    atoms, policy, cohort_data["registry"], query["entities"],
                    goal, horizon, tick, v56_config,
                )
                graph = model_statistics(model)
                maximum_preseal_error = max(
                    abs(independent_value - policy["value"]),
                    abs(direct["expected_return"] - policy["value"]),
                    abs(graph["expected_return"] - policy["value"]),
                    abs(graph["success_probability"] - direct["success_probability"]),
                    abs(graph["termination_probability"] - 1.0),
                )
                if (
                    maximum_preseal_error > 1e-10
                    or not transition_rows_normalize(model)
                    or not finite_model(model)
                ):
                    raise RuntimeError(
                        f"{cohort}/{row['id']} failed preseal compiler checks: "
                        f"{maximum_preseal_error}"
                    )
                relative = Path(cohort) / row["id"]
                directory = temp_root / relative
                metadata = {
                    "schema_version": 56,
                    "experiment": "v56_policy_model_bundle",
                    "cohort": cohort,
                    "id": row["id"],
                    "record": row["record"],
                    "history_class": row["history_class"],
                    "source_population_seal": str(
                        cohort_data["seal_path"].relative_to(PROJECT_ROOT)
                    ),
                    "source_population_seal_sha256": file_sha256(
                        cohort_data["seal_path"]
                    ),
                    "source_outcome_lock": str(
                        cohort_data["outcome_path"].relative_to(PROJECT_ROOT)
                    ),
                    "source_outcome_lock_sha256": file_sha256(
                        cohort_data["outcome_path"]
                    ),
                    "source_result": str(
                        cohort_data["result_path"].relative_to(PROJECT_ROOT)
                    ),
                    "source_result_sha256": file_sha256(cohort_data["result_path"]),
                    "frozen_root_action_key": frozen["selected_action_key"],
                    "frozen_root_value": frozen["root_value"],
                    "reconstructed_root_action_key": policy["selected_action_key"],
                    "reconstructed_root_value": policy["value"],
                    "reconstructed_root_value_error": value_error,
                    "independent_policy_value": independent_value,
                    "direct_executor": direct,
                    "compiled_graph": graph,
                    "belief_atoms": len(atoms),
                    "truth_field_access_count": 0,
                    "maximum_preseal_reference_error": maximum_preseal_error,
                }
                write_policy_bundle(model, policy, directory, metadata)
                files = []
                for name in v56_config["verificationBundle"]["requiredFilesPerPolicy"]:
                    path = directory / name
                    files.append({
                        "path": str((relative / name).as_posix()),
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    })
                manifest_rows.append({
                    "cohort": cohort,
                    "id": row["id"],
                    "record": row["record"],
                    "directory": str(relative.as_posix()),
                    "states": len(model["states"]),
                    "transitions": len(model["transitions"]),
                    "belief_atoms": len(atoms),
                    "frozen_root_action_key": frozen["selected_action_key"],
                    "frozen_root_value": frozen["root_value"],
                    "reconstructed_root_value_error": value_error,
                    "maximum_preseal_reference_error": maximum_preseal_error,
                    "files": files,
                })
                completed += 1
                print(json.dumps({
                    "completed": completed,
                    "total": 48,
                    "cohort": cohort,
                    "id": row["id"],
                    "states": len(model["states"]),
                    "transitions": len(model["transitions"]),
                    "seconds": time.time() - started,
                }, sort_keys=True), flush=True)
        manifest = {
            "schema_version": 56,
            "experiment": "v56_verification_bundle_manifest",
            "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
            "implementation_lock_sha256": file_sha256(implementation_path),
            "policy_count": len(manifest_rows),
            "cohort_counts": {
                cohort: sum(row["cohort"] == cohort for row in manifest_rows)
                for cohort in ("v55", "v55r1")
            },
            "policies": manifest_rows,
            "truth_field_access_count": 0,
            "candidate_formal_verification_runs": 0,
            "runtime_seconds": time.time() - started,
        }
        (temp_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temp_root.replace(output)
    print(json.dumps({
        "bundle": str(output.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "policies": 48,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
