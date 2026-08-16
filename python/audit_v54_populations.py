#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from generate_v54_eig import (
    observation_design_key,
    population_hash,
    prior_observation_design_keys,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import atom_universe
from v54_eig import assert_selection_payload_is_public, candidate_interventions


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/v54-exact-one-step-eig/manifest.json")
    parser.add_argument("--implementation-lock", default="configs/v54-implementation-lock.json")
    parser.add_argument(
        "--evaluation-implementation-lock",
        default="configs/v54-evaluation-implementation-lock.json",
    )
    parser.add_argument(
        "--output", default="outputs/v54-exact-one-step-eig/population-audit.json"
    )
    args = parser.parse_args()
    manifest_path, implementation_path, evaluation_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (
            args.manifest, args.implementation_lock,
            args.evaluation_implementation_lock, args.output,
        )
    )
    manifest = json.loads(manifest_path.read_text())
    implementation = json.loads(implementation_path.read_text())
    evaluation_lock = json.loads(evaluation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    errors = []

    locks_ok = (
        implementation["authorization"]["construct_v54_active_populations"]
        and evaluation_lock["authorization"]["construct_v54_active_populations"]
        and not evaluation_lock["authorization"]["run_v54_active_evaluation"]
        and evaluation_lock["implementation_lock_sha256"] == file_sha256(implementation_path)
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for lock in (implementation, evaluation_lock)
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in lock.get(section, {}).items()
        )
    )
    if not locks_ok:
        errors.append("V54 implementation or evaluation lock is not intact")

    populations = {}
    files_ok = True
    for name in ("selection", "adaptive_sbc"):
        artifact = manifest["files"][name]
        path = PROJECT_ROOT / artifact["path"]
        files_ok = files_ok and file_sha256(path) == artifact["sha256"]
        populations[name] = read_jsonl(path)
    files_ok = files_ok and population_hash(populations) == manifest["population_hash"]
    if not files_ok:
        errors.append("V54 population files or combined hash do not match manifest")

    counts_ok = (
        len(populations["selection"]) == config["population"]["selectionRecords"] == 64
        and len(populations["adaptive_sbc"]) == config["adaptiveSbc"]["replications"] == 256
        and manifest["counts"] == {"selection": 64, "adaptive_sbc": 256}
    )
    if not counts_ok:
        errors.append("V54 population counts do not match preregistration")

    ids = [row["id"] for rows in populations.values() for row in rows]
    ids_unique = len(ids) == len(set(ids))
    if not ids_unique:
        errors.append("V54 record identifiers are not unique")

    public_ok, truth_nested = True, True
    design_keys, history_keys = [], []
    class_counts = Counter()
    program_counts = Counter()
    selection_shapes_ok = True
    adaptive_shapes_ok = True
    for name, rows in populations.items():
        for row in rows:
            try:
                assert_selection_payload_is_public(row["public_history"])
            except PermissionError:
                public_ok = False
            truth_nested = truth_nested and "truth" in row and not any(
                key.startswith("target_") or key.startswith("true_")
                for key in row if key != "truth"
            )
            history_keys.append(sha256_text(canonical_json(row["public_history"])))
            for episode in [
                *row["public_history"]["supports"], row["public_history"]["query"]
            ]:
                design_keys.append(observation_design_key(episode))
            if name == "selection":
                class_counts[row["history_class"]] += 1
                program_counts[row["truth"]["target_program_index"]] += 1
                selection_shapes_ok = selection_shapes_ok and (
                    "selected_intervention" not in row and "realized_outcome" not in row
                )
            else:
                query = row["public_history"]["query"]
                valid = {
                    candidate["key"] for candidate in candidate_interventions(query["entities"])
                }
                panel = set(atom_universe(query["entities"]))
                outcome = row["realized_outcome"]
                adaptive_shapes_ok = adaptive_shapes_ok and (
                    row["selected_intervention"]["key"] in valid
                    and len(row["selected_intervention"]["assay"]) == 3
                    and len(outcome["observations"]) == len(outcome["masks"]) == 3
                    and all(set(mask) == panel for mask in outcome["masks"])
                )
    if not public_ok or not truth_nested:
        errors.append("V54 truth or outcome fields crossed the public selection boundary")
    if not selection_shapes_ok or not adaptive_shapes_ok:
        errors.append("V54 selection or adaptive-SBC record shape is invalid")

    allocation_ok = (
        dict(class_counts) == config["population"]["historyClasses"]
        and program_counts == Counter({index: 8 for index in range(8)})
    )
    if not allocation_ok:
        errors.append("V54 selection histories are not balanced by class and generating template")

    prior = prior_observation_design_keys()
    freshness_ok = (
        len(design_keys) == len(set(design_keys))
        and not (set(design_keys) & prior)
        and len(history_keys) == len(set(history_keys))
    )
    if not freshness_ok:
        errors.append("V54 public histories or observation designs are reused")

    evaluation_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v54-population-seal.json",
            "configs/v54-outcome-lock.json",
            "outputs/v54-exact-one-step-eig/evaluation-attempt.json",
            "outputs/v54-exact-one-step-eig/evaluation",
        )
    )
    if not evaluation_absent:
        errors.append("V54 evaluation or downstream lock exists before population seal")

    audit = {
        "schema_version": 54,
        "experiment": "v54_population_audit",
        "passed": not errors,
        "decision": "authorize_v54_population_seal" if not errors else "reject_v54_populations",
        "errors": errors,
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "evaluation_implementation_lock": str(evaluation_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_path),
        "checks": {
            "locks_intact_and_pre_evaluation": locks_ok,
            "manifest_and_population_hashes": files_ok,
            "population_counts": counts_ok,
            "record_ids_unique": ids_unique,
            "public_selection_boundary": public_ok,
            "truth_fields_nested": truth_nested,
            "selection_record_shapes": selection_shapes_ok,
            "adaptive_record_shapes": adaptive_shapes_ok,
            "selection_class_and_template_balance": allocation_ok,
            "fresh_unique_histories_and_designs": freshness_ok,
            "evaluation_absent": evaluation_absent,
        },
        "counts": {
            "selection": len(populations["selection"]),
            "adaptive_sbc": len(populations["adaptive_sbc"]),
            "history_classes": dict(class_counts),
            "selection_generating_programs": dict(sorted(program_counts.items())),
            "unique_public_histories": len(set(history_keys)),
            "unique_observation_designs": len(set(design_keys)),
        },
        "data_access": {
            "active_evaluation_runs": 0,
            "selection_metrics_computed": 0,
            "adaptive_sbc_ranks_computed": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
