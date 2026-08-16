#!/usr/bin/env python3
"""Pre-run audit of the V57 one-shot evaluator using altered-seed data only."""
from __future__ import annotations

import argparse
import copy
import inspect
import json

from evaluate_v57_definition_transfer import (
    aggregate,
    evaluate_core_record,
    evaluate_safety_record,
)
from generate_v57_definition_transfer import build_populations
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v57_definition_compiler import compile_agent_input


EVALUATION_FILES = (
    "python/evaluate_v57_definition_transfer.py",
    "python/audit_and_summarize_v57.py",
    "python/freeze_v57_outcome.py",
    "python/audit_v57_evaluation_implementation.py",
)

FROZEN_DEPENDENCIES = (
    "python/v57_definition_compiler.py",
    "python/generate_v57_definition_transfer.py",
    "configs/v57-design-lock.json",
    "configs/v57-implementation-lock.json",
    "configs/v57-population-seal.json",
)


EXPECTED_CHECKS = {
    "core_coverage",
    "core_exact_ast",
    "predicate_compiled_truth",
    "exact_bound_action",
    "every_ontology_pack_exact_ast",
    "every_concept_kind_exact_ast",
    "every_definition_template_exact_ast",
    "no_definition_abstention_rate",
    "opaque_name_only_abstention_rate",
    "shuffled_schema_exact_ast",
    "safety_rate",
    "every_safety_condition_rate",
    "mutation_kill_rate",
    "truth_access_count",
    "unexpected_evaluation_attempt_count",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population-seal", default="configs/v57-population-seal.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v57-definition-augmented-ontology-transfer/"
            "evaluation-implementation-audit.json"
        ),
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config = copy.deepcopy(implementation["config_payload"])
    errors: list[str] = []

    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    seal_bound = (
        seal["authorization"]["write_and_audit_v57_candidate_runner"]
        and not seal["authorization"]["run_v57_candidate_evaluation"]
        and not seal["authorization"]["modify_v57_population"]
        and file_sha256(manifest_path) == seal["manifest_sha256"]
        and file_sha256(PROJECT_ROOT / seal["population_audit"])
        == seal["population_audit_sha256"]
        and file_sha256(implementation_path)
        == seal["implementation_lock_sha256"]
        and manifest["evaluation_runs"] == 0
        and manifest["truth_access_count"] == 0
        and manifest["artifacts"] == seal["artifacts"]
    )
    if not seal_bound:
        errors.append("V57 population seal is not intact and pre-evaluation")

    compiler_source = inspect.getsource(compile_agent_input)
    evaluator_source = inspect.getsource(evaluate_core_record)
    firewall_ok = (
        set(inspect.signature(compile_agent_input).parameters)
        == {"agent_input", "mutation"}
        and "target" not in compiler_source
        and "oracle_metadata" not in compiler_source
        and "model" not in compiler_source
        and 'compile_agent_input(row["public"])' in evaluator_source
        and "compile_agent_input(row)" not in evaluator_source
    )
    if not firewall_ok:
        errors.append("V57 evaluator crosses the public-input compiler firewall")

    config["population"]["generatorSeed"] += 9_000_000
    populations = build_populations(config)
    synthetic_core = [
        evaluate_core_record(row) for row in populations["core"]
    ]
    synthetic_safety = [
        evaluate_safety_record(row) for row in populations["safety"]
    ]
    integrity = {
        "truth_access_count": 0,
        "unexpected_evaluation_attempt_count": 0,
        "population_hash_mismatch_count": 0,
    }
    aggregated = aggregate(
        synthetic_core,
        synthetic_safety,
        config,
        integrity,
        implementation["mutation_kill_rate"],
    )
    full_fixture_ok = (
        len(synthetic_core) == config["population"]["coreRecords"] == 1920
        and len(synthetic_safety)
        == config["safetyPopulation"]["records"]
        == 224
        and aggregated["passed"]
        and len(aggregated["checks"]) == len(config["gates"]) == 15
        and set(aggregated["checks"]) == EXPECTED_CHECKS
    )
    if not full_fixture_ok:
        errors.append("V57 altered-seed full evaluator fixture failed")

    missing_core = aggregate(
        synthetic_core[:-1],
        synthetic_safety,
        config,
        integrity,
        implementation["mutation_kill_rate"],
    )
    duplicate_core = aggregate(
        synthetic_core[:-1] + [synthetic_core[0]],
        synthetic_safety,
        config,
        integrity,
        implementation["mutation_kill_rate"],
    )
    missing_safety = aggregate(
        synthetic_core,
        synthetic_safety[:-1],
        config,
        integrity,
        implementation["mutation_kill_rate"],
    )
    denominator_ok = (
        not missing_core["passed"]
        and not duplicate_core["passed"]
        and not missing_safety["passed"]
        and missing_core["metrics"]["core"]["exact_ast"] < 1.0
        and duplicate_core["metrics"]["core"]["exact_ast"] < 1.0
        and missing_safety["metrics"]["safety"]["rate"] < 1.0
    )
    if not denominator_ok:
        errors.append("V57 fixed-denominator missing/duplicate controls failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v57-evaluation-implementation-lock.json",
            "configs/v57-outcome-lock.json",
            "outputs/v57-definition-augmented-ontology-transfer/evaluation-attempt.json",
            "outputs/v57-definition-augmented-ontology-transfer/evaluation",
            "outputs/v57-definition-augmented-ontology-transfer/post-result-audit.json",
            "docs/v57-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V57 candidate evaluation or downstream artifact exists")

    audit = {
        "schema_version": 57,
        "experiment": "v57_evaluation_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v57_evaluation_implementation_lock"
            if not errors else "repair_v57_evaluation_implementation"
        ),
        "errors": errors,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "implementation_lock": str(
            implementation_path.relative_to(PROJECT_ROOT)
        ),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in EVALUATION_FILES
        },
        "frozen_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path)
            for path in FROZEN_DEPENDENCIES
        },
        "checks": {
            "sealed_population_metadata_and_implementation": seal_bound,
            "public_input_compiler_firewall": firewall_ok,
            "altered_seed_full_evaluator_and_fifteen_gates": full_fixture_ok,
            "fixed_denominator_missing_and_duplicate_controls": denominator_ok,
            "single_attempt_and_downstream_absence": downstream_absent,
        },
        "altered_seed_metrics": aggregated["metrics"],
        "data_access": {
            "v57_candidate_records_accessed": 0,
            "v57_candidate_evaluation_runs": 0,
            "altered_seed_core_records_generated": len(synthetic_core),
            "altered_seed_safety_records_generated": len(synthetic_safety),
            "compiler_truth_or_target_access_count": 0,
            "human_authored_records_collected": 0,
            "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
