#!/usr/bin/env python3
"""One-shot sealed evaluation for V57 definition-conditioned transfer."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v57_definition_compiler import (
    compile_agent_input,
    compiled_truth,
    render_controlled_definition,
)


KINDS = ("unary_predicate", "binary_relation", "bound_action")
FAMILIES = ("signature_first", "meaning_first", "example_first")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def shuffled_definitions(public: dict[str, Any]) -> dict[str, Any]:
    """Rotate complete definitions across opaque IDs within each concept kind."""
    result = copy.deepcopy(public)
    definitions = result["concept_definitions"]
    for kind in KINDS:
        indices = [
            index for index, row in enumerate(definitions)
            if row["kind"] == kind
        ]
        old_ids = [definitions[index]["opaque_id"] for index in indices]
        rotated = old_ids[1:] + old_ids[:1]
        for index, new_id in zip(indices, rotated, strict=True):
            row = definitions[index]
            row["opaque_id"] = new_id
            row["controlled_definition"] = render_controlled_definition(
                new_id,
                row["kind"],
                row["typed_signature"],
                row["lexical_forms"],
                row["definition_template_family"],
            )
    return result


def evaluate_core_record(row: dict[str, Any]) -> dict[str, Any]:
    """Compile only the public payload, then score outside the compiler."""
    result = compile_agent_input(row["public"])
    status = result.get("status", "error")
    parse = result.get("parse")
    target = row["target"]
    exact = status == "ok" and parse == target["parse"]
    kind = row["oracle_metadata"]["concept_kind"]
    predicate_truth_exact = None
    if kind != "bound_action":
        predicate_truth_exact = (
            status == "ok"
            and compiled_truth(parse) == target["compiled_truth"]
        )

    no_definition = copy.deepcopy(row["public"])
    no_definition["concept_definitions"] = []
    no_definition_status = compile_agent_input(no_definition)["status"]

    opaque_only = copy.deepcopy(row["public"])
    opaque_only["concept_definitions"] = [
        {"opaque_id": concept["opaque_id"]}
        for concept in opaque_only["concept_definitions"]
    ]
    opaque_only_status = compile_agent_input(opaque_only)["status"]

    shuffled = compile_agent_input(shuffled_definitions(row["public"]))
    shuffled_exact = (
        shuffled.get("status") == "ok"
        and shuffled.get("parse") == target["parse"]
    )
    return {
        "id": row["id"],
        "ontology_pack": row["ontology_pack"],
        "concept_kind": kind,
        "definition_template_family": row["oracle_metadata"][
            "definition_template_family"
        ],
        "status": status,
        "parse": parse,
        "exact_ast": exact,
        "predicate_compiled_truth_exact": predicate_truth_exact,
        "exact_bound_action": exact if kind == "bound_action" else None,
        "no_definition_status": no_definition_status,
        "opaque_name_only_status": opaque_only_status,
        "shuffled_schema_status": shuffled.get("status", "error"),
        "shuffled_schema_exact_ast": shuffled_exact,
    }


def evaluate_safety_record(row: dict[str, Any]) -> dict[str, Any]:
    result = compile_agent_input(row["public"])
    status = result.get("status", "error")
    return {
        "id": row["id"],
        "condition": row["oracle_metadata"]["condition"],
        "status": status,
        "passed": status in row["expected"]["statuses"],
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _unique(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prevent duplicate output rows from compensating for missing records."""
    unique: dict[str, dict[str, Any]] = {}
    for row in records:
        unique.setdefault(row["id"], row)
    return list(unique.values())


def aggregate(
    core_records: list[dict[str, Any]],
    safety_records: list[dict[str, Any]],
    config: dict[str, Any],
    integrity: dict[str, int],
    mutation_kill_rate: float,
) -> dict[str, Any]:
    """Apply all 15 preregistered gates with fixed expected denominators."""
    core = _unique(core_records)
    safety = _unique(safety_records)
    population = config["population"]
    safety_population = config["safetyPopulation"]
    expected_core = population["coreRecords"]
    expected_packs = [
        f"pack_{index:02d}" for index in range(population["ontologyPacks"])
    ]
    per_pack = population["coreRecordsPerPack"]
    per_kind = expected_core // len(KINDS)
    per_family = expected_core // len(FAMILIES)
    expected_predicates = per_kind * 2
    expected_actions = per_kind
    expected_safety = safety_population["records"]
    per_condition = (
        population["ontologyPacks"]
        * safety_population["recordsPerConditionPerPack"]
    )

    pack_rates = {
        pack: _rate(
            sum(row["exact_ast"] for row in core if row["ontology_pack"] == pack),
            per_pack,
        )
        for pack in expected_packs
    }
    kind_rates = {
        kind: _rate(
            sum(row["exact_ast"] for row in core if row["concept_kind"] == kind),
            per_kind,
        )
        for kind in KINDS
    }
    family_rates = {
        family: _rate(
            sum(
                row["exact_ast"] for row in core
                if row["definition_template_family"] == family
            ),
            per_family,
        )
        for family in FAMILIES
    }
    safety_rates = {
        condition: _rate(
            sum(
                row["passed"] for row in safety
                if row["condition"] == condition
            ),
            per_condition,
        )
        for condition in safety_population["conditions"]
    }
    metrics = {
        "core": {
            "coverage": _rate(
                sum(row["status"] == "ok" for row in core), expected_core
            ),
            "exact_ast": _rate(
                sum(row["exact_ast"] for row in core), expected_core
            ),
            "predicate_compiled_truth": _rate(
                sum(
                    row["predicate_compiled_truth_exact"] is True
                    for row in core
                ),
                expected_predicates,
            ),
            "exact_bound_action": _rate(
                sum(row["exact_bound_action"] is True for row in core),
                expected_actions,
            ),
            "ontology_pack_exact_ast": pack_rates,
            "worst_ontology_pack_exact_ast": min(pack_rates.values()),
            "concept_kind_exact_ast": kind_rates,
            "worst_concept_kind_exact_ast": min(kind_rates.values()),
            "definition_template_exact_ast": family_rates,
            "worst_definition_template_exact_ast": min(
                family_rates.values()
            ),
        },
        "definition_dependence_controls": {
            "no_definition_abstention_rate": _rate(
                sum(row["no_definition_status"] == "abstain" for row in core),
                expected_core,
            ),
            "opaque_name_only_abstention_rate": _rate(
                sum(
                    row["opaque_name_only_status"] == "abstain"
                    for row in core
                ),
                expected_core,
            ),
            "shuffled_schema_exact_ast": _rate(
                sum(row["shuffled_schema_exact_ast"] for row in core),
                expected_core,
            ),
        },
        "safety": {
            "rate": _rate(sum(row["passed"] for row in safety), expected_safety),
            "condition_rates": safety_rates,
            "worst_condition_rate": min(safety_rates.values()),
        },
        "implementation_controls": {
            "mutation_kill_rate": mutation_kill_rate,
        },
        "integrity": integrity,
        "census": {
            "unique_core_records": len(core),
            "expected_core_records": expected_core,
            "unique_safety_records": len(safety),
            "expected_safety_records": expected_safety,
        },
    }
    gates = config["gates"]
    checks = {
        "core_coverage": metrics["core"]["coverage"]
        >= gates["minimumCoreCoverage"],
        "core_exact_ast": metrics["core"]["exact_ast"]
        >= gates["minimumCoreExactAst"],
        "predicate_compiled_truth": metrics["core"][
            "predicate_compiled_truth"
        ] >= gates["minimumPredicateCompiledTruth"],
        "exact_bound_action": metrics["core"]["exact_bound_action"]
        >= gates["minimumExactBoundAction"],
        "every_ontology_pack_exact_ast": metrics["core"][
            "worst_ontology_pack_exact_ast"
        ] >= gates["minimumEveryOntologyPackExactAst"],
        "every_concept_kind_exact_ast": metrics["core"][
            "worst_concept_kind_exact_ast"
        ] >= gates["minimumEveryConceptKindExactAst"],
        "every_definition_template_exact_ast": metrics["core"][
            "worst_definition_template_exact_ast"
        ] >= gates["minimumEveryDefinitionTemplateExactAst"],
        "no_definition_abstention_rate": metrics[
            "definition_dependence_controls"
        ]["no_definition_abstention_rate"]
        >= gates["minimumNoDefinitionAbstentionRate"],
        "opaque_name_only_abstention_rate": metrics[
            "definition_dependence_controls"
        ]["opaque_name_only_abstention_rate"]
        >= gates["minimumOpaqueNameOnlyAbstentionRate"],
        "shuffled_schema_exact_ast": metrics[
            "definition_dependence_controls"
        ]["shuffled_schema_exact_ast"]
        <= gates["maximumShuffledSchemaExactAst"],
        "safety_rate": metrics["safety"]["rate"]
        >= gates["minimumSafetyRate"],
        "every_safety_condition_rate": metrics["safety"][
            "worst_condition_rate"
        ] >= gates["minimumEverySafetyConditionRate"],
        "mutation_kill_rate": mutation_kill_rate
        >= gates["minimumMutationKillRate"],
        "truth_access_count": integrity["truth_access_count"]
        <= gates["maximumTruthAccessCount"],
        "unexpected_evaluation_attempt_count": integrity[
            "unexpected_evaluation_attempt_count"
        ] <= gates["maximumUnexpectedEvaluationAttemptCount"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def verify_population(seal: dict[str, Any]) -> tuple[dict[str, Any], int]:
    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    mismatches = int(file_sha256(manifest_path) != seal["manifest_sha256"])
    for name, artifact in seal["artifacts"].items():
        path = PROJECT_ROOT / artifact["path"]
        manifest_artifact = manifest["artifacts"][name]
        mismatches += int(
            not path.exists()
            or file_sha256(path) != artifact["sha256"]
            or artifact != manifest_artifact
        )
    return manifest, mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock",
        default="configs/v57-evaluation-implementation-lock.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v57-definition-augmented-ontology-transfer/evaluation",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if attempt_path.exists() or output_dir.exists():
        raise RuntimeError("V57 permits exactly one sealed candidate evaluation")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_v57_candidate_evaluation"]:
        raise RuntimeError("V57 evaluation lock does not authorize the run")
    for section in ("evaluation_files_sha256", "frozen_dependencies_sha256"):
        for relative, digest in lock[section].items():
            if file_sha256(PROJECT_ROOT / relative) != digest:
                raise RuntimeError(f"V57 frozen evaluation input changed: {relative}")

    seal_path = PROJECT_ROOT / lock["population_seal"]
    if file_sha256(seal_path) != lock["population_seal_sha256"]:
        raise RuntimeError("V57 population seal changed")
    seal = json.loads(seal_path.read_text())
    manifest, population_mismatches = verify_population(seal)
    if population_mismatches:
        raise RuntimeError("V57 sealed population changed")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = {
        "schema_version": 57,
        "experiment": "v57_candidate_evaluation_attempt",
        "evaluation_run": 1,
        "evaluation_implementation_lock": str(
            lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir()

    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config = implementation["config_payload"]
    core_rows = read_jsonl(PROJECT_ROOT / seal["artifacts"]["core"]["path"])
    safety_rows = read_jsonl(
        PROJECT_ROOT / seal["artifacts"]["safety"]["path"]
    )
    started = time.time()
    core_records = [evaluate_core_record(row) for row in core_rows]
    safety_records = [evaluate_safety_record(row) for row in safety_rows]
    integrity = {
        "truth_access_count": 0,
        "unexpected_evaluation_attempt_count": 0,
        "population_hash_mismatch_count": population_mismatches,
    }
    aggregated = aggregate(
        core_records,
        safety_records,
        config,
        integrity,
        implementation["mutation_kill_rate"],
    )
    result = {
        "schema_version": 57,
        "experiment": "v57_definition_augmented_ontology_transfer_result",
        "evaluation_run": 1,
        "evaluation_implementation_lock": str(
            lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "manifest": seal["manifest"],
        "manifest_sha256": file_sha256(PROJECT_ROOT / seal["manifest"]),
        "records": {"core": core_records, "safety": safety_records},
        "metrics": aggregated["metrics"],
        "qualification": {
            "passed": aggregated["passed"],
            "checks": aggregated["checks"],
        },
        "runtime_seconds": time.time() - started,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "qualification": result["qualification"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
