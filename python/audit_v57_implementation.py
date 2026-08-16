#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json

from generate_v57_definition_transfer import build_populations
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v57_definition_compiler import (
    compile_agent_input, compiled_truth, render_controlled_definition,
)


IMPLEMENTATION_FILES = (
    "python/v57_definition_compiler.py",
    "python/generate_v57_definition_transfer.py",
    "python/test_v57_definition_compiler.py",
)


def shuffled_definitions(public: dict) -> dict:
    result = copy.deepcopy(public)
    definitions = result["concept_definitions"]
    for kind in ("unary_predicate", "binary_relation", "bound_action"):
        indices = [index for index, row in enumerate(definitions) if row["kind"] == kind]
        old_ids = [definitions[index]["opaque_id"] for index in indices]
        rotated = old_ids[1:] + old_ids[:1]
        for index, new_id in zip(indices, rotated, strict=True):
            row = definitions[index]
            row["opaque_id"] = new_id
            row["controlled_definition"] = render_controlled_definition(
                new_id, row["kind"], row["typed_signature"],
                row["lexical_forms"], row["definition_template_family"],
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v57-design-lock.json")
    parser.add_argument("--output", default="outputs/v57-definition-augmented-ontology-transfer/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = copy.deepcopy(design["config_payload"])
    errors: list[str] = []

    design_ok = (
        design["authorization"]["write_and_audit_definition_compiler"]
        and design["authorization"]["write_and_audit_independent_generator"]
        and not design["authorization"]["construct_v57_population"]
        and not design["authorization"]["run_v57_candidate_evaluation"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
    )
    if not design_ok:
        errors.append("V57 design lock is not intact or authorized")

    compiler_source = inspect.getsource(compile_agent_input)
    boundary_ok = (
        set(inspect.signature(compile_agent_input).parameters)
        == {"agent_input", "mutation"}
        and "target" not in compiler_source
        and "oracle_metadata" not in compiler_source
        and "model" not in compiler_source
    )
    if not boundary_ok:
        errors.append("V57 compiler crosses target, oracle, or model boundary")

    config["population"]["generatorSeed"] += 7_000_000
    populations = build_populations(config)
    core, safety = populations["core"], populations["safety"]
    core_passes = truth_passes = 0
    definition_template_counts = {}
    concept_kind_counts = {}
    pack_counts = {}
    no_definition_abstentions = opaque_only_abstentions = shuffled_exact = 0
    for row in core:
        result = compile_agent_input(row["public"])
        exact = result.get("status") == "ok" and result.get("parse") == row["target"]["parse"]
        core_passes += int(exact)
        truth_passes += int(
            exact and compiled_truth(result["parse"]) == row["target"]["compiled_truth"]
        )
        metadata = row["oracle_metadata"]
        for mapping, key in (
            (definition_template_counts, metadata["definition_template_family"]),
            (concept_kind_counts, metadata["concept_kind"]),
            (pack_counts, row["ontology_pack"]),
        ):
            mapping[key] = mapping.get(key, 0) + 1
        no_def = copy.deepcopy(row["public"])
        no_def["concept_definitions"] = []
        no_definition_abstentions += int(
            compile_agent_input(no_def)["status"] == "abstain"
        )
        opaque = copy.deepcopy(row["public"])
        opaque["concept_definitions"] = [
            {"opaque_id": concept["opaque_id"]}
            for concept in opaque["concept_definitions"]
        ]
        opaque_only_abstentions += int(
            compile_agent_input(opaque)["status"] == "abstain"
        )
        shuffled = compile_agent_input(shuffled_definitions(row["public"]))
        shuffled_exact += int(
            shuffled.get("status") == "ok"
            and shuffled.get("parse") == row["target"]["parse"]
        )
    safety_passes = sum(
        compile_agent_input(row["public"])["status"] in row["expected"]["statuses"]
        for row in safety
    )
    altered_ok = (
        len(core) == 1920 and core_passes == truth_passes == len(core)
        and len(safety) == safety_passes == 224
        and set(definition_template_counts) == set(config["population"]["factors"]["definitionTemplateFamily"])
        and set(concept_kind_counts) == set(config["population"]["factors"]["conceptKind"])
        and len(pack_counts) == 16 and set(pack_counts.values()) == {120}
        and no_definition_abstentions == len(core)
        and opaque_only_abstentions == len(core)
        and shuffled_exact == 0
    )
    if not altered_ok:
        errors.append("V57 altered-seed population, controls, or compiler failed")

    relation = next(
        row for row in core
        if row["oracle_metadata"]["concept_kind"] == "binary_relation"
        and row["target"]["parse"]["arguments"][0]
        != row["target"]["parse"]["arguments"][1]
    )
    negative = next(
        row for row in core
        if row["oracle_metadata"]["concept_kind"] != "bound_action"
        and row["target"]["parse"]["lexical_sign"] == "negative"
    )
    action = next(
        row for row in core if row["oracle_metadata"]["concept_kind"] == "bound_action"
    )
    type_case = next(row for row in safety if row["oracle_metadata"]["condition"] == "type_mismatch")
    contradiction = next(row for row in safety if row["oracle_metadata"]["condition"] == "contradictory_definition")
    duplicate = next(row for row in safety if row["oracle_metadata"]["condition"] == "duplicate_lexeme")
    mutants = [
        {
            "mutant": "ignore_typed_signature",
            "killed": compile_agent_input(type_case["public"], "ignore_typed_signature")["status"] == "ok",
        },
        {
            "mutant": "swap_relation_roles",
            "killed": compile_agent_input(relation["public"], "swap_relation_roles").get("parse") != relation["target"]["parse"],
        },
        {
            "mutant": "drop_negative_form",
            "killed": compile_agent_input(negative["public"], "drop_negative_form").get("parse") != negative["target"]["parse"],
        },
        {
            "mutant": "treat_action_as_relation",
            "killed": compile_agent_input(action["public"], "treat_action_as_relation").get("parse") != action["target"]["parse"],
        },
        {
            "mutant": "ignore_definition_body",
            "killed": compile_agent_input(contradiction["public"], "ignore_definition_body")["status"] == "ok",
        },
        {
            "mutant": "accept_duplicate_lexeme",
            "killed": compile_agent_input(duplicate["public"], "accept_duplicate_lexeme")["status"] == "ok",
        },
    ]
    mutation_ok = all(row["killed"] for row in mutants)
    if not mutation_ok:
        errors.append("V57 implementation did not kill every registered mutant")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v57-implementation-lock.json", "configs/v57-population-seal.json",
            "configs/v57-outcome-lock.json", "data/v57-definition-augmented-ontology-transfer",
            "outputs/v57-definition-augmented-ontology-transfer/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V57 population or downstream artifact exists before implementation lock")

    audit = {
        "schema_version": 57,
        "experiment": "v57_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v57_implementation_lock" if not errors else "repair_v57_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "checks": {
            "design_lock_authorization_and_binding": design_ok,
            "target_oracle_and_model_firewall": boundary_ok,
            "altered_seed_full_population_and_definition_controls": altered_ok,
            "all_registered_mutants_killed": mutation_ok,
            "candidate_population_and_downstream_absent": downstream_absent,
        },
        "altered_seed_metrics": {
            "core_records": len(core), "core_exact_ast": core_passes,
            "compiled_truth": truth_passes, "safety_records": len(safety),
            "safety_passes": safety_passes,
            "no_definition_abstentions": no_definition_abstentions,
            "opaque_name_only_abstentions": opaque_only_abstentions,
            "shuffled_schema_exact_ast": shuffled_exact,
            "ontology_pack_counts": pack_counts,
            "concept_kind_counts": concept_kind_counts,
            "definition_template_counts": definition_template_counts,
        },
        "mutation_controls": {
            "registered": len(mutants), "kills": sum(row["killed"] for row in mutants),
            "kill_rate": sum(row["killed"] for row in mutants) / len(mutants),
            "records": mutants,
        },
        "data_access": {
            "v57_candidate_records_accessed": 0,
            "v57_candidate_evaluation_runs": 0,
            "altered_seed_records_generated": len(core) + len(safety),
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
