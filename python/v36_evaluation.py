"""Metrics, gates, bootstrap, and localization for V36 confirmation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from v32_language import compile_truth


def score_confirmation(
    rows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
    v32_config: dict[str, Any], bootstrap_seed: int, bootstrap_replicates: int,
) -> dict[str, Any]:
    lookup = {row["id"]: row for row in predictions}
    if set(lookup) != {row["id"] for row in rows}:
        raise ValueError("V36 predictions do not exactly cover confirmation records")
    details, scenes, families = [], defaultdict(list), defaultdict(list)
    for row in rows:
        prediction, target = lookup[row["id"]], row["target"]
        fields, factors = prediction["selected_fields"], prediction["selected_intermediates"]
        expected2 = target["arguments"][1] if target["predicate_kind"] == "relation" else "N/A"
        predicate = fields["predicate"] == target["predicate"]
        argument1 = fields["argument_1"] == target["arguments"][0]
        argument2 = fields["argument_2"] == expected2
        atom = predicate and argument1 and argument2
        sign = factors["lexical_sign"] == target["factorization"]["lexical_sign"]
        operation = factors["outer_operation"] == target["factorization"]["outer_operation"]
        compiled = compile_truth(factors["lexical_sign"], factors["outer_operation"], v32_config)
        truth = compiled == target["truth_status"]
        exact = atom and truth
        detail = {
            "id": row["id"], "scene_id": row["scene_id"],
            "surface_family": row["oracle_metadata"]["surface_family"],
            "predicate_correct": predicate, "argument1_correct": argument1,
            "argument2_correct": argument2, "atom_correct": atom,
            "relation": target["predicate_kind"] == "relation",
            "relation_order_correct": argument1 and argument2,
            "lexical_sign_correct": sign, "outer_operation_correct": operation,
            "compiled_truth_correct": truth, "exact_fact_correct": exact,
            "negative_composition": target["factorization"]["lexical_sign"] == "negative"
            and target["factorization"]["outer_operation"] in ("deny", "double_deny", "contrast_select"),
        }
        details.append(detail); scenes[row["scene_id"]].append(detail); families[detail["surface_family"]].append(detail)
    mean = lambda values: float(np.mean(values)) if values else 0.0
    relations = [row for row in details if row["relation"]]
    negative = [row for row in details if row["negative_composition"]]
    family_metrics = {
        family: {"records": len(values), "exact_fact_accuracy": mean([row["exact_fact_correct"] for row in values])}
        for family, values in sorted(families.items())
    }
    row_by_id = {row["id"]: row for row in details}
    pair_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source in rows:
        for pair in source["oracle_metadata"]["pairs"]:
            pair_members[(pair["kind"], pair["id"])].append(source["id"])
    pairs_by_kind: dict[str, list[bool]] = defaultdict(list)
    for (kind, _), identifiers in pair_members.items():
        pairs_by_kind[kind].append(all(row_by_id[identifier]["exact_fact_correct"] for identifier in identifiers))
    pair_metrics = {
        kind: {"pairs": len(values), "pair_exact_accuracy": mean(values)}
        for kind, values in sorted(pairs_by_kind.items())
    }
    family_values = np.asarray([value["exact_fact_accuracy"] for value in family_metrics.values()], dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.mean(rng.choice(family_values, size=(bootstrap_replicates, len(family_values)), replace=True), axis=1)
    return {
        "records": len(details), "scenes": len(scenes), "surface_families": len(families),
        "predicate_accuracy": mean([row["predicate_correct"] for row in details]),
        "argument1_accuracy": mean([row["argument1_correct"] for row in details]),
        "argument2_accuracy": mean([row["argument2_correct"] for row in details]),
        "atom_exact_accuracy": mean([row["atom_correct"] for row in details]),
        "relation_order_accuracy": mean([row["relation_order_correct"] for row in relations]),
        "lexical_sign_accuracy": mean([row["lexical_sign_correct"] for row in details]),
        "outer_operation_accuracy": mean([row["outer_operation_correct"] for row in details]),
        "compiled_truth_accuracy": mean([row["compiled_truth_correct"] for row in details]),
        "compiled_exact_fact_accuracy": mean([row["exact_fact_correct"] for row in details]),
        "exact_scene_accuracy": mean([all(row["exact_fact_correct"] for row in values) for values in scenes.values()]),
        "worst_surface_family_exact_fact": min(value["exact_fact_accuracy"] for value in family_metrics.values()),
        "negative_composition_exact_fact": mean([row["exact_fact_correct"] for row in negative]),
        "by_surface_family": family_metrics, "by_pair_kind": pair_metrics,
        "family_bootstrap_exact_fact_95_interval": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def gate_checks(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["gates"]
    checks = {
        "predicate": metrics["predicate_accuracy"] >= gates["minimumPredicateAccuracy"],
        "atom": metrics["atom_exact_accuracy"] >= gates["minimumAtomExactAccuracy"],
        "relation_order": metrics["relation_order_accuracy"] >= gates["minimumRelationOrderAccuracy"],
        "lexical_sign": metrics["lexical_sign_accuracy"] >= gates["minimumLexicalSignAccuracy"],
        "outer_operation": metrics["outer_operation_accuracy"] >= gates["minimumOuterOperationAccuracy"],
        "compiled_truth": metrics["compiled_truth_accuracy"] >= gates["minimumCompiledTruthAccuracy"],
        "compiled_exact_fact": metrics["compiled_exact_fact_accuracy"] >= gates["minimumCompiledExactFactAccuracy"],
        "exact_scene": metrics["exact_scene_accuracy"] >= gates["minimumExactSceneAccuracy"],
        "worst_surface_family": metrics["worst_surface_family_exact_fact"] >= gates["minimumWorstSurfaceFamilyExactFact"],
        "negative_composition": metrics["negative_composition_exact_fact"] >= gates["minimumCompositionNegativeExactFact"],
    }
    checks.update({
        f"pair_{kind}": value["pair_exact_accuracy"] >= gates["minimumPairExact"]
        for kind, value in metrics["by_pair_kind"].items()
    })
    return checks


def decision_from_checks(metrics: dict[str, Any], checks: dict[str, bool]) -> tuple[str, str]:
    if all(checks.values()):
        magnitude = "near_v35_replication" if metrics["compiled_exact_fact_accuracy"] >= 0.97 else "gate_level_confirmation"
        return "confirmation_pass_preregister_end_to_end_relational_suite", magnitude
    semantic = not checks["lexical_sign"] or not checks["outer_operation"]
    atom = not checks["predicate"] or not checks["atom"] or not checks["relation_order"]
    if semantic and not atom:
        return "confirmation_fail_reopen_semantic_interface_only", "semantic_representation_failure"
    if atom and not semantic:
        return "confirmation_fail_reopen_atom_interface_only", "atom_binding_failure"
    component_names = ("predicate", "atom", "relation_order", "lexical_sign", "outer_operation", "compiled_truth")
    if all(checks[name] for name in component_names):
        return "confirmation_fail_investigate_assembly_dependence", "assembly_dependence_failure"
    return "confirmation_fail_broad_prompt_factorized_transfer", "broad_component_failure"
