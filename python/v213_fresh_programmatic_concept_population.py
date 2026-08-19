from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Iterable

from v212_open_class_identifiability_oracle import (
    REPRESENTATION_ORDER,
    all_behavior_ids,
    behavior_bits,
    behavior_id,
    build_predictions,
    classify_behavior,
    equivalent_rewrite,
    evidence_status,
    expressibility_set,
    first_boundary_witness,
    language_catalog,
    rename_record,
    resolve_episode,
    reverse_commutative_order,
    reverse_evidence_order,
    shadow_action,
)


FAMILIES = (
    "EXACT_ALIAS",
    "EXISTING_COMPOSITION",
    "NEAR_ALIAS_BOUNDARY",
    "BROADER_NARROWER",
    "MISSING_OPERATOR",
    "IRREDUCIBLE_RELATIVE_TO_LANGUAGES",
    "REFERENCE_GROUNDED_SYMBOL",
    "GENUINELY_AMBIGUOUS",
    "CONTRADICTORY",
    "OUTSIDE_DESCRIPTION",
)
VARIANT_CODES = ("V0", "V1", "V2", "V3")


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _rank(values: Iterable[Any], *seed_parts: Any) -> list[Any]:
    return sorted(values, key=lambda value: (_digest(*seed_parts, value), str(value)))


def _cycle(values: list[Any], index: int, *seed_parts: Any) -> Any:
    ranked = _rank(values, *seed_parts)
    if not ranked:
        raise ValueError("V213 cannot select from an empty frozen pool")
    return deepcopy(ranked[index % len(ranked)])


def _opaque(prefix: str, salt: str, *parts: Any) -> str:
    return f"{prefix}-{_digest(salt, *parts)[:20]}"


def _observations(identifier: str, worlds: list[str], omitted: int | None = None) -> list[dict[str, Any]]:
    bits = behavior_bits(identifier)
    return [
        {"world": world, "output": int(bits[index])}
        for index, world in enumerate(worlds)
        if index != omitted
    ]


def _primitive_expression_for(identifier: str, semantics: dict[str, Any]) -> dict[str, Any]:
    for name, bits in semantics["registered_primitives"].items():
        if behavior_id(bits) == identifier:
            return {"op": "PRIMITIVE", "name": name}
    raise ValueError(f"V213 behavior is not a registered primitive: {identifier}")


def _expression_for(identifier: str, semantics: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    representation = classify_behavior(identifier, catalog)
    if representation == "EXISTING_PRIMITIVE":
        return _primitive_expression_for(identifier, semantics)
    if representation == "EXISTING_COMPOSITION":
        return deepcopy(catalog["base_programs_by_behavior"][identifier][0])
    if representation == "MISSING_OPERATOR":
        return deepcopy(catalog["extension_programs_by_behavior"][identifier][0])
    raise ValueError(f"V213 irreducible behavior has no frozen-language expression: {identifier}")


def _candidate_pools(catalog: dict[str, Any]) -> dict[str, list[str]]:
    all_ids = all_behavior_ids()
    return {
        representation: [identifier for identifier in all_ids if classify_behavior(identifier, catalog) == representation]
        for representation in REPRESENTATION_ORDER
    }


def _base_public() -> dict[str, Any]:
    return {
        "definition": {"kind": "UNCONSTRAINED"},
        "references": [],
        "reference_facts": [],
        "observations": [],
        "comparison_anchor": None,
    }


def _reference_for_behavior(reference_id: str, identifier: str, worlds: list[str]) -> dict[str, Any]:
    return {
        "reference_id": reference_id,
        "definition": {"kind": "UNCONSTRAINED"},
        "observations": _observations(identifier, worlds),
    }


def _flip_identifier(identifier: str, index: int) -> str:
    bits = behavior_bits(identifier)
    flipped = bits[:index] + ("1" if bits[index] == "0" else "0") + bits[index + 1 :]
    return behavior_id(flipped)


def _group_instruction(
    family: str,
    index: int,
    config: dict[str, Any],
    semantics: dict[str, Any],
    catalog: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    design = config["populationDesign"]
    seed = design["generatorSeed"]
    worlds = semantics["world_order"]
    pools = _candidate_pools(catalog)
    public = _base_public()
    target: str | None = None
    expected_candidates: list[str]
    comparison_relation = None
    comparison_world = None

    if family == "EXACT_ALIAS":
        target = _cycle(pools["EXISTING_PRIMITIVE"], index, seed, family)
        expression = _primitive_expression_for(target, semantics)
        for _ in range(1 + index % 3):
            expression = {"op": "IDENTITY", "arg": expression}
        public["definition"] = {"kind": "EXPRESSION", "expression": expression}
        expected_candidates = [target]
    elif family == "EXISTING_COMPOSITION":
        target = _cycle(pools["EXISTING_COMPOSITION"], index, seed, family)
        public["definition"] = {
            "kind": "EXPRESSION",
            "expression": _expression_for(target, semantics, catalog),
        }
        expected_candidates = [target]
    elif family == "NEAR_ALIAS_BOUNDARY":
        near_pairs = []
        for anchor in pools["EXISTING_PRIMITIVE"]:
            for world_index in range(len(worlds)):
                candidate = _flip_identifier(anchor, world_index)
                if classify_behavior(candidate, catalog) == "IRREDUCIBLE_PROVISIONAL":
                    near_pairs.append((anchor, candidate, world_index))
        anchor, target, world_index = _cycle(near_pairs, index, seed, family)
        public["observations"] = _observations(target, worlds)
        public["comparison_anchor"] = {
            "kind": "EXPRESSION",
            "expression": _primitive_expression_for(anchor, semantics),
        }
        comparison_relation = "NEAR_ALIAS_NOT_EQUAL"
        comparison_world = worlds[world_index]
        expected_candidates = [target]
    elif family == "BROADER_NARROWER":
        names = sorted(semantics["registered_primitives"])
        instructions = []
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                for op, relation in (("AND", "NARROWER_THAN_ANCHOR"), ("OR", "BROADER_THAN_ANCHOR")):
                    expression = {
                        "op": op,
                        "args": [
                            {"op": "PRIMITIVE", "name": left},
                            {"op": "PRIMITIVE", "name": right},
                        ],
                    }
                    target_id = behavior_id(
                        "".join(
                            str(
                                int(
                                    (a == "1" and b == "1")
                                    if op == "AND"
                                    else (a == "1" or b == "1")
                                )
                            )
                            for a, b in zip(
                                semantics["registered_primitives"][left],
                                semantics["registered_primitives"][right],
                            )
                        )
                    )
                    instructions.append((left, expression, target_id, relation))
        anchor_name, expression, target, comparison_relation = _cycle(instructions, index, seed, family)
        public["definition"] = {"kind": "EXPRESSION", "expression": expression}
        public["comparison_anchor"] = {
            "kind": "EXPRESSION",
            "expression": {"op": "PRIMITIVE", "name": anchor_name},
        }
        anchor_id = behavior_id(semantics["registered_primitives"][anchor_name])
        comparison_world = first_boundary_witness(target, anchor_id, worlds)["world"]
        expected_candidates = [target]
    elif family == "MISSING_OPERATOR":
        target = _cycle(pools["MISSING_OPERATOR"], index, seed, family)
        public["definition"] = {
            "kind": "EXPRESSION",
            "expression": _expression_for(target, semantics, catalog),
        }
        expected_candidates = [target]
    elif family == "IRREDUCIBLE_RELATIVE_TO_LANGUAGES":
        target = _cycle(pools["IRREDUCIBLE_PROVISIONAL"], index, seed, family)
        public["observations"] = _observations(target, worlds)
        expected_candidates = [target]
    elif family == "REFERENCE_GROUNDED_SYMBOL":
        representation = REPRESENTATION_ORDER[index % len(REPRESENTATION_ORDER)]
        target = _cycle(pools[representation], index // len(REPRESENTATION_ORDER), seed, family, representation)
        relation = "SAME_BEHAVIOR" if index % 2 == 0 else "COMPLEMENT_BEHAVIOR"
        reference_behavior = target if relation == "SAME_BEHAVIOR" else behavior_id(
            "".join("1" if bit == "0" else "0" for bit in behavior_bits(target))
        )
        symbol = f"symbol_{_digest(seed, family, index)[:12]}"
        reference_id = f"reference_{_digest(seed, family, index, 'reference')[:12]}"
        public["definition"] = {"kind": "SYMBOL", "name": symbol}
        public["references"] = [_reference_for_behavior(reference_id, reference_behavior, worlds)]
        public["reference_facts"] = [
            {"symbol": symbol, "relation": relation, "reference_id": reference_id}
        ]
        expected_candidates = [target]
    elif family == "GENUINELY_AMBIGUOUS":
        representation = REPRESENTATION_ORDER[index % len(REPRESENTATION_ORDER)]
        target = _cycle(pools[representation], index // len(REPRESENTATION_ORDER), seed, family, representation)
        held_out = int(_digest(seed, family, index, "held-out"), 16) % len(worlds)
        alternate = _flip_identifier(target, held_out)
        public["observations"] = _observations(target, worlds, held_out)
        expected_candidates = sorted([target, alternate])
    elif family == "CONTRADICTORY":
        mode = index % 3
        target = _cycle(
            pools["EXISTING_PRIMITIVE"] + pools["EXISTING_COMPOSITION"] + pools["MISSING_OPERATOR"],
            index,
            seed,
            family,
        )
        conflict_world = int(_digest(seed, family, index, "conflict"), 16) % len(worlds)
        if mode == 0:
            public["definition"] = {
                "kind": "EXPRESSION",
                "expression": _expression_for(target, semantics, catalog),
            }
            public["observations"] = [
                {
                    "world": worlds[conflict_world],
                    "output": 1 - int(behavior_bits(target)[conflict_world]),
                }
            ]
        elif mode == 1:
            symbol = f"symbol_{_digest(seed, family, index)[:12]}"
            reference_id = f"reference_{_digest(seed, family, index, 'reference')[:12]}"
            public["definition"] = {"kind": "SYMBOL", "name": symbol}
            public["references"] = [_reference_for_behavior(reference_id, target, worlds)]
            public["reference_facts"] = [
                {"symbol": symbol, "relation": relation, "reference_id": reference_id}
                for relation in ("SAME_BEHAVIOR", "COMPLEMENT_BEHAVIOR")
            ]
        else:
            public["observations"] = [
                {"world": worlds[conflict_world], "output": 0},
                {"world": worlds[conflict_world], "output": 1},
            ]
        expected_candidates = []
    elif family == "OUTSIDE_DESCRIPTION":
        token = f"outside_{_digest(design['outsideTokenSalt'], index)[:16]}"
        public["definition"] = {"kind": "OUTSIDE_DESCRIPTION", "token": token}
        expected_candidates = all_behavior_ids()
    else:
        raise ValueError(f"V213 unknown family: {family}")

    interface_status = public["definition"]["kind"]
    truth = {
        "concept_family": family,
        "target_behavior_id": target,
        "expected_candidate_ids": expected_candidates,
        "expected_evidence_status": evidence_status(expected_candidates),
        "expected_expressibility_set": expressibility_set(expected_candidates, catalog),
        "expected_shadow_action": shadow_action(expected_candidates, interface_status, catalog, semantics),
        "comparison_relation": comparison_relation,
        "expected_comparison_boundary_world": comparison_world,
    }
    return public, truth


def _rotate(values: list[Any], offset: int) -> list[Any]:
    if not values:
        return values
    offset %= len(values)
    return values[offset:] + values[:offset]


def _variant_public(base: dict[str, Any], code: str, group_id: str) -> dict[str, Any]:
    value = deepcopy(base)
    if code == "V0":
        return value
    if code == "V1":
        value = equivalent_rewrite(rename_record(value))
        if value["definition"]["kind"] == "OUTSIDE_DESCRIPTION":
            value["definition"]["token"] = f"outside_alias_{_digest(group_id, code)[:12]}"
        value["observations"] = _rotate(value["observations"], 1)
        for reference in value["references"]:
            reference["observations"] = _rotate(reference["observations"], 1)
        return value
    if code == "V2":
        value = reverse_commutative_order(reverse_evidence_order(value))
        if value["definition"]["kind"] == "OUTSIDE_DESCRIPTION":
            value["definition"]["token"] = f"outside_alias_{_digest(group_id, code)[:12]}"
        return value
    if code == "V3":
        value = equivalent_rewrite(equivalent_rewrite(reverse_commutative_order(value)))
        value["observations"] = _rotate(value["observations"], 2)
        for reference in value["references"]:
            reference["observations"] = _rotate(reference["observations"], 2)
        if value["definition"]["kind"] == "SYMBOL":
            old = value["definition"]["name"]
            new = f"symbol_alias_{_digest(group_id, code)[:12]}"
            value["definition"]["name"] = new
            for fact in value["reference_facts"]:
                if fact["symbol"] == old:
                    fact["symbol"] = new
        if value["definition"]["kind"] == "OUTSIDE_DESCRIPTION":
            value["definition"]["token"] = f"outside_alias_{_digest(group_id, code)[:12]}"
        return value
    raise ValueError(f"V213 unknown variant code: {code}")


def project_public_blueprints(blueprints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(row["public_record"]) for row in sorted(blueprints, key=lambda row: row["public_record"]["case_id"])]


def generate_population(
    config: dict[str, Any], semantics: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    design = config["populationDesign"]
    split_design = config["splitDesign"]
    catalog = language_catalog(semantics)
    group_rows: list[tuple[str, int, str]] = []
    for family in FAMILIES:
        for index in range(design["groupsPerFamily"]):
            group_id = _opaque("group", design["groupIdentifierSalt"], family, index)
            group_rows.append((family, index, group_id))
    split_by_group: dict[str, str] = {}
    for family in FAMILIES:
        family_groups = [group_id for name, _, group_id in group_rows if name == family]
        ranked = _rank(family_groups, design["splitSeed"], family)
        development = set(ranked[: split_design["developmentGroupsPerFamily"]])
        for group_id in family_groups:
            split_by_group[group_id] = "development" if group_id in development else "protected"

    blueprints: list[dict[str, Any]] = []
    truth_records: list[dict[str, Any]] = []
    for family, index, group_id in group_rows:
        base_public, base_truth = _group_instruction(family, index, config, semantics, catalog)
        split = split_by_group[group_id]
        for code in VARIANT_CODES:
            case_id = _opaque("case", design["caseIdentifierSalt"], group_id, code)
            public = _variant_public(base_public, code, group_id)
            public.update(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "split": split,
                    "variant_code": code,
                }
            )
            ordered_public = {
                field: public[field] for field in config["roleSeparation"]["publicFields"]
            }
            blueprints.append(
                {
                    "blueprint_index": len(blueprints),
                    "public_record": ordered_public,
                }
            )
            truth_records.append(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "split": split,
                    "variant_code": code,
                    **deepcopy(base_truth),
                }
            )
    blueprints.sort(key=lambda row: row["public_record"]["case_id"])
    truth_records.sort(key=lambda row: row["case_id"])
    split = {
        "schema_version": "213-group-split",
        "unit": "opaque_concept_group",
        "development_group_ids": sorted(group_id for group_id, name in split_by_group.items() if name == "development"),
        "protected_group_ids": sorted(group_id for group_id, name in split_by_group.items() if name == "protected"),
    }
    manifest = {
        "schema_version": "213-blueprint-generator-manifest",
        "experiment": config["experiment"],
        "generator_seed": design["generatorSeed"],
        "group_identifier_salt": design["groupIdentifierSalt"],
        "case_identifier_salt": design["caseIdentifierSalt"],
        "split_seed": design["splitSeed"],
        "family_count": len(FAMILIES),
        "group_count": len(group_rows),
        "record_count": len(blueprints),
        "public_blueprint_contains_hidden_truth": False,
        "natural_language_surface_record_count": 0,
        "external_ontology_payload_read_count": 0,
    }
    return blueprints, truth_records, split, manifest


def _recursive_key_count(value: Any, forbidden: set[str]) -> int:
    if isinstance(value, dict):
        return sum(key in forbidden for key in value) + sum(
            _recursive_key_count(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_recursive_key_count(item, forbidden) for item in value)
    return 0


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def score_population(
    blueprints: list[dict[str, Any]],
    public_records: list[dict[str, Any]],
    truth_records: list[dict[str, Any]],
    split: dict[str, Any],
    semantics: dict[str, Any],
    parent_public_records: list[dict[str, Any]],
    projection_freeze: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    public = {row["case_id"]: row for row in public_records}
    truth = {row["case_id"]: row for row in truth_records}
    predictions = {row["case_id"]: row for row in build_predictions(public_records, semantics)}
    if not (set(public) == set(truth) == set(predictions)):
        raise ValueError("V213 public/truth/prediction identifier mismatch")
    rows = [(public[key], truth[key], predictions[key]) for key in sorted(public)]
    group_ids = {hidden["group_id"] for _, hidden, _ in rows}
    family_group_sets = {
        family: {hidden["group_id"] for _, hidden, _ in rows if hidden["concept_family"] == family}
        for family in FAMILIES
    }
    family_record_counts = {
        family: sum(hidden["concept_family"] == family for _, hidden, _ in rows)
        for family in FAMILIES
    }
    split_group_sets = {
        name: {hidden["group_id"] for _, hidden, _ in rows if hidden["split"] == name}
        for name in ("development", "protected")
    }
    split_record_counts = {
        name: sum(hidden["split"] == name for _, hidden, _ in rows)
        for name in ("development", "protected")
    }
    family_split_group_counts = {
        family: {
            name: len({hidden["group_id"] for _, hidden, _ in rows if hidden["concept_family"] == family and hidden["split"] == name})
            for name in ("development", "protected")
        }
        for family in FAMILIES
    }
    variants_by_group = {
        group_id: {hidden["variant_code"] for _, hidden, _ in rows if hidden["group_id"] == group_id}
        for group_id in group_ids
    }
    exact_candidate = [prediction["candidate_ids"] == hidden["expected_candidate_ids"] for _, hidden, prediction in rows]
    exact_status = [prediction["evidence_status"] == hidden["expected_evidence_status"] for _, hidden, prediction in rows]
    exact_expressibility = [prediction["expressibility_set"] == hidden["expected_expressibility_set"] for _, hidden, prediction in rows]
    exact_action = [prediction["shadow_action"] == hidden["expected_shadow_action"] for _, hidden, prediction in rows]
    semantic_fields = (
        "concept_family",
        "target_behavior_id",
        "expected_candidate_ids",
        "expected_evidence_status",
        "expected_expressibility_set",
        "expected_shadow_action",
        "comparison_relation",
        "expected_comparison_boundary_world",
        "split",
    )
    group_consistency = []
    variant_invariance = []
    for group_id in sorted(group_ids):
        group_rows = [(surface, hidden, prediction) for surface, hidden, prediction in rows if hidden["group_id"] == group_id]
        truth_signatures = [{field: hidden[field] for field in semantic_fields} for _, hidden, _ in group_rows]
        prediction_signatures = [
            {
                "candidate_ids": prediction["candidate_ids"],
                "evidence_status": prediction["evidence_status"],
                "expressibility_set": prediction["expressibility_set"],
                "shadow_action": prediction["shadow_action"],
            }
            for _, _, prediction in group_rows
        ]
        group_consistency.append(all(signature == truth_signatures[0] for signature in truth_signatures))
        variant_invariance.append(all(signature == prediction_signatures[0] for signature in prediction_signatures))
    pair_count = sum(prediction["candidate_pair_count"] for _, _, prediction in rows)
    witnessed_count = sum(prediction["witnessed_pair_count"] for _, _, prediction in rows)
    comparison_rows = [row for row in rows if row[1]["comparison_relation"] is not None]
    comparison_checks = [
        prediction["comparison_anchor_boundary_witness"] is not None
        and prediction["comparison_anchor_boundary_witness"]["world"] == hidden["expected_comparison_boundary_world"]
        for _, hidden, prediction in comparison_rows
    ]
    forbidden = set(config["roleSeparation"]["publicProjectionForbiddenTokens"])
    public_hidden_leakage = _recursive_key_count(blueprints, forbidden) + _recursive_key_count(public_records, forbidden)
    serialized_public = json.dumps([blueprints, public_records], sort_keys=True)
    forbidden_token_count = sum(serialized_public.count(token) for token in forbidden)
    public_fields = set(config["roleSeparation"]["publicFields"])
    public_schema_exact = all(set(row) == public_fields for row in public_records)
    projected = project_public_blueprints(blueprints)
    blueprint_accuracy = _rate(
        [left == right for left, right in zip(projected, sorted(public_records, key=lambda row: row["case_id"]))]
    ) if len(projected) == len(public_records) else 0.0
    case_ids = [hidden["case_id"] for _, hidden, _ in rows]
    parent_ids = {row["case_id"] for row in parent_public_records}
    opaque_checks = [
        re.fullmatch(r"case-[0-9a-f]{20}", hidden["case_id"]) is not None
        and re.fullmatch(r"group-[0-9a-f]{20}", hidden["group_id"]) is not None
        for _, hidden, _ in rows
    ]
    provenance_fields = (
        "design_lock_sha256",
        "parent_outcome_lock_sha256",
        "parent_public_semantics_sha256",
        "generator_manifest_sha256",
        "public_blueprints_sha256",
        "sealed_truth_sha256",
        "split_sha256",
        "public_records_sha256",
    )
    metrics = {
        "group_count": len(group_ids),
        "record_count": len(rows),
        "family_count": len(family_group_sets),
        "family_group_counts": {family: len(values) for family, values in family_group_sets.items()},
        "family_record_counts": family_record_counts,
        "split_group_counts": {name: len(values) for name, values in split_group_sets.items()},
        "split_record_counts": split_record_counts,
        "family_split_group_counts": family_split_group_counts,
        "variant_completeness_rate": _rate([values == set(VARIANT_CODES) for values in variants_by_group.values()]),
        "cross_split_group_overlap_count": len(split_group_sets["development"] & split_group_sets["protected"]),
        "split_artifact_exact": bool(
            set(split["development_group_ids"]) == split_group_sets["development"]
            and set(split["protected_group_ids"]) == split_group_sets["protected"]
        ),
        "parent_identifier_reuse_count": len(set(case_ids) & parent_ids),
        "duplicate_case_identifier_count": len(case_ids) - len(set(case_ids)),
        "duplicate_group_identifier_count": config["populationDesign"]["groupCount"] - len(group_ids),
        "nonopaque_identifier_count": sum(not value for value in opaque_checks),
        "public_schema_exact": public_schema_exact,
        "public_hidden_field_leakage_count": public_hidden_leakage,
        "public_forbidden_token_count": forbidden_token_count,
        "exact_candidate_set_accuracy": _rate(exact_candidate),
        "evidence_status_accuracy": _rate(exact_status),
        "expressibility_set_accuracy": _rate(exact_expressibility),
        "shadow_action_accuracy": _rate(exact_action),
        "within_group_semantic_consistency": _rate(group_consistency),
        "variant_resolution_invariance": _rate(variant_invariance),
        "candidate_pair_count": pair_count,
        "witnessed_candidate_pair_count": witnessed_count,
        "distinct_pair_boundary_witness_coverage": witnessed_count / pair_count if pair_count else 1.0,
        "comparison_anchor_boundary_witness_rate": _rate(comparison_checks),
        "public_projection_freeze_before_truth_join": bool(
            projection_freeze["public_projection_frozen_before_truth_join"]
            and not projection_freeze["sealed_truth_joined_before_public_projection_freeze"]
        ),
        "blueprint_reconstruction_accuracy": blueprint_accuracy,
        "provenance_hash_coverage": _rate([field in projection_freeze and len(projection_freeze[field]) == 64 for field in provenance_fields]),
    }
    metrics["finite_metrics"] = _finite(metrics)
    return metrics


def audit_population(
    metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["populationGates"]
    access_gates = config["accessGates"]
    checks = {
        "population_family_and_variant_counts_exact": bool(
            metrics["group_count"] == gates["requiredGroupCount"]
            and metrics["record_count"] == gates["requiredRecordCount"]
            and metrics["family_count"] == gates["requiredFamilyCount"]
            and all(value == gates["requiredGroupsPerFamily"] for value in metrics["family_group_counts"].values())
            and all(value == gates["requiredRecordsPerFamily"] for value in metrics["family_record_counts"].values())
            and metrics["variant_completeness_rate"] == 1.0
        ),
        "balanced_group_disjoint_split_exact": bool(
            metrics["split_group_counts"]["development"] == gates["requiredDevelopmentGroupCount"]
            and metrics["split_group_counts"]["protected"] == gates["requiredProtectedGroupCount"]
            and metrics["split_record_counts"]["development"] == gates["requiredDevelopmentRecordCount"]
            and metrics["split_record_counts"]["protected"] == gates["requiredProtectedRecordCount"]
            and all(value["development"] == gates["requiredDevelopmentGroupsPerFamily"] for value in metrics["family_split_group_counts"].values())
            and all(value["protected"] == gates["requiredProtectedGroupsPerFamily"] for value in metrics["family_split_group_counts"].values())
            and metrics["cross_split_group_overlap_count"] <= gates["maximumCrossSplitGroupOverlapCount"]
            and metrics["split_artifact_exact"]
        ),
        "identifiers_fresh_unique_and_opaque": bool(
            metrics["parent_identifier_reuse_count"] <= gates["maximumParentIdentifierReuseCount"]
            and metrics["duplicate_case_identifier_count"] <= gates["maximumDuplicateCaseIdentifierCount"]
            and metrics["duplicate_group_identifier_count"] <= gates["maximumDuplicateGroupIdentifierCount"]
            and metrics["nonopaque_identifier_count"] <= gates["maximumNonOpaqueIdentifierCount"]
        ),
        "public_projection_schema_and_nonleakage_exact": bool(
            metrics["public_schema_exact"]
            and metrics["public_hidden_field_leakage_count"] <= gates["maximumPublicHiddenFieldLeakageCount"]
            and metrics["public_forbidden_token_count"] <= gates["maximumPublicForbiddenTokenCount"]
        ),
        "oracle_reconstruction_and_group_invariance_exact": bool(
            metrics["exact_candidate_set_accuracy"] == gates["requiredExactCandidateSetAccuracy"]
            and metrics["evidence_status_accuracy"] == gates["requiredEvidenceStatusAccuracy"]
            and metrics["expressibility_set_accuracy"] == gates["requiredExpressibilitySetAccuracy"]
            and metrics["shadow_action_accuracy"] == gates["requiredShadowActionAccuracy"]
            and metrics["within_group_semantic_consistency"] == gates["requiredWithinGroupSemanticConsistency"]
            and metrics["variant_resolution_invariance"] == gates["requiredVariantResolutionInvariance"]
        ),
        "boundary_witnesses_exact": bool(
            metrics["distinct_pair_boundary_witness_coverage"] == gates["requiredDistinctPairBoundaryWitnessCoverage"]
            and metrics["comparison_anchor_boundary_witness_rate"] == gates["requiredComparisonAnchorBoundaryWitnessRate"]
        ),
        "freeze_reconstruction_and_provenance_exact": bool(
            metrics["public_projection_freeze_before_truth_join"] == gates["requiredPublicProjectionFreezeBeforeTruthJoin"]
            and metrics["blueprint_reconstruction_accuracy"] == gates["requiredBlueprintReconstructionAccuracy"]
            and metrics["provenance_hash_coverage"] == gates["requiredProvenanceHashCoverage"]
        ),
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    access_checks = {
        "one_blueprint_projection_and_structural_verification": bool(
            access["blueprint_generation_run_count"] == access_gates["requiredBlueprintGenerationRunCount"]
            and access["public_projection_run_count"] == access_gates["requiredPublicProjectionRunCount"]
            and access["structural_verification_run_count"] == access_gates["requiredStructuralVerificationRunCount"]
        ),
        "all_unauthorized_access_and_effect_counts_zero": bool(
            access["natural_language_surface_read_count"] <= access_gates["maximumNaturalLanguageSurfaceReadCount"]
            and access["external_ontology_payload_read_count"] <= access_gates["maximumExternalOntologyPayloadReadCount"]
            and access["protected_downstream_evaluation_count"] <= access_gates["maximumProtectedDownstreamEvaluationCount"]
            and access["model_load_count"] <= access_gates["maximumModelLoadCount"]
            and access["model_generation_count"] <= access_gates["maximumModelGenerationCount"]
            and access["api_call_count"] <= access_gates["maximumAPICallCount"]
            and access["training_run_count"] <= access_gates["maximumTrainingRunCount"]
            and access["ontology_registration_count"] <= access_gates["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= access_gates["maximumTrustedStateMutationCount"]
            and access["service_call_count"] <= access_gates["maximumServiceCallCount"]
            and access["external_side_effect_count"] <= access_gates["maximumExternalSideEffectCount"]
            and access["actual_execution_count"] <= access_gates["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    return {
        "passed": passed,
        "branch": "V214_DETERMINISTIC_CONTROL_DESIGN_ELIGIBLE" if passed else "NEGATIVE_PROGRAMMATIC_POPULATION",
        "decision": config["decisionRule"]["ifEveryIntegrityPopulationAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }
