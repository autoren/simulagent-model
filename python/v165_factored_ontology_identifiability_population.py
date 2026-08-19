from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import re
from typing import Any


EXPRESSIBILITY_CLASSES = ("alias", "composition", "provisional_primitive")
EVIDENCE_STATUSES = ("sufficient", "ambiguous", "contradictory")


def valuations() -> list[tuple[bool, bool, bool]]:
    return [
        (bool(index & 4), bool(index & 2), bool(index & 1))
        for index in range(8)
    ]


def table_string(values: list[bool] | tuple[bool, ...]) -> str:
    if len(values) != 8:
        raise ValueError("truth table must have eight values")
    return "".join("1" if value else "0" for value in values)


def candidate_id(table: str) -> str:
    if len(table) != 8 or set(table) - {"0", "1"}:
        raise ValueError("invalid truth table")
    return f"C{int(table, 2):03d}"


def registered_tables() -> dict[str, dict[str, Any]]:
    domain = valuations()
    output: dict[str, dict[str, Any]] = {}
    for atom in range(3):
        table = table_string([row[atom] for row in domain])
        output[table] = {
            "expressibility_class": "alias",
            "form": "ATOM",
            "atoms": [atom],
        }
    for left, right in combinations(range(3), 2):
        for operator in ("AND", "OR"):
            table = table_string(
                [
                    (row[left] and row[right])
                    if operator == "AND"
                    else (row[left] or row[right])
                    for row in domain
                ]
            )
            output[table] = {
                "expressibility_class": "composition",
                "form": operator,
                "atoms": [left, right],
            }
    if len(output) != 9:
        raise AssertionError("registered DSL truth tables are not unique")
    return output


def candidate_universe() -> list[dict[str, Any]]:
    registered = registered_tables()
    rows = []
    for value in range(256):
        table = f"{value:08b}"
        metadata = registered.get(table)
        rows.append(
            {
                "candidate_id": candidate_id(table),
                "truth_table": table,
                "expressibility_class": (
                    metadata["expressibility_class"]
                    if metadata
                    else "provisional_primitive"
                ),
                "registered_form": metadata["form"] if metadata else None,
                "registered_atoms": metadata["atoms"] if metadata else None,
            }
        )
    return rows


def build_frozen_ontology(config: dict[str, Any]) -> dict[str, Any]:
    registered = registered_tables()
    namespaces = []
    for namespace in config["primitiveNamespaces"]:
        formulas = []
        for table, metadata in sorted(registered.items()):
            atoms = [namespace["primitive_names"][index] for index in metadata["atoms"]]
            formulas.append(
                {
                    "candidate_id": candidate_id(table),
                    "truth_table": table,
                    "form": metadata["form"],
                    "atoms": atoms,
                    "expressibility_class": metadata["expressibility_class"],
                }
            )
        namespaces.append(
            {
                "namespace_id": namespace["namespace_id"],
                "entity_type": config["typedDSL"]["entityType"],
                "entity_noun": namespace["entity_noun"],
                "primitives": [
                    {
                        "name": name,
                        "arity": config["typedDSL"]["primitiveArity"],
                        "value_type": config["typedDSL"]["primitiveValueType"],
                    }
                    for name in namespace["primitive_names"]
                ],
                "registered_formulas": formulas,
            }
        )
    return {
        "schema_version": "165-frozen-finite-typed-ontology",
        "entity_type": config["typedDSL"]["entityType"],
        "valuation_order": [list(row) for row in valuations()],
        "registered_forms": config["typedDSL"]["registeredForms"],
        "maximum_composition_depth": config["typedDSL"]["maximumCompositionDepth"],
        "namespaces": namespaces,
        "authoritative_ontology_immutable": True,
        "provisional_registration_allowed": False,
    }


def _definition(
    expressibility: str,
    evidence_status: str,
    table: str,
    concept_name: str,
    namespace: dict[str, Any],
    config: dict[str, Any],
) -> str:
    grammar = config["definitionGrammar"]
    if evidence_status == "ambiguous":
        return grammar["underspecifiedTemplate"].format(concept_name=concept_name)
    if evidence_status == "contradictory":
        return grammar["contradictoryTemplate"].format(concept_name=concept_name)
    metadata = registered_tables().get(table)
    if expressibility == "alias" and metadata:
        return grammar["aliasTemplate"].format(
            entity_noun=namespace["entity_noun"],
            concept_name=concept_name,
            atom=namespace["primitive_names"][metadata["atoms"][0]],
        )
    if expressibility == "composition" and metadata:
        key = "andTemplate" if metadata["form"] == "AND" else "orTemplate"
        return grammar[key].format(
            entity_noun=namespace["entity_noun"],
            concept_name=concept_name,
            left_atom=namespace["primitive_names"][metadata["atoms"][0]],
            right_atom=namespace["primitive_names"][metadata["atoms"][1]],
        )
    return grammar["tableTemplate"].format(concept_name=concept_name)


def parse_definition(
    definition: str, namespace: dict[str, Any], concept_name: str
) -> dict[str, Any]:
    entity = re.escape(namespace["entity_noun"])
    concept = re.escape(concept_name)
    atom_pattern = "(" + "|".join(
        re.escape(value) for value in namespace["primitive_names"]
    ) + ")"
    alias = re.fullmatch(
        rf"A {entity} is {concept} exactly when it is {atom_pattern}\.", definition
    )
    if alias:
        atom = namespace["primitive_names"].index(alias.group(1))
        table = table_string([row[atom] for row in valuations()])
        return {"parse_kind": "exact_registered_expression", "candidate_id": candidate_id(table)}
    conjunction = re.fullmatch(
        rf"A {entity} is {concept} exactly when it is both {atom_pattern} and {atom_pattern}\.",
        definition,
    )
    disjunction = re.fullmatch(
        rf"A {entity} is {concept} exactly when it is either {atom_pattern} or {atom_pattern} or both\.",
        definition,
    )
    match = conjunction or disjunction
    if match:
        left = namespace["primitive_names"].index(match.group(1))
        right = namespace["primitive_names"].index(match.group(2))
        if left == right:
            return {"parse_kind": "invalid_registered_expression", "candidate_id": None}
        operator = "AND" if conjunction else "OR"
        table = table_string(
            [
                (row[left] and row[right])
                if operator == "AND"
                else (row[left] or row[right])
                for row in valuations()
            ]
        )
        return {"parse_kind": "exact_registered_expression", "candidate_id": candidate_id(table)}
    if definition.startswith("The rule for "):
        return {"parse_kind": "complete_table_required", "candidate_id": None}
    if definition.startswith("The available evidence constrains "):
        return {"parse_kind": "underspecified", "candidate_id": None}
    if definition.startswith("The submitted evidence for "):
        return {"parse_kind": "consistency_check", "candidate_id": None}
    return {"parse_kind": "unparsed", "candidate_id": None}


def _observation(
    ordinal: int,
    valuation_index: int,
    observed: bool,
    namespace: dict[str, Any],
) -> dict[str, Any]:
    assignment = {
        name: valuations()[valuation_index][index]
        for index, name in enumerate(namespace["primitive_names"])
    }
    return {
        "observation_id": f"E{ordinal + 1:02d}",
        "kind": "example" if observed else "counterexample",
        "intervention": {
            "entity_type": "Device",
            "set_primitives": assignment,
        },
        "observed_qualifies": observed,
    }


def _valuation_index(observation: dict[str, Any], namespace: dict[str, Any]) -> int:
    assignment = observation["intervention"]["set_primitives"]
    bits = tuple(bool(assignment[name]) for name in namespace["primitive_names"])
    return valuations().index(bits)


def enumerate_version_space(
    definition_parse: dict[str, Any],
    observations: list[dict[str, Any]],
    namespace: dict[str, Any],
    universe: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates = candidate_universe() if universe is None else universe
    exact = definition_parse.get("candidate_id")
    if definition_parse["parse_kind"] == "invalid_registered_expression":
        return []
    if exact is not None:
        candidates = [row for row in candidates if row["candidate_id"] == exact]
    constraints: dict[int, bool] = {}
    for observation in observations:
        index = _valuation_index(observation, namespace)
        value = bool(observation["observed_qualifies"])
        if index in constraints and constraints[index] != value:
            return []
        constraints[index] = value
    return [
        row
        for row in candidates
        if all((row["truth_table"][index] == "1") == value for index, value in constraints.items())
    ]


def _balanced_consistent_indices(table: str) -> list[int]:
    positives = [index for index, value in enumerate(table) if value == "1"]
    negatives = [index for index, value in enumerate(table) if value == "0"]
    return positives[:2] + negatives[:2]


def _ambiguous_indices(table: str, namespace: dict[str, Any]) -> list[int]:
    universe = candidate_universe()
    target = candidate_id(table)
    for count in (2, 3, 4):
        for indices in combinations(range(8), count):
            labels = {table[index] for index in indices}
            if labels != {"0", "1"}:
                continue
            observations = [
                _observation(ordinal, index, table[index] == "1", namespace)
                for ordinal, index in enumerate(indices)
            ]
            version = enumerate_version_space(
                {"parse_kind": "underspecified", "candidate_id": None},
                observations,
                namespace,
                universe,
            )
            classes = {row["expressibility_class"] for row in version}
            if target in {row["candidate_id"] for row in version} and classes == set(
                EXPRESSIBILITY_CLASSES
            ):
                return list(indices)
    raise ValueError("unable to construct cross-class ambiguous evidence")


def _target_tables() -> dict[str, list[str]]:
    registered = registered_tables()
    aliases = sorted(
        table for table, row in registered.items() if row["expressibility_class"] == "alias"
    )
    compositions = sorted(
        table
        for table, row in registered.items()
        if row["expressibility_class"] == "composition"
    )
    provisional = [
        row["truth_table"]
        for row in candidate_universe()
        if row["expressibility_class"] == "provisional_primitive"
        and 2 <= row["truth_table"].count("1") <= 6
    ]
    return {
        "alias": [aliases[index % len(aliases)] for index in range(4)],
        "composition": compositions[:4],
        "provisional_primitive": provisional[:4],
    }


def _record_id(cell: str, group: str, namespace_id: str) -> str:
    digest = hashlib.sha256(f"{cell}|{group}|{namespace_id}".encode()).hexdigest()[:16]
    return f"v165-{digest}"


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    universe = candidate_universe()
    targets = _target_tables()
    public_records: list[dict[str, Any]] = []
    hidden_records: list[dict[str, Any]] = []
    for expressibility in config["factorialDesign"]["expressibilityClasses"]:
        for evidence_status in config["factorialDesign"]["evidenceStatuses"]:
            factor_cell = f"{expressibility}::{evidence_status}"
            for base_index, table in enumerate(targets[expressibility]):
                logical_group = f"{factor_cell}::G{base_index + 1}"
                for namespace in config["primitiveNamespaces"]:
                    concept_name = f"concept_{base_index + 1}_{namespace['namespace_id'].lower()}"
                    definition = _definition(
                        expressibility,
                        evidence_status,
                        table,
                        concept_name,
                        namespace,
                        config,
                    )
                    definition_parse = parse_definition(
                        definition, namespace, concept_name
                    )
                    if evidence_status == "sufficient":
                        indices = (
                            list(range(8))
                            if expressibility == "provisional_primitive"
                            else _balanced_consistent_indices(table)
                        )
                        observations = [
                            _observation(ordinal, index, table[index] == "1", namespace)
                            for ordinal, index in enumerate(indices)
                        ]
                    elif evidence_status == "ambiguous":
                        indices = _ambiguous_indices(table, namespace)
                        observations = [
                            _observation(ordinal, index, table[index] == "1", namespace)
                            for ordinal, index in enumerate(indices)
                        ]
                    else:
                        index = next(
                            value for value in range(8) if table[value] in {"0", "1"}
                        )
                        true_value = table[index] == "1"
                        observations = [
                            _observation(0, index, true_value, namespace),
                            _observation(1, index, not true_value, namespace),
                        ]
                    version = enumerate_version_space(
                        definition_parse, observations, namespace, universe
                    )
                    version_ids = [row["candidate_id"] for row in version]
                    version_classes = sorted(
                        {row["expressibility_class"] for row in version}
                    )
                    inferred_status = (
                        "contradictory"
                        if not version
                        else "sufficient"
                        if len(version) == 1
                        else "ambiguous"
                    )
                    record_id = _record_id(
                        factor_cell, logical_group, namespace["namespace_id"]
                    )
                    public = {
                        "record_id": record_id,
                        "split": "development_only",
                        "entity_type": config["typedDSL"]["entityType"],
                        "entity_noun": namespace["entity_noun"],
                        "concept_name": concept_name,
                        "registered_primitives": list(namespace["primitive_names"]),
                        "definition": definition,
                        "observations": observations,
                    }
                    hidden = {
                        **public,
                        "factor_cell": factor_cell,
                        "logical_target_group": logical_group,
                        "namespace_id": namespace["namespace_id"],
                        "target_truth_table": table,
                        "target_candidate_id": candidate_id(table),
                        "generative_expressibility": expressibility,
                        "evidence_status": evidence_status,
                        "version_space_candidate_ids": version_ids,
                        "version_space_size": len(version),
                        "version_space_classes": version_classes,
                        "definition_parse": definition_parse,
                        "identifiability_contract": inferred_status,
                    }
                    public_records.append(public)
                    hidden_records.append(hidden)
    public_records.sort(key=lambda row: row["record_id"])
    hidden_records.sort(key=lambda row: row["record_id"])
    class_counts = Counter(row["expressibility_class"] for row in universe)
    summary = {
        "record_count": len(hidden_records),
        "cell_count": len({row["factor_cell"] for row in hidden_records}),
        "cell_counts": dict(sorted(Counter(row["factor_cell"] for row in hidden_records).items())),
        "logical_target_group_count": len(
            {row["logical_target_group"] for row in hidden_records}
        ),
        "namespace_count": len(config["primitiveNamespaces"]),
        "candidate_truth_table_count": len(universe),
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "evidence_status_counts": dict(
            sorted(Counter(row["evidence_status"] for row in hidden_records).items())
        ),
        "generative_expressibility_counts": dict(
            sorted(
                Counter(row["generative_expressibility"] for row in hidden_records).items()
            )
        ),
        "evaluation_record_count": 0,
        "project_authored_synthetic_development": True,
    }
    return {
        "frozen_ontology": build_frozen_ontology(config),
        "public_records": public_records,
        "hidden_records": hidden_records,
        "population_summary": summary,
    }


def _recursive_key_count(value: Any, forbidden: set[str]) -> int:
    if isinstance(value, dict):
        return sum(key in forbidden for key in value) + sum(
            _recursive_key_count(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_recursive_key_count(child, forbidden) for child in value)
    return 0


def audit_population(
    population: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    public = population["public_records"]
    hidden = population["hidden_records"]
    summary = population["population_summary"]
    gates = config["populationGates"]
    hidden_by_id = {row["record_id"]: row for row in hidden}
    target_retained = [
        row["target_candidate_id"] in row["version_space_candidate_ids"]
        for row in hidden
        if row["evidence_status"] != "contradictory"
    ]
    sufficient = [row for row in hidden if row["evidence_status"] == "sufficient"]
    ambiguous = [row for row in hidden if row["evidence_status"] == "ambiguous"]
    contradictory = [
        row for row in hidden if row["evidence_status"] == "contradictory"
    ]
    renaming_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hidden:
        renaming_groups[row["logical_target_group"]].append(row)
    renaming_invariant = [
        len({row["version_space_size"] for row in rows}) == 1
        and len({tuple(row["version_space_classes"]) for row in rows}) == 1
        for rows in renaming_groups.values()
    ]
    public_exact = all(
        set(row) == set(config["publicFields"]) for row in public
    )
    public_matches_hidden = all(
        row
        == {key: hidden_by_id[row["record_id"]][key] for key in config["publicFields"]}
        for row in public
    )
    leak_count = _recursive_key_count(public, set(config["hiddenFields"]))
    evidence_status_accuracy = sum(
        row["identifiability_contract"] == row["evidence_status"] for row in hidden
    ) / len(hidden)
    sufficient_class_accuracy = sum(
        row["version_space_classes"] == [row["generative_expressibility"]]
        for row in sufficient
    ) / len(sufficient)
    checks = {
        "record_count": summary["record_count"] == gates["requiredRecordCount"],
        "cell_count": summary["cell_count"] == gates["requiredCellCount"],
        "records_per_cell": set(summary["cell_counts"].values())
        == {gates["requiredRecordsPerCell"]},
        "logical_target_group_count": summary["logical_target_group_count"]
        == gates["requiredLogicalTargetGroupCount"],
        "renamings_per_logical_target": all(
            len(rows) == gates["requiredRenamingsPerLogicalTarget"]
            for rows in renaming_groups.values()
        ),
        "candidate_truth_table_count": summary["candidate_truth_table_count"]
        == gates["requiredCandidateTruthTableCount"],
        "registered_alias_truth_table_count": summary["candidate_class_counts"]["alias"]
        == gates["requiredRegisteredAliasTruthTableCount"],
        "registered_composition_truth_table_count": summary["candidate_class_counts"][
            "composition"
        ]
        == gates["requiredRegisteredCompositionTruthTableCount"],
        "provisional_truth_table_count": summary["candidate_class_counts"][
            "provisional_primitive"
        ]
        == gates["requiredProvisionalTruthTableCount"],
        "sufficient_version_space_size": all(
            row["version_space_size"] == gates["requiredSufficientVersionSpaceSize"]
            for row in sufficient
        ),
        "ambiguous_version_space_and_class_coverage": all(
            row["version_space_size"] >= gates["minimumAmbiguousVersionSpaceSize"]
            and len(row["version_space_classes"])
            == gates["requiredAmbiguousExpressibilityClassCoverage"]
            for row in ambiguous
        ),
        "contradictory_version_space_size": all(
            row["version_space_size"] == gates["requiredContradictoryVersionSpaceSize"]
            for row in contradictory
        ),
        "target_retention_when_noncontradictory": sum(target_retained)
        / len(target_retained)
        == gates["requiredTargetRetentionWhenNonContradictory"],
        "sufficient_expressibility_classification": sufficient_class_accuracy
        == gates["requiredSufficientExpressibilityClassificationAccuracy"],
        "evidence_status_classification": evidence_status_accuracy
        == gates["requiredEvidenceStatusClassificationAccuracy"],
        "renaming_version_space_invariance": sum(renaming_invariant)
        / len(renaming_invariant)
        == gates["requiredRenamingVersionSpaceInvariance"],
        "public_projection_is_exact": public_exact and public_matches_hidden,
        "zero_public_hidden_field_leaks": leak_count
        == gates["requiredPublicHiddenFieldLeakCount"],
        "project_authored_development_disclosed": summary[
            "project_authored_synthetic_development"
        ]
        == gates["requiredProjectAuthoredDevelopmentDisclosure"],
        "zero_evaluation_records": summary["evaluation_record_count"]
        <= gates["maximumEvaluationRecordCount"],
        "zero_manual_model_API_training_registration_and_execution": all(
            gates[key] == 0
            for key in (
                "maximumManualJudgmentCount",
                "maximumModelLoadCount",
                "maximumModelGenerationCount",
                "maximumAPICallCount",
                "maximumTrainingRunCount",
                "maximumOntologyRegistrationCount",
                "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount",
                "maximumActualExecutionCount",
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "target_retention_when_noncontradictory": sum(target_retained)
        / len(target_retained),
        "sufficient_expressibility_classification_accuracy": sufficient_class_accuracy,
        "evidence_status_classification_accuracy": evidence_status_accuracy,
        "renaming_version_space_invariance": sum(renaming_invariant)
        / len(renaming_invariant),
        "public_hidden_field_leak_count": leak_count,
        "version_space_size_by_status": {
            status: {
                "minimum": min(
                    row["version_space_size"]
                    for row in hidden
                    if row["evidence_status"] == status
                ),
                "maximum": max(
                    row["version_space_size"]
                    for row in hidden
                    if row["evidence_status"] == status
                ),
            }
            for status in EVIDENCE_STATUSES
        },
        "access": {
            "evaluation_record_count": 0,
            "manual_judgment_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "ontology_registration_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }


__all__ = [
    "EVIDENCE_STATUSES",
    "EXPRESSIBILITY_CLASSES",
    "audit_population",
    "build_frozen_ontology",
    "build_population",
    "candidate_universe",
    "enumerate_version_space",
    "parse_definition",
    "registered_tables",
    "valuations",
]
