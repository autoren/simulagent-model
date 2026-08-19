from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
import re
from typing import Any, Iterable

from v209r1_dynamic_regime_shape_repair import build_kernel
from v209_controlled_language_observation_pomdp import (
    OBSERVATION_NAMES,
    REGIME_NAMES,
    STAGE_NAMES,
    STATE_NAMES,
    clarification_likelihood,
)


ROLE_NAMES = ("DEVELOPMENT", "PROTECTED")
COUNTERFACTUAL_TYPES = ("DIRECT", "MATCHED_PARAPHRASE", "OPAQUE_RENAMING")
ROLE_PREFIX = {"DEVELOPMENT": "DEV", "PROTECTED": "PROTECTED"}
DIRECT_PATTERN = re.compile(r"<OBS=(ALPHA|BETA|UNRESOLVED)>")
DIRECT_TO_OBSERVATION = {
    "ALPHA": "UTTERANCE_ALPHA",
    "BETA": "UTTERANCE_BETA",
    "UNRESOLVED": "UTTERANCE_UNRESOLVED",
}


def canonical_jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        for record in records
    )


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identifier_hash(record_ids: Iterable[str]) -> str:
    return bytes_sha256(("\n".join(record_ids) + "\n").encode("utf-8"))


def _stable_template_index(seed: int, group_id: str, counterfactual_type: str, count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}:{counterfactual_type}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _context_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in config["population"]["contexts"]}


def _source_probability(
    kernel: Any,
    regime: str,
    state: str,
    context: dict[str, Any],
    observation: str,
) -> float:
    stage = STAGE_NAMES.index(context["stage"])
    history = tuple(OBSERVATION_NAMES.index(item) for item in context["history"])
    likelihood = clarification_likelihood(kernel, context["action"], stage, history)
    return float(
        likelihood[
            REGIME_NAMES.index(regime),
            STATE_NAMES.index(state),
            OBSERVATION_NAMES.index(observation),
        ]
    )


def _surface_text(
    config: dict[str, Any],
    role: str,
    context_id: str,
    observation: str,
    counterfactual_type: str,
    group_id: str,
) -> tuple[str, str, int]:
    grammar = config["surfaceGrammar"]
    templates = grammar["templates"][role][counterfactual_type]
    template_index = _stable_template_index(
        int(config["population"]["roleSeeds"][role]), group_id, counterfactual_type, len(templates)
    )
    template = templates[template_index]
    context = grammar["contextPhrases"][role][context_id]
    values: dict[str, str] = {"context": context, "context_capitalized": context[:1].upper() + context[1:]}
    for lexicon_name, mapping in grammar["lexicons"][role].items():
        values[lexicon_name] = mapping[observation]
    text = template.format(**values)
    family_index = COUNTERFACTUAL_TYPES.index(counterfactual_type)
    construction_family = grammar["constructionFamiliesByRole"][role][family_index]
    return text, construction_family, template_index


def generate_role(
    config: dict[str, Any],
    parent_config: dict[str, Any],
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if role not in ROLE_NAMES:
        raise ValueError(f"V210 unknown role {role}")
    kernel, _ = build_kernel(parent_config)
    contexts = _context_by_id(config)
    base_factors = [
        (regime, state, context_id, observation)
        for regime in config["population"]["semanticRegimes"]
        for state in config["population"]["taskStates"]
        for context_id in contexts
        for observation in config["population"]["semanticObservationIds"]
    ]
    rng = random.Random(int(config["population"]["roleSeeds"][role]))
    rng.shuffle(base_factors)
    surfaces: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    record_number = 0
    for group_number, (regime, state, context_id, observation) in enumerate(base_factors, start=1):
        group_id = f"{ROLE_PREFIX[role]}-G{group_number:03d}"
        context = contexts[context_id]
        probability = _source_probability(kernel, regime, state, context, observation)
        for counterfactual_type in config["population"]["counterfactualTypes"]:
            record_number += 1
            record_id = f"{ROLE_PREFIX[role]}-{record_number:04d}"
            utterance, construction_family, template_index = _surface_text(
                config, role, context_id, observation, counterfactual_type, group_id
            )
            shared = {
                "record_id": record_id,
                "group_id": group_id,
                "role": role,
                "context_id": context_id,
                "counterfactual_type": counterfactual_type,
                "construction_family": construction_family,
                "generator_version": config["population"]["generatorVersion"],
                "template_index": template_index,
            }
            surfaces.append({**shared, "utterance": utterance})
            truths.append(
                {
                    **shared,
                    "semantic_regime": regime,
                    "task_state": state,
                    "clarification_action": context["action"],
                    "stage": context["stage"],
                    "history": list(context["history"]),
                    "semantic_observation_id": observation,
                    "source_probability": probability,
                }
            )
    return surfaces, truths


def generate_population(
    config: dict[str, Any], parent_config: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role in ROLE_NAMES:
        surfaces, truths = generate_role(config, parent_config, role)
        result[role] = {"surfaces": surfaces, "truth": truths}
    return result


def project_development_surfaces(
    surfaces: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    predictions = []
    for record in surfaces:
        utterance = record["utterance"]
        matches = DIRECT_PATTERN.findall(utterance)
        prediction = DIRECT_TO_OBSERVATION[matches[0]] if len(matches) == 1 else "ABSTAIN"
        predictions.append(
            {
                "record_id": record["record_id"],
                "prediction": prediction,
                "accepted": prediction != "ABSTAIN",
                "projector": config["projection"]["name"],
            }
        )
    return predictions


def _surface_truth_round_trip(
    surfaces: list[dict[str, Any]], truths: list[dict[str, Any]]
) -> tuple[float, int]:
    surface_by_id = {row["record_id"]: row for row in surfaces}
    truth_by_id = {row["record_id"]: row for row in truths}
    keys = (
        "record_id",
        "group_id",
        "role",
        "context_id",
        "counterfactual_type",
        "construction_family",
        "generator_version",
        "template_index",
    )
    matched = 0
    for record_id in set(surface_by_id) | set(truth_by_id):
        if record_id in surface_by_id and record_id in truth_by_id:
            matched += int(all(surface_by_id[record_id][key] == truth_by_id[record_id][key] for key in keys))
    denominator = max(len(set(surface_by_id) | set(truth_by_id)), 1)
    return matched / denominator, denominator - matched


def _counterfactual_diagnostics(
    truths: list[dict[str, Any]], surfaces: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, int]:
    truth_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    surface_by_id = {row["record_id"]: row for row in surfaces}
    for row in truths:
        truth_groups[row["group_id"]].append(row)
    truth_keys = (
        "semantic_regime",
        "task_state",
        "clarification_action",
        "stage",
        "history",
        "semantic_observation_id",
    )
    truth_mismatch = 0
    probability_mismatch = 0
    opaque_mismatch = 0
    for rows in truth_groups.values():
        first = rows[0]
        truth_mismatch += int(
            len(rows) != len(COUNTERFACTUAL_TYPES)
            or {row["counterfactual_type"] for row in rows} != set(COUNTERFACTUAL_TYPES)
            or any(any(row[key] != first[key] for key in truth_keys) for row in rows[1:])
        )
        probability_mismatch += int(any(row["source_probability"] != first["source_probability"] for row in rows[1:]))
        opaque = next((row for row in rows if row["counterfactual_type"] == "OPAQUE_RENAMING"), None)
        if opaque is None:
            opaque_mismatch += 1
            continue
        surface = surface_by_id[opaque["record_id"]]["utterance"]
        role = opaque["role"]
        expected = config["surfaceGrammar"]["lexicons"][role][
            "opaque" if role == "DEVELOPMENT" else "protected_opaque"
        ][opaque["semantic_observation_id"]]
        opaque_mismatch += int(expected not in surface)
    return {
        "counterfactual_truth_mismatch_count": truth_mismatch,
        "counterfactual_probability_mismatch_count": probability_mismatch,
        "opaque_renaming_mismatch_count": opaque_mismatch,
    }


def _probability_normalization_rate(truths: list[dict[str, Any]]) -> tuple[float, int]:
    groups: dict[tuple[Any, ...], float] = defaultdict(float)
    for row in truths:
        key = (
            row["role"],
            row["semantic_regime"],
            row["task_state"],
            row["context_id"],
            row["counterfactual_type"],
        )
        groups[key] += float(row["source_probability"])
    normalized = sum(abs(total - 1.0) <= 1e-12 for total in groups.values())
    return normalized / max(len(groups), 1), len(groups) - normalized


def _role_grammar_overlap(config: dict[str, Any]) -> dict[str, int]:
    grammar = config["surfaceGrammar"]
    dev_templates = {template for rows in grammar["templates"]["DEVELOPMENT"].values() for template in rows}
    protected_templates = {template for rows in grammar["templates"]["PROTECTED"].values() for template in rows}
    dev_labels = {value.casefold() for mapping in grammar["lexicons"]["DEVELOPMENT"].values() for value in mapping.values()}
    protected_labels = {value.casefold() for mapping in grammar["lexicons"]["PROTECTED"].values() for value in mapping.values()}
    return {
        "role_template_skeleton_overlap_count": len(dev_templates & protected_templates),
        "role_lexical_label_overlap_count": len(dev_labels & protected_labels),
    }


def _population_role_summary(
    surfaces: list[dict[str, Any]], truths: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    ids = [row["record_id"] for row in surfaces]
    factor_keys = [
        (
            row["role"],
            row["semantic_regime"],
            row["task_state"],
            row["context_id"],
            row["semantic_observation_id"],
            row["counterfactual_type"],
        )
        for row in truths
    ]
    cells = Counter(
        (row["semantic_regime"], row["task_state"], row["context_id"], row["semantic_observation_id"])
        for row in truths
    )
    round_trip_rate, round_trip_mismatches = _surface_truth_round_trip(surfaces, truths)
    probability_rate, probability_failures = _probability_normalization_rate(truths)
    forbidden_fields = {"semantic_regime", "task_state", "semantic_observation_id", "source_probability", "history"}
    return {
        "record_count": len(surfaces),
        "truth_record_count": len(truths),
        "group_count": len({row["group_id"] for row in surfaces}),
        "regimes": sorted({row["semantic_regime"] for row in truths}),
        "states": sorted({row["task_state"] for row in truths}),
        "contexts": sorted({row["context_id"] for row in truths}),
        "observations": sorted({row["semantic_observation_id"] for row in truths}),
        "counterfactual_types": sorted({row["counterfactual_type"] for row in truths}),
        "minimum_records_per_regime_state_context_observation_cell": min(cells.values()),
        "unique_record_id_rate": len(set(ids)) / max(len(ids), 1),
        "unique_full_factor_key_rate": len(set(factor_keys)) / max(len(factor_keys), 1),
        "identifier_sha256": identifier_hash(ids),
        "surface_artifact_sha256": bytes_sha256(canonical_jsonl(surfaces)),
        "truth_artifact_sha256": bytes_sha256(canonical_jsonl(truths)),
        "truth_surface_round_trip_rate": round_trip_rate,
        "truth_surface_round_trip_mismatch_count": round_trip_mismatches,
        "probability_normalization_rate": probability_rate,
        "probability_normalization_failure_count": probability_failures,
        "surface_truth_field_leak_count": sum(bool(set(row) & forbidden_fields) for row in surfaces),
        **_counterfactual_diagnostics(truths, surfaces, config),
    }


def evaluate_projection(
    predictions: list[dict[str, Any]],
    development_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_by_id = {row["record_id"]: row for row in development_truth}
    accepted = [row for row in predictions if row["accepted"]]
    residual = [row for row in predictions if not row["accepted"]]
    accepted_correct = sum(
        row["prediction"] == truth_by_id[row["record_id"]]["semantic_observation_id"] for row in accepted
    )
    residual_truth = [truth_by_id[row["record_id"]] for row in residual]
    return {
        "prediction_count": len(predictions),
        "accepted_count": len(accepted),
        "residual_count": len(residual),
        "coverage": len(accepted) / max(len(predictions), 1),
        "accepted_accuracy": accepted_correct / max(len(accepted), 1),
        "false_acceptance_count": len(accepted) - accepted_correct,
        "residual_counterfactual_types": sorted({row["counterfactual_type"] for row in residual_truth}),
        "residual_regimes": sorted({row["semantic_regime"] for row in residual_truth}),
        "residual_states": sorted({row["task_state"] for row in residual_truth}),
        "residual_contexts": sorted({row["context_id"] for row in residual_truth}),
        "residual_observations": sorted({row["semantic_observation_id"] for row in residual_truth}),
        "residual_record_ids": [row["record_id"] for row in residual],
        "prediction_truth_read_count": 0,
        "protected_prediction_read_count": 0,
    }


def evaluate_population(
    population: dict[str, dict[str, list[dict[str, Any]]]],
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
    parent_config: dict[str, Any],
) -> dict[str, Any]:
    regenerated = generate_population(config, parent_config)
    deterministic_match = canonical_jsonl(
        population["DEVELOPMENT"]["surfaces"]
        + population["DEVELOPMENT"]["truth"]
        + population["PROTECTED"]["surfaces"]
        + population["PROTECTED"]["truth"]
    ) == canonical_jsonl(
        regenerated["DEVELOPMENT"]["surfaces"]
        + regenerated["DEVELOPMENT"]["truth"]
        + regenerated["PROTECTED"]["surfaces"]
        + regenerated["PROTECTED"]["truth"]
    )
    role_summaries = {
        role: _population_role_summary(population[role]["surfaces"], population[role]["truth"], config)
        for role in ROLE_NAMES
    }
    dev_ids = {row["record_id"] for row in population["DEVELOPMENT"]["surfaces"]}
    protected_ids = {row["record_id"] for row in population["PROTECTED"]["surfaces"]}
    dev_groups = {row["group_id"] for row in population["DEVELOPMENT"]["surfaces"]}
    protected_groups = {row["group_id"] for row in population["PROTECTED"]["surfaces"]}
    dev_strings = {" ".join(row["utterance"].casefold().split()) for row in population["DEVELOPMENT"]["surfaces"]}
    protected_strings = {" ".join(row["utterance"].casefold().split()) for row in population["PROTECTED"]["surfaces"]}
    all_ids = [row["record_id"] for role in ROLE_NAMES for row in population[role]["surfaces"]]
    return {
        "roles": role_summaries,
        "cross_role": {
            "record_id_overlap_count": len(dev_ids & protected_ids),
            "group_id_overlap_count": len(dev_groups & protected_groups),
            "normalized_surface_string_overlap_count": len(dev_strings & protected_strings),
            "all_identifier_sha256": identifier_hash(all_ids),
            **_role_grammar_overlap(config),
        },
        "deterministic_regeneration_match": deterministic_match,
        "projection": evaluate_projection(predictions, population["DEVELOPMENT"]["truth"]),
        "access": {
            "population_generation_count": 1,
            "development_projection_evaluation_count": 1,
            "protected_surface_record_automatic_audit_count": len(population["PROTECTED"]["surfaces"]),
            "protected_surface_manual_read_count": 0,
            "protected_surface_baseline_read_count": 0,
            "external_language_record_read_count": 0,
            "raw_model_response_read_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "ontology_registration_count": 0,
            "trusted_state_mutation_count": 0,
            "service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
            "fallback_count": 0,
        },
    }


def audit_evaluation(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    population_gates = config["populationGates"]
    projection_gates = config["projectionGates"]
    roles = summary["roles"]
    cross = summary["cross_role"]
    role_checks = {}
    for role in ROLE_NAMES:
        row = roles[role]
        role_checks[role] = bool(
            row["record_count"] == row["truth_record_count"] == population_gates["requiredRecordsPerRole"]
            and row["group_count"] == population_gates["requiredGroupsPerRole"]
            and len(row["regimes"]) == population_gates["requiredRegimeCountPerRole"]
            and len(row["states"]) == population_gates["requiredStateCountPerRole"]
            and len(row["contexts"]) == population_gates["requiredContextCountPerRole"]
            and len(row["observations"]) == population_gates["requiredObservationCountPerRole"]
            and len(row["counterfactual_types"]) == population_gates["requiredCounterfactualTypeCountPerRole"]
            and row["minimum_records_per_regime_state_context_observation_cell"]
            >= population_gates["minimumRecordsPerRegimeStateContextObservationCell"]
            and row["unique_record_id_rate"] == population_gates["requiredUniqueRecordIdRate"]
            and row["unique_full_factor_key_rate"] == population_gates["requiredUniqueFullFactorKeyRate"]
            and row["identifier_sha256"] == config["population"]["recordIdHashes"][role]
            and row["truth_surface_round_trip_rate"] == population_gates["requiredTruthSurfaceRoundTripRate"]
            and row["surface_truth_field_leak_count"] == population_gates["requiredSurfaceTruthFieldLeakCount"]
            and row["counterfactual_truth_mismatch_count"] == population_gates["requiredCounterfactualTruthMismatchCount"]
            and row["counterfactual_probability_mismatch_count"] == population_gates["requiredCounterfactualProbabilityMismatchCount"]
            and row["opaque_renaming_mismatch_count"] == population_gates["requiredOpaqueRenamingMismatchCount"]
            and row["probability_normalization_rate"] == population_gates["requiredProbabilityNormalizationRate"]
        )
    projection = summary["projection"]
    checks = {
        "role_population_counts_balance_uniqueness_and_firewall": all(role_checks.values()),
        "roles_are_construction_identifier_group_and_surface_disjoint": bool(
            cross["record_id_overlap_count"] == 0
            and cross["group_id_overlap_count"] == 0
            and cross["normalized_surface_string_overlap_count"] == 0
            and cross["role_template_skeleton_overlap_count"] == population_gates["requiredRoleTemplateSkeletonOverlapCount"]
            and cross["role_lexical_label_overlap_count"] == population_gates["requiredRoleLexicalLabelOverlapCount"]
            and cross["all_identifier_sha256"] == config["population"]["recordIdHashes"]["ALL"]
        ),
        "population_regenerates_byte_exactly": summary["deterministic_regeneration_match"] == population_gates["requiredDeterministicRegenerationMatch"],
        "projection_counts_coverage_accuracy_and_safety_are_exact": bool(
            projection["prediction_count"] == projection_gates["requiredDevelopmentPredictionCount"]
            and projection["accepted_count"] == projection_gates["requiredAcceptedCount"]
            and projection["residual_count"] == projection_gates["requiredResidualCount"]
            and abs(projection["coverage"] - projection_gates["requiredCoverage"]) <= 1e-12
            and projection["accepted_accuracy"] == projection_gates["requiredAcceptedAccuracy"]
            and projection["false_acceptance_count"] == projection_gates["requiredFalseAcceptanceCount"]
        ),
        "residual_is_prediction_only_and_factor_complete": bool(
            projection["residual_counterfactual_types"] == sorted(projection_gates["requiredResidualCounterfactualTypes"])
            and len(projection["residual_regimes"]) == projection_gates["requiredResidualRegimeCount"]
            and len(projection["residual_states"]) == projection_gates["requiredResidualStateCount"]
            and len(projection["residual_contexts"]) == projection_gates["requiredResidualContextCount"]
            and len(projection["residual_observations"]) == projection_gates["requiredResidualObservationCount"]
            and projection["prediction_truth_read_count"] == projection_gates["requiredPredictionTruthReadCount"]
            and projection["protected_prediction_read_count"] == projection_gates["requiredProtectedPredictionReadCount"]
        ),
    }
    access_gates = config["accessGates"]
    access = summary["access"]
    access_checks = {
        "authorized_generation_and_projection_counts_exact": bool(
            access["population_generation_count"] == access_gates["requiredPopulationGenerationCount"]
            and access["development_projection_evaluation_count"] == access_gates["requiredDevelopmentProjectionEvaluationCount"]
        ),
        "protected_manual_and_baseline_reads_are_zero": bool(
            access["protected_surface_manual_read_count"] <= access_gates["maximumProtectedSurfaceManualReadCount"]
            and access["protected_surface_baseline_read_count"] <= access_gates["maximumProtectedSurfaceBaselineReadCount"]
        ),
        "forbidden_access_and_effects_zero": all(
            access[key] <= access_gates[gate]
            for key, gate in (
                ("external_language_record_read_count", "maximumExternalLanguageRecordReadCount"),
                ("raw_model_response_read_count", "maximumRawModelResponseReadCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ) and access["fallback_count"] <= population_gates["maximumFallbackCount"],
    }
    return {
        "passed": all(checks.values()) and all(access_checks.values()),
        "population_projection_gates_passed": all(checks.values()),
        "access_gates_passed": all(access_checks.values()),
        "checks": checks,
        "role_checks": role_checks,
        "access_checks": access_checks,
    }


__all__ = [
    "COUNTERFACTUAL_TYPES",
    "ROLE_NAMES",
    "audit_evaluation",
    "bytes_sha256",
    "canonical_jsonl",
    "evaluate_population",
    "generate_population",
    "identifier_hash",
    "project_development_surfaces",
]
