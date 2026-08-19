from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from v218_mondo_artifact_population import (
    UnionFind,
    absence_signature,
    semantic_state,
    stable_hash,
    state_class,
    status_state,
    text_state,
)


TEXT_EVENTS = {"NAME_CHANGED", "DEFINITION_CHANGED", "SYNONYM_CHANGED"}
LIFECYCLE_EVENTS = {"OBSOLETION_CHANGED", "REPLACEMENT_CHANGED", "OBSOLETION_CANDIDATE_STATUS_CHANGED"}


def normalize_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def tokens(value: str) -> frozenset[str]:
    return frozenset(normalize_surface(value).split())


def character_ngrams(value: str, sizes: Iterable[int] = (3, 4, 5)) -> frozenset[str]:
    compact = f" {normalize_surface(value)} "
    return frozenset(compact[index : index + size] for size in sizes for index in range(max(0, len(compact) - size + 1)))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def dice(left: frozenset[str], right: frozenset[str]) -> float:
    return 2 * len(left & right) / (len(left) + len(right)) if left or right else 1.0


def role_order(group_id: str) -> str:
    return hashlib.sha256(f"V221_ROLE|{group_id}".encode("utf-8")).hexdigest()


def derive_role_manifest(development_group_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(development_group_ids, key=lambda group_id: (role_order(group_id), group_id))
    evaluation = [group_id for index, group_id in enumerate(ordered) if index % 5 == 0]
    calibration = [group_id for index, group_id in enumerate(ordered) if index % 5 != 0]
    return {
        "schema_version": "221-deterministic-mondo-role-manifest",
        "experiment": config["experiment"],
        "assignment": config["roleSplit"]["assignment"],
        "ordered_development_group_count": len(ordered),
        "calibration_group_ids": sorted(calibration),
        "evaluation_group_ids": sorted(evaluation),
        "group_overlap_count": len(set(calibration) & set(evaluation)),
        "source_group_accounting_exact": set(calibration) | set(evaluation) == set(development_group_ids),
        "evaluation_tuning_count": 0,
    }


@dataclass(frozen=True)
class SurfaceEntry:
    normalized_surface: str
    family_id: str
    term_id: str
    release: str
    state_class_id: str
    token_set: frozenset[str]
    ngram_set: frozenset[str]


def _surface_values(fields: dict[str, list[str]]) -> list[str]:
    text = text_state(fields)
    return sorted({value for value in [text["name"], *text["synonyms"]] if value})


def build_catalog(
    older_terms: dict[str, dict[str, list[str]]],
    newer_terms: dict[str, dict[str, list[str]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    all_ids = set(older_terms) | set(newer_terms)
    union_find = UnionFind(all_ids)
    for fields in list(older_terms.values()) + list(newer_terms.values()):
        if not fields.get("id"):
            continue
        source = fields["id"][0].strip()
        lifecycle = status_state(fields)
        for target in lifecycle["replaced_by"] + lifecycle["consider"]:
            union_find.union(source, target)
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for term_id in sorted(union_find.parent):
        members_by_root[union_find.find(term_id)].append(term_id)
    entries: list[SurfaceEntry] = []
    families: dict[str, dict[str, Any]] = {}
    valid_classes: set[str] = set()
    source_experiment = config["catalogDesign"]["sourcePopulationExperiment"]
    for member_ids in members_by_root.values():
        member_ids = sorted(member_ids)
        family_id = "F_" + stable_hash(member_ids)[:24]
        group_id = "G_" + stable_hash([source_experiment, family_id])[:24]
        old_classes: list[str] = []
        current_classes: list[str] = []
        replacement_classes: list[str] = []
        replacement_ids: set[str] = set()
        class_decisions: dict[str, str] = {}
        for term_id in member_ids:
            for release, fields in (("OLDER", older_terms.get(term_id)), ("CURRENT", newer_terms.get(term_id))):
                if not fields:
                    continue
                signature = semantic_state(fields, config)
                class_id = state_class(signature)
                valid_classes.add(class_id)
                (old_classes if release == "OLDER" else current_classes).append(class_id)
                lifecycle = status_state(fields)
                if release == "CURRENT":
                    if lifecycle["is_obsolete"] and lifecycle["replaced_by"]:
                        class_decisions[class_id] = "FOLLOW_ASSERTED_REPLACEMENT"
                    elif lifecycle["is_obsolete"]:
                        class_decisions[class_id] = "ABSTAIN_NOT_EXPRESSIBLE"
                    else:
                        class_decisions[class_id] = "RESOLVE_CURRENT_STATE"
                for surface in _surface_values(fields):
                    normalized = normalize_surface(surface)
                    if normalized:
                        entries.append(
                            SurfaceEntry(
                                normalized, family_id, term_id, release, class_id,
                                tokens(normalized), character_ngrams(normalized),
                            )
                        )
                replacement_ids.update(lifecycle["replaced_by"])
        if not old_classes:
            old_absent = state_class(absence_signature("OLDER"))
            valid_classes.add(old_absent)
            old_classes.append(old_absent)
        if not current_classes:
            current_absent = state_class(absence_signature("CURRENT"))
            valid_classes.add(current_absent)
            current_classes.append(current_absent)
        for target in sorted(replacement_ids):
            if target in newer_terms:
                replacement_class = state_class(semantic_state(newer_terms[target], config))
                replacement_classes.append(replacement_class)
                valid_classes.add(replacement_class)
                class_decisions[replacement_class] = "RESOLVE_CURRENT_STATE"
        current_present = any(term_id in newer_terms for term_id in member_ids)
        current_obsolete = any(
            status_state(newer_terms[term_id])["is_obsolete"] for term_id in member_ids if term_id in newer_terms
        )
        if not current_present:
            declared_classes: list[str] = []
            family_decision = "ABSTAIN_NOT_EXPRESSIBLE"
        elif current_obsolete and replacement_classes:
            declared_classes = sorted(set(replacement_classes))
            family_decision = "FOLLOW_ASSERTED_REPLACEMENT"
        elif current_obsolete:
            declared_classes = []
            family_decision = "ABSTAIN_NOT_EXPRESSIBLE"
        else:
            declared_classes = sorted(set(current_classes))
            family_decision = "RESOLVE_CURRENT_STATE"
        families[family_id] = {
            "family_id": family_id,
            "group_id": group_id,
            "member_ids": member_ids,
            "all_version_class_ids": sorted(set(old_classes + current_classes + replacement_classes)),
            "declared_current_class_ids": declared_classes,
            "family_decision": family_decision,
            "class_decisions": class_decisions,
        }
    entries = sorted(set(entries), key=lambda row: (
        row.normalized_surface, row.release, row.term_id, row.state_class_id, row.family_id
    ))
    exact_index: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, set[int]] = defaultdict(set)
    ngram_index: dict[str, set[int]] = defaultdict(set)
    for index, entry in enumerate(entries):
        exact_index[entry.normalized_surface].append(index)
        for token in entry.token_set:
            token_index[token].add(index)
        for ngram in entry.ngram_set:
            ngram_index[ngram].add(index)
    return {
        "families": families,
        "entries": entries,
        "exact_index": dict(exact_index),
        "token_index": dict(token_index),
        "ngram_index": dict(ngram_index),
        "valid_classes": valid_classes,
        "manifest": {
            "schema_version": "221-deterministic-mondo-catalog-manifest",
            "family_count": len(families),
            "surface_entry_count": len(entries),
            "unique_normalized_surface_count": len(exact_index),
            "valid_state_class_count": len(valid_classes),
            "older_term_count": len(older_terms),
            "newer_term_count": len(newer_terms),
            "remote_import_resolution_count": 0,
            "state_class_definition": config["catalogDesign"]["stateClassDefinition"],
        },
    }


def _applicable_classes(family: dict[str, Any], mode: str) -> list[str]:
    return family["all_version_class_ids"] if mode == "VERSION_UNSPECIFIED" else family["declared_current_class_ids"]


def _rank_families(query: str, catalog: dict[str, Any]) -> tuple[list[str], set[str]]:
    normalized = normalize_surface(query)
    query_tokens = tokens(normalized)
    query_ngrams = character_ngrams(normalized)
    exact_indices = catalog["exact_index"].get(normalized, [])
    exact_families = {catalog["entries"][index].family_id for index in exact_indices}
    if exact_families:
        return sorted(exact_families), exact_families
    candidates: set[int] = set(exact_indices)
    for token in query_tokens:
        candidates.update(catalog["token_index"].get(token, set()))
    for ngram in query_ngrams:
        candidates.update(catalog["ngram_index"].get(ngram, set()))
    best: dict[str, tuple[float, str, str, str, str]] = {}
    for index in candidates:
        entry = catalog["entries"][index]
        score = (
            3.0 if entry.normalized_surface == normalized else 0.0
        ) + jaccard(query_tokens, entry.token_set) + dice(query_ngrams, entry.ngram_set)
        tie = (score, entry.normalized_surface, entry.release, entry.term_id, entry.state_class_id)
        prior = best.get(entry.family_id)
        if prior is None or (-tie[0], *tie[1:]) < (-prior[0], *prior[1:]):
            best[entry.family_id] = tie
    ranked = sorted(best, key=lambda family_id: (-best[family_id][0], *best[family_id][1:], family_id))
    ranked = [family_id for family_id in ranked if family_id in exact_families] + [
        family_id for family_id in ranked if family_id not in exact_families
    ]
    return ranked, exact_families


def generate_candidates(
    public: dict[str, Any], catalog: dict[str, Any], method_id: str, budget: int
) -> dict[str, Any]:
    normalized = normalize_surface(public["surface_text"])
    entries = catalog["entries"]
    exact_indices = catalog["exact_index"].get(normalized, [])
    exact_families = {entries[index].family_id for index in exact_indices}
    mode = public["evidence_mode"]
    overflow = False
    atomic = True
    retrieval_used = False
    selected_families: list[str] = []
    if method_id == "M0_NORMALIZED_EXACT":
        classes = []
        for index in exact_indices:
            entry = entries[index]
            if mode == "CURRENT_RELEASE_DECLARED" and entry.release != "CURRENT":
                continue
            if entry.state_class_id not in classes and len(classes) < budget:
                classes.append(entry.state_class_id)
            elif entry.state_class_id not in classes:
                overflow = True
        return {
            "candidate_class_ids": sorted(classes), "selected_family_ids": [],
            "exact_family_ids": sorted(exact_families), "exact_evidence": bool(exact_families),
            "retrieval_used": False, "overflow": overflow, "atomic_expansion": True,
        }
    if method_id == "M1_EXACT_FAMILY":
        ranked = sorted(exact_families)
    else:
        ranked, exact_families = _rank_families(public["surface_text"], catalog)
    classes: list[str] = []
    for family_id in ranked:
        family_classes = _applicable_classes(catalog["families"][family_id], mode)
        new_classes = [class_id for class_id in family_classes if class_id not in classes]
        if len(classes) + len(new_classes) > budget:
            overflow = True
            continue
        selected_families.append(family_id)
        classes.extend(new_classes)
        if family_id not in exact_families:
            retrieval_used = True
        if len(classes) == budget:
            break
    for family_id in selected_families:
        family_classes = set(_applicable_classes(catalog["families"][family_id], mode))
        atomic = atomic and family_classes <= set(classes)
    return {
        "candidate_class_ids": sorted(set(classes)), "selected_family_ids": selected_families,
        "exact_family_ids": sorted(exact_families), "exact_evidence": bool(exact_families),
        "retrieval_used": retrieval_used, "overflow": overflow, "atomic_expansion": atomic,
    }


def controller_decision(public: dict[str, Any], generated: dict[str, Any], catalog: dict[str, Any], method_id: str) -> tuple[str, bool]:
    preserve = "PRESERVE_VERSION_SPACE_OR_CLARIFY"
    if public["evidence_mode"] == "VERSION_UNSPECIFIED":
        return preserve, True
    if not generated["candidate_class_ids"]:
        return preserve, True
    if method_id == "M3_FINAL_FAIL_CLOSED" and (
        not generated["exact_evidence"] or generated["overflow"]
    ):
        return preserve, True
    decisions = {
        catalog["families"][family_id]["family_decision"]
        for family_id in generated["selected_family_ids"]
        if family_id in catalog["families"]
    }
    if method_id == "M0_NORMALIZED_EXACT":
        decisions = {
            decision
            for family_id in generated["exact_family_ids"]
            if (decision := catalog["families"][family_id]["family_decision"])
        }
    if len(decisions) != 1:
        return preserve, True
    return next(iter(decisions)), False


def event_stratum(events: list[str]) -> str:
    values = set(events)
    if "ADDED" in values:
        return "ADDED"
    if values & TEXT_EVENTS:
        return "TEXT_CHANGE"
    if values & LIFECYCLE_EVENTS:
        return "LIFECYCLE_CHANGE"
    if "MAPPING_CHANGED" in values:
        return "MAPPING_CHANGED"
    if "LOGICAL_AXIOM_CHANGED" in values:
        return "LOGICAL_AXIOM_CHANGED"
    return "OTHER"


def score_record(
    public: dict[str, Any], truth: dict[str, Any], role: str,
    catalog: dict[str, Any], method_id: str, budget: int,
) -> dict[str, Any]:
    generated = generate_candidates(public, catalog, method_id, budget)
    candidates = set(generated["candidate_class_ids"])
    oracle = set(truth["candidate_class_ids"])
    recall = len(candidates & oracle) / len(oracle) if oracle else (1.0 if not candidates else 0.0)
    predicted, fail_closed = controller_decision(public, generated, catalog, method_id)
    losses = truth["decision_consequence"]["action_loss"]
    if predicted in losses:
        regret = losses[predicted]
    elif predicted == "PRESERVE_VERSION_SPACE_OR_CLARIFY" and "PRESERVE_VERSION_SPACE_OR_CLARIFY" in losses:
        regret = losses["PRESERVE_VERSION_SPACE_OR_CLARIFY"]
    else:
        regret = 1.0
    ambiguous = truth["evidence_state"] == "AMBIGUOUS" or len(oracle) > 1
    singleton_under_ambiguous = ambiguous and len(candidates) == 1
    unsafe_singleton = singleton_under_ambiguous and predicted != "PRESERVE_VERSION_SPACE_OR_CLARIFY"
    decision_relevant_missing = recall < 1.0 and bool(truth["boundary_witness"] or len(oracle) <= 1)
    return {
        "case_id": public["case_id"], "group_id": public["group_id"], "role": role,
        "method_id": method_id, "budget": budget, "evidence_mode": public["evidence_mode"],
        "event_stratum": event_stratum(truth["event_types"]),
        "candidate_class_ids": sorted(candidates), "oracle_class_count": len(oracle),
        "oracle_class_recall": recall, "full_version_space_retention": recall == 1.0,
        "candidate_set_size": len(candidates), "singleton_under_ambiguous_truth": singleton_under_ambiguous,
        "unsafe_singleton_collapse": unsafe_singleton, "predicted_decision": predicted,
        "correct_decision": truth["correct_decision"], "exact_decision": predicted == truth["correct_decision"],
        "decision_regret": regret, "decision_relevant_missing": decision_relevant_missing,
        "candidate_classes_valid": candidates <= catalog["valid_classes"],
        "candidate_budget_compliant": len(candidates) <= budget,
        "atomic_family_expansion": generated["atomic_expansion"],
        "exact_evidence": generated["exact_evidence"], "retrieval_used": generated["retrieval_used"],
        "overflow": generated["overflow"], "fail_closed": fail_closed,
        "conflict_or_insufficient_requires_fail_closed": bool(
            public["evidence_mode"] == "VERSION_UNSPECIFIED"
            or not generated["candidate_class_ids"] or not generated["exact_evidence"] or generated["overflow"]
        ),
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def summarize_observations(observations: list[dict[str, Any]], role_manifest: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cells: dict[str, Any] = {}
    expected_cells = 0
    for role, method_id, budget, mode in itertools.product(
        config["metrics"]["requiredByRole"], config["metrics"]["requiredByMethod"],
        config["metrics"]["requiredByBudget"], config["metrics"]["requiredByEvidenceMode"],
    ):
        expected_cells += 1
        rows = [row for row in observations if row["role"] == role and row["method_id"] == method_id and row["budget"] == budget and row["evidence_mode"] == mode]
        key = f"{role}|{method_id}|k={budget}|{mode}"
        cells[key] = {
            "record_count": len(rows), "oracle_class_recall": _mean(rows, "oracle_class_recall"),
            "full_version_space_retention": _mean(rows, "full_version_space_retention"),
            "mean_candidate_set_size": _mean(rows, "candidate_set_size"),
            "singleton_under_ambiguous_truth_rate": _mean(rows, "singleton_under_ambiguous_truth"),
            "unsafe_singleton_collapse_rate": _mean(rows, "unsafe_singleton_collapse"),
            "exact_decision_rate": _mean(rows, "exact_decision"),
            "mean_decision_regret": _mean(rows, "decision_regret"),
        }
    strata: dict[str, Any] = {}
    primary_rows = [
        row for row in observations
        if row["role"] == "EVALUATION"
        and row["method_id"] == config["residualDefinition"]["methodId"]
        and row["budget"] == config["primaryAcceptedBudget"]
    ]
    for mode, stratum in itertools.product(config["metrics"]["requiredByEvidenceMode"], config["metrics"]["eventStrata"]):
        rows = [row for row in primary_rows if row["evidence_mode"] == mode and row["event_stratum"] == stratum]
        strata[f"{mode}|{stratum}"] = {
            "record_count": len(rows), "oracle_class_recall": _mean(rows, "oracle_class_recall"),
            "full_version_space_retention": _mean(rows, "full_version_space_retention"),
            "unsafe_singleton_collapse_rate": _mean(rows, "unsafe_singleton_collapse"),
            "exact_decision_rate": _mean(rows, "exact_decision"),
            "mean_decision_regret": _mean(rows, "decision_regret"),
        }
    residual_groups = sorted({row["group_id"] for row in primary_rows if row["decision_relevant_missing"]})
    fail_closed_rows = [row for row in primary_rows if row["conflict_or_insufficient_requires_fail_closed"]]
    metrics: dict[str, Any] = {
        "development_group_count": len(set(role_manifest["calibration_group_ids"]) | set(role_manifest["evaluation_group_ids"])),
        "calibration_group_count": len(role_manifest["calibration_group_ids"]),
        "evaluation_group_count": len(role_manifest["evaluation_group_ids"]),
        "role_overlap_count": role_manifest["group_overlap_count"],
        "role_source_accounting_exact": role_manifest["source_group_accounting_exact"],
        "observation_count": len(observations),
        "metric_cell_count": sum(cell["record_count"] > 0 for cell in cells.values()),
        "expected_metric_cell_count": expected_cells,
        "metric_cell_coverage": sum(cell["record_count"] > 0 for cell in cells.values()) / expected_cells,
        "candidate_class_validity": _mean(observations, "candidate_classes_valid"),
        "candidate_budget_compliance": _mean(observations, "candidate_budget_compliant"),
        "atomic_family_expansion_accuracy": _mean([row for row in observations if row["method_id"] != "M0_NORMALIZED_EXACT"], "atomic_family_expansion"),
        "contradiction_fail_closed_accuracy": _mean(fail_closed_rows, "fail_closed") if fail_closed_rows else 1.0,
        "unsafe_singleton_collapse_rate": _mean(primary_rows, "unsafe_singleton_collapse"),
        "primary_evaluation_mean_decision_regret": _mean(primary_rows, "decision_regret"),
        "primary_evaluation_oracle_class_recall": _mean(primary_rows, "oracle_class_recall"),
        "primary_evaluation_full_version_space_retention": _mean(primary_rows, "full_version_space_retention"),
        "primary_evaluation_exact_decision_rate": _mean(primary_rows, "exact_decision"),
        "evaluation_tuning_count": role_manifest["evaluation_tuning_count"],
        "residual_evaluation_group_ids": residual_groups,
        "residual_evaluation_group_count": len(residual_groups),
        "model_eligible_residual": len(residual_groups) >= config["residualDefinition"]["minimumModelEligibleResidualGroupCount"],
        "cells": cells,
        "primary_evaluation_strata": strata,
    }
    metrics["finite_metrics"] = _finite(metrics)
    residual = {
        "schema_version": "221-deterministic-mondo-residual-manifest",
        "method_id": config["residualDefinition"]["methodId"],
        "budget": config["residualDefinition"]["budget"],
        "residual_evaluation_group_ids": residual_groups,
        "residual_evaluation_group_count": len(residual_groups),
        "minimum_model_eligible_residual_group_count": config["residualDefinition"]["minimumModelEligibleResidualGroupCount"],
        "model_eligible_residual": metrics["model_eligible_residual"],
        "protected_groups_evaluated": 0,
    }
    return metrics, residual


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit_evaluation(metrics: dict[str, Any], catalog_manifest: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["evaluationGates"]
    checks = {
        "development_groups_and_roles_are_exact_and_disjoint": bool(
            metrics["development_group_count"] == config["inputContract"]["expectedDevelopmentGroupCount"]
            and metrics["calibration_group_count"] == config["roleSplit"]["expectedCalibrationGroupCount"]
            and metrics["evaluation_group_count"] == config["roleSplit"]["expectedEvaluationGroupCount"]
            and metrics["role_overlap_count"] == 0
            and metrics["role_source_accounting_exact"]
        ),
        "catalog_and_candidates_are_exact_valid_bounded_and_atomic": bool(
            catalog_manifest["state_class_accuracy"] == gates["requiredCatalogStateClassAccuracy"]
            and metrics["candidate_class_validity"] == gates["requiredCandidateClassValidity"]
            and metrics["candidate_budget_compliance"] == gates["requiredCandidateBudgetCompliance"]
            and metrics["atomic_family_expansion_accuracy"] == gates["requiredAtomicFamilyExpansionAccuracy"]
        ),
        "fail_closed_controller_is_safe_and_within_frozen_regret_bound": bool(
            metrics["contradiction_fail_closed_accuracy"] == gates["requiredContradictionFailClosedAccuracy"]
            and metrics["unsafe_singleton_collapse_rate"] <= gates["maximumUnsafeSingletonCollapseRate"]
            and metrics["primary_evaluation_mean_decision_regret"] <= gates["maximumPrimaryEvaluationMeanDecisionRegret"]
        ),
        "metric_cells_no_tuning_and_finite_outputs_are_complete": bool(
            metrics["metric_cell_coverage"] == gates["requiredMetricCellCoverage"]
            and (metrics["evaluation_tuning_count"] == 0) == gates["requiredNoEvaluationTuning"]
            and metrics["finite_metrics"] == gates["requiredFiniteMetrics"]
        ),
    }
    limits = config["accessGates"]
    access_checks = {
        "one_catalog_and_development_evaluation_with_exact_input_loads": bool(
            access["catalog_build_count"] == limits["requiredCatalogBuildCount"]
            and access["development_evaluation_run_count"] == limits["requiredDevelopmentEvaluationRunCount"]
            and access["development_public_load_count"] == limits["requiredDevelopmentPublicLoadCount"]
            and access["development_truth_load_count"] == limits["requiredDevelopmentTruthLoadCount"]
        ),
        "protected_prior_model_network_and_effect_boundaries_are_zero": bool(
            access["protected_public_load_count"] <= limits["maximumProtectedPublicLoadCount"]
            and access["protected_truth_load_count"] <= limits["maximumProtectedTruthLoadCount"]
            and access["v218_development_record_read_count"] <= limits["maximumV218DevelopmentRecordReadCount"]
            and access["v218_protected_record_read_count"] <= limits["maximumV218ProtectedRecordReadCount"]
            and access["v216_protected_access_count"] <= limits["maximumV216ProtectedAccessCount"]
            and access["v213_protected_access_count"] <= limits["maximumV213ProtectedAccessCount"]
            and access["model_load_count"] <= limits["maximumModelLoadCount"]
            and access["model_generation_count"] <= limits["maximumModelGenerationCount"]
            and access["model_api_call_count"] <= limits["maximumModelAPICallCount"]
            and access["training_run_count"] <= limits["maximumTrainingRunCount"]
            and access["network_request_count"] <= limits["maximumNetworkRequestCount"]
            and access["ontology_registration_count"] <= limits["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= limits["maximumTrustedStateMutationCount"]
            and access["service_action_count"] <= limits["maximumServiceActionCount"]
            and access["external_side_effect_count"] <= limits["maximumExternalSideEffectCount"]
            and access["actual_execution_count"] <= limits["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    if not passed:
        branch = "NEGATIVE_DETERMINISTIC_RESIDUAL_INTEGRITY_OR_SAFETY"
        decision = config["decisionRule"]["ifIntegritySafetyAndEvaluationGatesFail"]
    elif metrics["model_eligible_residual"]:
        branch = "MEANINGFUL_DETERMINISTIC_RESIDUAL_MODEL_DESIGN_ELIGIBLE"
        decision = config["decisionRule"]["ifGatesPassAndResidualAtLeastTwelveGroups"]
    else:
        branch = "DETERMINISTIC_SUFFICIENT_CLOSE_MODEL_ESCALATION"
        decision = config["decisionRule"]["ifGatesPassAndResidualBelowTwelveGroups"]
    return {"passed": passed, "branch": branch, "decision": decision, "checks": checks, "access_checks": access_checks}
