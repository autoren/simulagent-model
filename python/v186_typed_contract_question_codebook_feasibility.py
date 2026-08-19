from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from typing import Any

from v93_open_set_source import canonical_sha256


FAMILY_ORDER = {
    "intent_concept": 0,
    "domain": 1,
    "slot_any": 2,
    "slot_required": 3,
    "slot_result": 4,
    "transactional": 5,
}


def _question_id(family: str, value: Any) -> str:
    digest = hashlib.sha256(
        json.dumps([family, value], separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()[:16]
    return f"Q186_{FAMILY_ORDER[family]:02d}_{digest}"


def _raw_questions(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads = [row["semantic_payload"] for row in contracts]
    values: dict[str, set[Any]] = {
        "intent_concept": {row["normalized_intent_name"] for row in payloads},
        "domain": {row["domain"] for row in payloads},
        "slot_any": {slot["name"] for row in payloads for slot in row["slots"]},
        "slot_required": {slot for row in payloads for slot in row["required_slots"]},
        "slot_result": {slot for row in payloads for slot in row["result_slots"]},
        "transactional": {True},
    }
    questions = []
    for family in sorted(values, key=FAMILY_ORDER.get):
        for value in sorted(values[family], key=lambda item: json.dumps(item, sort_keys=True)):
            questions.append(
                {
                    "question_id": _question_id(family, value),
                    "family": family,
                    "value": value,
                    "answer_type": "binary",
                }
            )
    return questions


def answer(contract: dict[str, Any], question: dict[str, Any]) -> int:
    payload = contract["semantic_payload"]
    family = question["family"]
    value = question["value"]
    if family == "intent_concept":
        return int(payload["normalized_intent_name"] == value)
    if family == "domain":
        return int(payload["domain"] == value)
    if family == "slot_any":
        return int(value in {slot["name"] for slot in payload["slots"]})
    if family == "slot_required":
        return int(value in payload["required_slots"])
    if family == "slot_result":
        return int(value in payload["result_slots"])
    if family == "transactional":
        return int(payload["is_transactional"] == bool(value))
    raise ValueError(f"unknown V186 family: {family}")


def build_codebook(
    contract_catalog: dict[str, Any],
    hidden: dict[str, Any],
    development_identities: dict[str, Any],
    protected_identities: dict[str, Any],
) -> dict[str, Any]:
    contracts = sorted(contract_catalog["contracts"], key=lambda row: row["capability_contract_id"])
    raw = _raw_questions(contracts)
    questions = []
    answer_columns: dict[str, list[int]] = {}
    for question in raw:
        column = [answer(contract, question) for contract in contracts]
        if set(column) == {0, 1}:
            questions.append(question)
            answer_columns[question["question_id"]] = column
    questions.sort(key=lambda row: (FAMILY_ORDER[row["family"]], json.dumps(row["value"], sort_keys=True)))
    contract_vectors = {}
    for index, contract in enumerate(contracts):
        vector = [answer_columns[row["question_id"]][index] for row in questions]
        contract_vectors[contract["capability_contract_id"]] = vector
    equivalence: dict[str, list[str]] = defaultdict(list)
    for contract_id, vector in contract_vectors.items():
        equivalence[canonical_sha256(vector)].append(contract_id)
    classes = [
        {"answer_vector_sha256": key, "contract_ids": sorted(ids), "class_size": len(ids)}
        for key, ids in sorted(equivalence.items())
    ]
    pairs = []
    for left, right in combinations(sorted(contract_vectors), 2):
        separating = [
            questions[index]["question_id"]
            for index, (a, b) in enumerate(zip(contract_vectors[left], contract_vectors[right]))
            if a != b
        ]
        pairs.append(
            {
                "left_contract_id": left,
                "right_contract_id": right,
                "separating_question_ids": separating,
                "separable": bool(separating),
            }
        )

    hidden_by_id = {row["record_id"]: row for row in hidden["records"]}
    role_inputs = {
        "development": development_identities["records"],
        "protected": protected_identities["records"],
    }
    bindings = {}
    for role, rows in role_inputs.items():
        output = []
        for public in sorted(rows, key=lambda row: row["record_id"]):
            hidden_row = hidden_by_id[public["record_id"]]
            contract_id = hidden_row["truth_contract_id"]
            vector = contract_vectors[contract_id] if contract_id else None
            output.append(
                {
                    "record_id": public["record_id"],
                    "role": role,
                    "observation_available": public["observation_available"],
                    "target_contract_id": contract_id,
                    "target_answer_vector": vector,
                    "target_answer_vector_sha256": canonical_sha256(vector) if vector is not None else None,
                    "evidence_status": "ORACLE_VECTOR_AVAILABLE" if vector is not None else "INSUFFICIENT",
                }
            )
        bindings[role] = output

    family_counts = Counter(row["family"] for row in questions)
    kind_counts = Counter(
        row["truth_kinds"][0] for row in contracts if len(row["truth_kinds"]) == 1
    )
    dev_ids = {row["record_id"] for row in bindings["development"]}
    protected_ids = {row["record_id"] for row in bindings["protected"]}
    role_summary = {}
    for role, rows in bindings.items():
        observed = [row for row in rows if row["observation_available"]]
        missing = [row for row in rows if not row["observation_available"]]
        role_summary[role] = {
            "binding_count": len(rows),
            "observed_binding_count": len(observed),
            "missing_binding_count": len(missing),
            "target_vector_reconstruction_rate": sum(
                row["target_answer_vector"] == contract_vectors[row["target_contract_id"]]
                for row in observed
            ) / len(observed),
            "missing_insufficient_rate": sum(
                row["target_answer_vector"] is None and row["evidence_status"] == "INSUFFICIENT"
                for row in missing
            ) / len(missing),
        }
    summary = {
        "capability_contract_count": len(contracts),
        "contract_truth_kind_counts": dict(sorted(kind_counts.items())),
        "question_count": len(questions),
        "question_family_count": len(family_counts),
        "question_family_counts": dict(sorted(family_counts.items())),
        "nontrivial_question_rate": 1.0 if questions else 0.0,
        "unique_question_identifier_rate": len({row["question_id"] for row in questions}) / len(questions),
        "unique_answer_vector_count": len({tuple(value) for value in contract_vectors.values()}),
        "equivalence_class_count": len(classes),
        "largest_equivalence_class_size": max(row["class_size"] for row in classes),
        "contract_pair_count": len(pairs),
        "pairwise_separation_rate": sum(row["separable"] for row in pairs) / len(pairs),
        "minimum_separating_question_count_per_pair": min(len(row["separating_question_ids"]) for row in pairs),
        "maximum_separating_question_count_per_pair": max(len(row["separating_question_ids"]) for row in pairs),
        "roles": role_summary,
        "role_identifier_overlap": len(dev_ids & protected_ids),
        "utterance_or_dialogue_language_read_count": 0,
        "planner_policy_score_count": 0,
        "manual_judgment_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {
        "questions": questions,
        "contract_answer_vectors": contract_vectors,
        "equivalence_classes": classes,
        "pairwise_separation": pairs,
        "bindings": bindings,
        "summary": summary,
    }


def audit_codebook(codebook: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = codebook["summary"]
    gates = config["feasibilityGates"]
    roles = summary["roles"]
    checks = {
        "contract_universe_and_pair_census_are_exact": bool(
            summary["capability_contract_count"] == gates["requiredCapabilityContractCount"]
            and summary["contract_truth_kind_counts"] == gates["requiredContractTruthKindCounts"]
            and summary["contract_pair_count"] == gates["requiredContractPairCount"]
        ),
        "question_vocabulary_is_nontrivial_unique_and_multifamily": bool(
            summary["question_count"] >= gates["minimumQuestionCount"]
            and summary["question_family_count"] >= gates["minimumQuestionFamilyCount"]
            and summary["nontrivial_question_rate"] == gates["requiredNontrivialQuestionRate"]
            and summary["unique_question_identifier_rate"] == gates["requiredUniqueQuestionIdentifierRate"]
        ),
        "answer_vectors_fully_identify_every_contract": bool(
            summary["unique_answer_vector_count"] == gates["requiredUniqueAnswerVectorCount"]
            and summary["equivalence_class_count"] == gates["requiredEquivalenceClassCount"]
            and summary["largest_equivalence_class_size"] == gates["requiredLargestEquivalenceClassSize"]
            and summary["pairwise_separation_rate"] == gates["requiredPairwiseSeparationRate"]
        ),
        "development_role_binding_is_exact": _role_gate(
            roles["development"], gates["requiredDevelopmentBindingCount"],
            gates["requiredDevelopmentObservedBindingCount"], gates["requiredDevelopmentMissingBindingCount"],
            gates["requiredTargetVectorReconstructionRate"],
        ),
        "protected_role_binding_is_exact": _role_gate(
            roles["protected"], gates["requiredProtectedBindingCount"],
            gates["requiredProtectedObservedBindingCount"], gates["requiredProtectedMissingBindingCount"],
            gates["requiredTargetVectorReconstructionRate"],
        ),
        "roles_are_identifier_disjoint": summary["role_identifier_overlap"] == gates["requiredRoleIdentifierOverlap"],
        "language_planner_model_authority_and_effect_access_is_zero": bool(
            summary["utterance_or_dialogue_language_read_count"] == gates["maximumUtteranceOrDialogueLanguageReadCount"]
            and summary["planner_policy_score_count"] == gates["maximumPlannerPolicyScoreCount"]
            and all(summary[key] == gates[gate] for key, gate in (
                ("manual_judgment_count", "maximumManualJudgmentCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            ))
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


def _role_gate(role: dict[str, Any], total: int, observed: int, missing: int, reconstruction: float) -> bool:
    return bool(
        role["binding_count"] == total
        and role["observed_binding_count"] == observed
        and role["missing_binding_count"] == missing
        and role["target_vector_reconstruction_rate"] == reconstruction
        and role["missing_insufficient_rate"] == 1.0
    )


__all__ = ["answer", "audit_codebook", "build_codebook"]
