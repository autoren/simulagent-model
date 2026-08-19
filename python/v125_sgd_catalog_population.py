from __future__ import annotations

from collections import Counter
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order


def build_catalog(inventory: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = config["catalog"]
    known_pairs = spec["declaredKnownPairs"]
    choices = []
    for index, pair in enumerate(known_pairs, start=1):
        service, intent = pair.split("::", 1)
        domain = next(
            row["domain"] for row in inventory["candidate_index"]
            if row["service"] == service and row["intent"] == intent
        )
        choices.append({"choice_id": f"K{index:02d}", "kind": "KNOWN", "domain": domain, "service": service, "intent": intent, "intent_id": pair})
    for index, domain in enumerate(spec["visibleDomains"], start=1):
        members = sorted({
            f"{row['service']}::{row['intent']}"
            for row in inventory["candidate_index"]
            if row["partition"] == "test" and row["class_label"] == "novel_valid" and row["domain"] == domain
        })
        choices.append({"choice_id": f"N{index:02d}", "kind": "NOVEL_COMPOSITE", "domain": domain, "member_intent_ids": members})
    unsupported_domains = sorted({
        row["domain"] for row in inventory["candidate_index"]
        if row["partition"] == "test" and row["class_label"] == "unsupported"
    })
    choices.append({"choice_id": "U00", "kind": "UNSUPPORTED_COMPOSITE", "domains": unsupported_domains})
    choices.append({"choice_id": "A00", "kind": "ABSTAIN", "meaning": "insufficient evidence"})
    return {
        "choice_count": len(choices),
        "choices": choices,
        "visible_domains": spec["visibleDomains"],
        "declared_known_pairs": known_pairs,
        "complete_safe_composite_hypothesis_universe": spec["completeSafeCompositeHypothesisUniverse"],
        "contains_language": False,
        "catalog_sha256": canonical_sha256(choices),
    }


def select_populations(inventory: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    known_pairs = set(config["catalog"]["declaredKnownPairs"])
    training = [
        {"population_id": f"v125::train::{row['candidate_id']}", **row}
        for row in inventory["candidate_index"]
        if row["partition"] == config["trainingPopulation"]["sourcePartition"]
        and f"{row['service']}::{row['intent']}" in known_pairs
    ]
    evaluation_spec = config["evaluationPopulation"]
    test = [row for row in inventory["candidate_index"] if row["partition"] == evaluation_spec["sourcePartition"]]
    selected: list[dict[str, Any]] = []

    def take(group: list[dict[str, Any]], count: int, group_id: str) -> None:
        ordered = sorted(group, key=lambda row: hash_order(evaluation_spec["baseSalt"], group_id, row["candidate_id"]))
        if len(ordered) < count:
            raise ValueError(f"insufficient V125 group: {group_id}")
        selected.extend(ordered[:count])

    for pair in config["catalog"]["declaredKnownPairs"]:
        take([row for row in test if row["class_label"] == "known" and f"{row['service']}::{row['intent']}" == pair], evaluation_spec["knownRecordCountPerDeclaredPair"], f"known::{pair}")
    for domain in config["catalog"]["visibleDomains"]:
        take([row for row in test if row["class_label"] == "novel_valid" and row["domain"] == domain], evaluation_spec["novelValidRecordCountPerVisibleDomain"], f"novel::{domain}")
    unsupported_domains = sorted({row["domain"] for row in test if row["class_label"] == "unsupported"})
    for domain in unsupported_domains:
        take([row for row in test if row["class_label"] == "unsupported" and row["domain"] == domain], evaluation_spec["unsupportedRecordCountPerDomain"], f"unsupported::{domain}")

    evaluation = [{"population_id": f"v125::evaluation::{row['candidate_id']}", **row} for row in selected]
    training.sort(key=lambda row: row["population_id"])
    evaluation.sort(key=lambda row: row["population_id"])
    train_ids = {row["candidate_id"] for row in training}
    eval_ids = {row["candidate_id"] for row in evaluation}
    return {
        "training_record_count": len(training),
        "evaluation_record_count": len(evaluation),
        "evaluation_class_counts": dict(sorted(Counter(row["class_label"] for row in evaluation).items())),
        "known_pair_coverage": len({f"{row['service']}::{row['intent']}" for row in evaluation if row["class_label"] == "known"}),
        "novel_domain_coverage": len({row["domain"] for row in evaluation if row["class_label"] == "novel_valid"}),
        "unsupported_domain_coverage": len({row["domain"] for row in evaluation if row["class_label"] == "unsupported"}),
        "training_evaluation_identifier_overlap_count": len(train_ids & eval_ids),
        "contains_language": False,
        "training_population_sha256": canonical_sha256(training),
        "evaluation_population_sha256": canonical_sha256(evaluation),
        "training_population": training,
        "evaluation_population": evaluation,
    }


def evaluate_gates(catalog: dict[str, Any], populations: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["populationGates"]
    kinds = Counter(row["kind"] for row in catalog["choices"])
    return {
        "choice_count": catalog["choice_count"] == gates["requiredChoiceCount"],
        "known_choice_count": kinds["KNOWN"] == gates["requiredKnownChoiceCount"],
        "novel_composite_choice_count": kinds["NOVEL_COMPOSITE"] == gates["requiredNovelCompositeChoiceCount"],
        "unsupported_choice_count": kinds["UNSUPPORTED_COMPOSITE"] == gates["requiredUnsupportedChoiceCount"],
        "insufficient_choice_count": kinds["ABSTAIN"] == gates["requiredInsufficientChoiceCount"],
        "training_record_count": populations["training_record_count"] == gates["requiredTrainingRecordCount"],
        "evaluation_record_count": populations["evaluation_record_count"] == gates["requiredEvaluationRecordCount"],
        "balanced_evaluation_classes": all(populations["evaluation_class_counts"].get(label) == gates["requiredEvaluationRecordCountPerClass"] for label in ("known", "novel_valid", "unsupported")),
        "known_pair_coverage": populations["known_pair_coverage"] == gates["requiredKnownPairCoverage"],
        "novel_domain_coverage": populations["novel_domain_coverage"] == gates["requiredNovelDomainCoverage"],
        "unsupported_domain_coverage": populations["unsupported_domain_coverage"] == gates["requiredUnsupportedDomainCoverage"],
        "training_evaluation_disjoint": gates["requireTrainingEvaluationIdentifierDisjointness"] and populations["training_evaluation_identifier_overlap_count"] == 0,
        "text_free": not catalog["contains_language"] and not populations["contains_language"],
        "zero_language_model_execution": gates["maximumEmittedLanguageRecordCount"] == gates["maximumManualLanguageInspectionCount"] == gates["maximumModelLoadCount"] == gates["maximumModelGenerationCount"] == gates["maximumActualExecutionCount"] == 0,
    }


__all__ = ["build_catalog", "evaluate_gates", "select_populations"]
