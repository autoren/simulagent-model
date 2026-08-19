from __future__ import annotations

from collections import Counter
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v126_sgd_retrieval_selectivity import (
    bayes_policy, catalog_maps, hypothesis_action_cost, joint_distribution,
    normalized_kind, truth_choice,
)


def select_fresh_population(
    inventory: dict[str, Any], excluded: dict[str, Any], catalog: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["freshPopulation"]
    excluded_ids = {row["candidate_id"] for row in excluded["evaluation_population"]}
    test = [
        row for row in inventory["candidate_index"]
        if row["partition"] == spec["sourcePartition"] and row["candidate_id"] not in excluded_ids
    ]
    known_pairs = catalog["declared_known_pairs"]
    visible_domains = catalog["visible_domains"]
    unsupported_domains = next(row["domains"] for row in catalog["choices"] if row["kind"] == "UNSUPPORTED_COMPOSITE")
    selected: list[dict[str, Any]] = []

    def take(rows: list[dict[str, Any]], count: int, group: str) -> None:
        ordered = sorted(rows, key=lambda row: hash_order(spec["baseSalt"], group, row["candidate_id"]))
        if len(ordered) < count:
            raise ValueError(f"insufficient V127 group: {group}")
        selected.extend(ordered[:count])

    for pair in known_pairs:
        take(
            [row for row in test if row["class_label"] == "known" and f"{row['service']}::{row['intent']}" == pair],
            spec["knownRecordCountPerDeclaredPair"], f"known::{pair}",
        )
    for domain in visible_domains:
        take(
            [row for row in test if row["class_label"] == "novel_valid" and row["domain"] == domain],
            spec["novelValidRecordCountPerVisibleDomain"], f"novel::{domain}",
        )
    for domain in unsupported_domains:
        take(
            [row for row in test if row["class_label"] == "unsupported" and row["domain"] == domain],
            spec["unsupportedRecordCountPerDomain"], f"unsupported::{domain}",
        )
    records = [
        {"record_id": f"v127::evaluation::{row['candidate_id']}", **row}
        for row in sorted(selected, key=lambda item: item["candidate_id"])
    ]
    counts = Counter(row["class_label"] for row in records)
    output = {
        "record_count": len(records),
        "class_counts": dict(sorted(counts.items())),
        "known_pair_coverage": len({f"{row['service']}::{row['intent']}" for row in records if row["class_label"] == "known"}),
        "novel_domain_coverage": len({row["domain"] for row in records if row["class_label"] == "novel_valid"}),
        "unsupported_domain_coverage": len({row["domain"] for row in records if row["class_label"] == "unsupported"}),
        "excluded_identifier_overlap_count": sum(row["candidate_id"] in excluded_ids for row in records),
        "contains_language": False,
        "records_sha256": canonical_sha256(records),
        "records": records,
    }
    return output


def population_gates(population: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    spec = config["freshPopulation"]
    return {
        "record_count": population["record_count"] == spec["requiredRecordCount"],
        "balanced_classes": all(population["class_counts"].get(label) == spec["requiredRecordCountPerClass"] for label in ("known", "novel_valid", "unsupported")),
        "known_pair_coverage": population["known_pair_coverage"] == spec["requiredKnownPairCoverage"],
        "novel_domain_coverage": population["novel_domain_coverage"] == spec["requiredNovelDomainCoverage"],
        "unsupported_domain_coverage": population["unsupported_domain_coverage"] == spec["requiredUnsupportedDomainCoverage"],
        "fresh_from_V125": population["excluded_identifier_overlap_count"] == 0,
        "text_free": not population["contains_language"],
    }


def signature_decision(
    observed_slots: set[str], signatures: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    ranked = []
    compatible = []
    for intent_id, signature in sorted(signatures.items()):
        allowed = signature["allowed"]
        required = signature["required"]
        if observed_slots and observed_slots <= allowed:
            compatible.append(intent_id)
        ranked.append((len(observed_slots - allowed), len(required - observed_slots), -len(observed_slots & allowed), intent_id))
    if not ranked:
        raise ValueError("V127 requires declared known signatures")
    candidate = compatible[0] if len(compatible) == 1 else min(ranked)[-1]
    return {
        "candidate_intent": candidate,
        "query": len(compatible) != 1,
        "compatible_count": len(compatible),
        "evidence_present": bool(observed_slots),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    known = [row for row in rows if row["kind"] == "KNOWN"]
    unsupported = [row for row in rows if row["kind"] == "UNSUPPORTED"]
    nonknown = [row for row in rows if row["kind"] != "KNOWN"]
    return {
        "mean_regret": sum(row["cost"] for row in rows) / len(rows),
        "known_exact_probability": sum(row["known_exact"] for row in known) / len(known),
        "unsupported_correct_probability": sum(row["unsupported_correct"] for row in unsupported) / len(unsupported),
        "false_known_probability": sum(row["false_known"] for row in nonknown) / len(nonknown),
    }


def evaluate_condition(
    records: list[dict[str, Any]], evidence: dict[str, set[str]], signatures: dict[str, dict[str, set[str]]],
    catalog: dict[str, Any], baseline: dict[str, Any], v119: dict[str, Any], config: dict[str, Any],
    candidate_probability: float, correlation: float,
) -> dict[str, Any]:
    _, by_id, _ = catalog_maps(catalog)
    known_choice = {row["intent_id"]: row["choice_id"] for row in catalog["choices"] if row["kind"] == "KNOWN"}
    reliability = config["queryChannel"]["marginalCorrectness"]
    fee = config["queryChannel"]["totalCost"]
    decisions = {row["record_id"]: signature_decision(evidence[row["record_id"]], signatures) for row in records}
    policies = {
        choice: bayes_policy(choice, reliability, correlation, candidate_probability, catalog, baseline, v119)
        for choice in sorted({known_choice[row["candidate_intent"]] for row in decisions.values()})
    }
    always_rows, selective_rows, values = [], [], []
    skip_correct = 0
    skip_class_counts: Counter[str] = Counter()
    for record in records:
        identifier = record["record_id"]
        decision = decisions[identifier]
        candidate = known_choice[decision["candidate_intent"]]
        truth = truth_choice(record, catalog)
        kind = normalized_kind(by_id[truth])
        exact = ("KNOWN", by_id[truth]["intent_id"]) if kind == "KNOWN" else ("UNSUPPORTED", None) if kind == "UNSUPPORTED" else ("ABSTAIN", None)
        skip_action = ("KNOWN", decision["candidate_intent"])
        skip_cost = hypothesis_action_cost(truth, skip_action, by_id, baseline)
        query_cost = query_known = query_unsupported = query_false_known = 0.0
        for observed, probability in joint_distribution(truth, candidate, reliability, correlation, by_id, v119).items():
            action = policies[candidate][observed]
            query_cost += probability * hypothesis_action_cost(truth, action, by_id, baseline)
            query_known += probability * (kind == "KNOWN" and action == exact)
            query_unsupported += probability * (kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
            query_false_known += probability * (kind != "KNOWN" and action[0] == "KNOWN")
        no_query = {
            "kind": kind, "cost": skip_cost,
            "known_exact": float(kind == "KNOWN" and skip_action == exact),
            "unsupported_correct": 0.0,
            "false_known": float(kind != "KNOWN"),
        }
        always = {
            "kind": kind, "cost": query_cost + fee, "known_exact": query_known,
            "unsupported_correct": query_unsupported, "false_known": query_false_known,
        }
        selective = always if decision["query"] else no_query
        always_rows.append(always); selective_rows.append(selective)
        values.append({"query": decision["query"], "query_value": skip_cost - query_cost})
        if not decision["query"]:
            skip_class_counts[kind] += 1
            skip_correct += int(skip_action == exact)
    queried = [row for row in values if row["query"]]
    skipped = [row for row in values if not row["query"]]
    return {
        "record_count": len(records),
        "typed_evidence_presence_fraction": sum(bool(evidence[row["record_id"]]) for row in records) / len(records),
        "skip_fraction": len(skipped) / len(values),
        "query_fraction": len(queried) / len(values),
        "skip_class_counts": dict(sorted(skip_class_counts.items())),
        "skipped_action_precision": skip_correct / len(skipped) if skipped else 0.0,
        "queried_average_query_value": sum(row["query_value"] for row in queried) / len(queried) if queried else 0.0,
        "skipped_average_query_value": sum(row["query_value"] for row in skipped) / len(skipped) if skipped else 0.0,
        "ask_always": {"mean_regret": config["outcomeGates"]["maximumSelectiveMeanRegretEveryPriorAndCorrelation"]},
        "always_query": summarize(always_rows),
        "selective_query": summarize(selective_rows),
    }


def run_evaluation(
    records: list[dict[str, Any]], evidence: dict[str, set[str]], signatures: dict[str, dict[str, set[str]]],
    catalog: dict[str, Any], baseline: dict[str, Any], v119: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    conditions = {}
    for prior in config["queryChannel"]["priorRegimes"]:
        for correlation in config["queryChannel"]["sharedFailureCorrelations"]:
            key = f"{prior['id']}@{correlation:.2f}"
            conditions[key] = evaluate_condition(records, evidence, signatures, catalog, baseline, v119, config, prior["candidateProbability"], correlation)
    gates = config["outcomeGates"]
    selective = [row["selective_query"] for row in conditions.values()]
    checks = {
        "selective_regret_every_prior_and_correlation": all(row["mean_regret"] <= gates["maximumSelectiveMeanRegretEveryPriorAndCorrelation"] for row in selective),
        "selective_known_exact_every_prior_and_correlation": all(row["known_exact_probability"] >= gates["minimumSelectiveKnownExactEveryPriorAndCorrelation"] for row in selective),
        "selective_unsupported_every_prior_and_correlation": all(row["unsupported_correct_probability"] >= gates["minimumSelectiveUnsupportedCorrectEveryPriorAndCorrelation"] for row in selective),
        "selective_false_known_every_prior_and_correlation": all(row["false_known_probability"] <= gates["maximumSelectiveFalseKnownEveryPriorAndCorrelation"] for row in selective),
        "nontrivial_skip_fraction": all(gates["minimumSkipFraction"] <= row["skip_fraction"] <= gates["maximumSkipFraction"] for row in conditions.values()),
        "skipped_action_precision": all(row["skipped_action_precision"] >= gates["minimumSkippedActionPrecision"] for row in conditions.values()),
        "queried_average_value_covers_cost": all(row["queried_average_query_value"] >= gates["minimumQueriedAverageQueryValueEveryPriorAndCorrelation"] for row in conditions.values()),
        "skipped_average_value_not_above_cost": all(row["skipped_average_query_value"] <= gates["maximumSkippedAverageQueryValueEveryPriorAndCorrelation"] for row in conditions.values()),
        "selective_no_worse_than_always_query": all(row["selective_query"]["mean_regret"] <= row["always_query"]["mean_regret"] for row in conditions.values()),
        "one_rule_zero_fit_selection_and_thresholds": config["typedConstraintMechanism"]["candidateCount"] == 1 and config["typedConstraintMechanism"]["fitCount"] == config["typedConstraintMechanism"]["selectionCount"] == config["typedConstraintMechanism"]["thresholdCount"] == 0,
        "schema_signature_count": len(signatures) == gates["requiredSchemaSignatureCount"],
        "complete_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "zero_individual_record_emission": gates["maximumIndividualRecordEmissionCount"] == 0,
        "zero_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "record_count": len(records), "schema_signature_count": len(signatures), "conditions": conditions,
        "outcome_gates": checks, "outcome_pass": passed,
        "decision": config["decisionRule"]["ifEveryOutcomeAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "primary_rule_count": 1, "fit_count": 0, "selection_count": 0, "threshold_count": 0,
        "true_hypothesis_retention": 1.0, "individual_record_emission_count": 0, "actual_execution_count": 0,
    }


__all__ = ["population_gates", "run_evaluation", "select_fresh_population", "signature_decision"]
