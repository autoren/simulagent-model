from __future__ import annotations

from collections import Counter
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v127_sgd_typed_constraint_feasibility import run_evaluation


def select_fresh_population(
    inventory: dict[str, Any], excluded_populations: list[dict[str, Any]],
    catalog: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["freshPopulation"]
    excluded_ids = set()
    for population in excluded_populations:
        rows = population.get("evaluation_population", population.get("records", []))
        excluded_ids.update(row["candidate_id"] for row in rows)
    source = [
        row for row in inventory["candidate_index"]
        if row["partition"] == spec["sourcePartition"] and row["candidate_id"] not in excluded_ids
    ]
    selected: list[dict[str, Any]] = []

    def take(rows: list[dict[str, Any]], count: int, group: str) -> None:
        ordered = sorted(rows, key=lambda row: hash_order(spec["baseSalt"], group, row["candidate_id"]))
        if len(ordered) < count: raise ValueError(f"insufficient V128 group: {group}")
        selected.extend(ordered[:count])

    for pair in catalog["declared_known_pairs"]:
        take([row for row in source if row["class_label"] == "known" and f"{row['service']}::{row['intent']}" == pair], spec["knownRecordCountPerDeclaredPair"], f"known::{pair}")
    for domain in catalog["visible_domains"]:
        take([row for row in source if row["class_label"] == "novel_valid" and row["domain"] == domain], spec["novelValidRecordCountPerVisibleDomain"], f"novel::{domain}")
    unsupported_domains = next(row["domains"] for row in catalog["choices"] if row["kind"] == "UNSUPPORTED_COMPOSITE")
    for domain in unsupported_domains:
        take([row for row in source if row["class_label"] == "unsupported" and row["domain"] == domain], spec["unsupportedRecordCountPerDomain"], f"unsupported::{domain}")
    records = [{"record_id": f"v128::evaluation::{row['candidate_id']}", **row} for row in sorted(selected, key=lambda item: item["candidate_id"])]
    counts = Counter(row["class_label"] for row in records)
    return {
        "record_count": len(records), "class_counts": dict(sorted(counts.items())),
        "known_pair_coverage": len({f"{row['service']}::{row['intent']}" for row in records if row["class_label"] == "known"}),
        "novel_domain_coverage": len({row["domain"] for row in records if row["class_label"] == "novel_valid"}),
        "unsupported_domain_coverage": len({row["domain"] for row in records if row["class_label"] == "unsupported"}),
        "excluded_identifier_overlap_count": sum(row["candidate_id"] in excluded_ids for row in records),
        "contains_language": False, "records_sha256": canonical_sha256(records), "records": records,
    }


def population_gates(population: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    spec = config["freshPopulation"]
    return {
        "record_count": population["record_count"] == spec["requiredRecordCount"],
        "balanced_classes": all(population["class_counts"].get(label) == spec["requiredRecordCountPerClass"] for label in ("known", "novel_valid", "unsupported")),
        "known_pair_coverage": population["known_pair_coverage"] == spec["requiredKnownPairCoverage"],
        "novel_domain_coverage": population["novel_domain_coverage"] == spec["requiredNovelDomainCoverage"],
        "unsupported_domain_coverage": population["unsupported_domain_coverage"] == spec["requiredUnsupportedDomainCoverage"],
        "fresh_from_all_prior_evaluations": population["excluded_identifier_overlap_count"] == 0,
        "text_free": not population["contains_language"],
    }


def relation_tokens(frame: dict[str, Any]) -> set[str]:
    state = frame.get("state", {})
    tokens = {f"STATE_SLOT::{slot}" for slot in state.get("slot_values", {}).keys()}
    tokens.update(f"REQUEST_SLOT::{slot}" for slot in state.get("requested_slots", []))
    for action in frame.get("actions", []):
        act = action.get("act")
        slot = action.get("slot")
        if act == "INFORM_INTENT" and slot == "intent": continue
        tokens.add(f"ACTION::{act or '<NONE>'}::{slot or '<NONE>'}")
    return tokens


def build_support_signatures(training: list[tuple[str, set[str]]]) -> dict[str, dict[str, set[str]]]:
    support: dict[str, set[str]] = {}
    for intent_id, tokens in training:
        support.setdefault(intent_id, set()).update(tokens)
    return {intent_id: {"required": set(), "allowed": tokens} for intent_id, tokens in sorted(support.items())}


__all__ = ["build_support_signatures", "population_gates", "relation_tokens", "run_evaluation", "select_fresh_population"]
