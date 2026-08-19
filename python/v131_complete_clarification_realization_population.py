from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("evaluation_population", "selected_population", "records"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("excluded population has no recognized record list")


def excluded_identifiers(paths: list[str], root: Path) -> set[str]:
    output: set[str] = set()
    for relative in paths:
        payload = json.loads((root / relative).read_text())
        output.update(row["candidate_id"] for row in _rows(payload))
    return output


def build_catalog(inventory: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = config["catalog"]
    choices: list[dict[str, Any]] = []
    for index, pair in enumerate(spec["declaredKnownPairs"], start=1):
        service, intent = pair.split("::", 1)
        domains = {
            row["domain"] for row in inventory["candidate_index"]
            if row["service"] == service and row["intent"] == intent
        }
        if len(domains) != 1:
            raise ValueError(f"known pair absent or ambiguous: {pair}")
        choices.append({
            "choice_id": f"K{index:02d}", "kind": "KNOWN", "domain": next(iter(domains)),
            "service": service, "intent": intent, "intent_id": pair,
        })
    for index, domain in enumerate(spec["novelCompositeDomains"], start=1):
        members = sorted({
            f"{row['service']}::{row['intent']}" for row in inventory["candidate_index"]
            if row["partition"] == config["population"]["sourcePartition"]
            and row["class_label"] == "novel_valid" and row["domain"] == domain
        })
        if not members:
            raise ValueError(f"novel composite domain absent: {domain}")
        choices.append({
            "choice_id": f"N{index:02d}", "kind": "NOVEL_COMPOSITE", "domain": domain,
            "member_intent_ids": members,
        })
    unsupported_domains = sorted({
        row["domain"] for row in inventory["candidate_index"]
        if row["partition"] == config["population"]["sourcePartition"]
        and row["class_label"] == "unsupported"
    })
    choices.append({"choice_id": "U00", "kind": "UNSUPPORTED_COMPOSITE", "domains": unsupported_domains})
    choices.append({"choice_id": "A00", "kind": "ABSTAIN", "meaning": "insufficient evidence"})
    return {
        "choice_count": len(choices), "choices": choices,
        "visible_domains": spec["visibleDomains"],
        "declared_known_pairs": spec["declaredKnownPairs"],
        "complete_safe_composite_hypothesis_universe": spec["completeSafeCompositeHypothesisUniverse"],
        "contains_language": False, "catalog_sha256": canonical_sha256(choices),
    }


def select_population(
    inventory: dict[str, Any], catalog: dict[str, Any], excluded: set[str], config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["population"]
    rows = [
        row for row in inventory["candidate_index"]
        if row["partition"] == spec["sourcePartition"] and row["candidate_id"] not in excluded
    ]
    known = [choice for choice in catalog["choices"] if choice["kind"] == "KNOWN"]
    novel = [choice for choice in catalog["choices"] if choice["kind"] == "NOVEL_COMPOSITE"]
    candidates = [choice["choice_id"] for choice in known]
    per_truth = spec["recordsPerTruthCandidateCell"] * len(candidates)
    selected_by_truth: dict[str, list[dict[str, Any]]] = {}

    def take(pool: list[dict[str, Any]], count: int, group: str) -> list[dict[str, Any]]:
        ordered = sorted(pool, key=lambda row: hash_order(spec["baseSalt"], group, row["candidate_id"]))
        if len(ordered) < count:
            raise ValueError(f"insufficient V131 group: {group}, have {len(ordered)}, need {count}")
        return ordered[:count]

    for choice in known:
        selected_by_truth[choice["choice_id"]] = take([
            row for row in rows if row["class_label"] == "known"
            and f"{row['service']}::{row['intent']}" == choice["intent_id"]
        ], per_truth, f"truth::{choice['choice_id']}")
    for choice in novel:
        selected_by_truth[choice["choice_id"]] = take([
            row for row in rows if row["class_label"] == "novel_valid" and row["domain"] == choice["domain"]
        ], per_truth, f"truth::{choice['choice_id']}")

    unsupported_domains = next(choice["domains"] for choice in catalog["choices"] if choice["choice_id"] == "U00")
    unsupported: list[dict[str, Any]] = []
    for domain in unsupported_domains:
        unsupported.extend(take([
            row for row in rows if row["class_label"] == "unsupported" and row["domain"] == domain
        ], spec["unsupportedRecordCountPerUnseenDomain"], f"unsupported::{domain}"))
    unsupported.sort(key=lambda row: hash_order(spec["baseSalt"], "truth::U00", row["candidate_id"]))
    if len(unsupported) != per_truth:
        raise ValueError("unsupported composite does not match per-truth count")
    selected_by_truth["U00"] = unsupported

    fixtures: list[dict[str, Any]] = []
    cell_size = spec["recordsPerTruthCandidateCell"]
    for truth in [choice["choice_id"] for choice in catalog["choices"] if choice["choice_id"] != "A00"]:
        ordered = selected_by_truth[truth]
        for candidate_index, candidate in enumerate(candidates):
            for row in ordered[candidate_index * cell_size:(candidate_index + 1) * cell_size]:
                fixtures.append({
                    "fixture_id": f"v131::{truth}::{candidate}::{row['candidate_id']}",
                    "truth_choice_id": truth, "presented_candidate_choice_id": candidate,
                    "observation_available": True, **row,
                })
    for candidate in candidates:
        for index in range(cell_size):
            fixtures.append({
                "fixture_id": f"v131::A00::{candidate}::missing::{index:02d}",
                "truth_choice_id": "A00", "presented_candidate_choice_id": candidate,
                "observation_available": False, "candidate_id": None, "partition": None,
                "domain": None, "service": None, "intent": None, "class_label": "insufficient",
            })
    fixtures.sort(key=lambda row: row["fixture_id"])
    source_ids = [row["candidate_id"] for row in fixtures if row["observation_available"]]
    cells = Counter((row["truth_choice_id"], row["presented_candidate_choice_id"]) for row in fixtures)
    truth_counts = Counter(row["truth_choice_id"] for row in fixtures)
    candidate_counts = Counter(row["presented_candidate_choice_id"] for row in fixtures)
    return {
        "fixture_count": len(fixtures),
        "source_record_count": len(source_ids),
        "missing_control_count": sum(not row["observation_available"] for row in fixtures),
        "cell_count": len(cells),
        "cell_counts": {f"{truth}::{candidate}": count for (truth, candidate), count in sorted(cells.items())},
        "truth_choice_counts": dict(sorted(truth_counts.items())),
        "presented_candidate_counts": dict(sorted(candidate_counts.items())),
        "known_pair_coverage": len({f"{row['service']}::{row['intent']}" for row in fixtures if row["class_label"] == "known"}),
        "novel_domain_coverage": len({row["domain"] for row in fixtures if row["class_label"] == "novel_valid"}),
        "unsupported_domain_coverage": len({row["domain"] for row in fixtures if row["class_label"] == "unsupported"}),
        "excluded_identifier_overlap_count": len(set(source_ids) & excluded),
        "unique_source_identifier_count": len(set(source_ids)),
        "contains_language": False,
        "fixtures_sha256": canonical_sha256(fixtures),
        "fixtures": fixtures,
    }


def evaluate_gates(catalog: dict[str, Any], population: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["populationGates"]
    kinds = Counter(choice["kind"] for choice in catalog["choices"])
    return {
        "choice_count": catalog["choice_count"] == gates["requiredChoiceCount"],
        "known_choice_count": kinds["KNOWN"] == gates["requiredKnownChoiceCount"],
        "novel_composite_choice_count": kinds["NOVEL_COMPOSITE"] == gates["requiredNovelCompositeChoiceCount"],
        "unsupported_choice_count": kinds["UNSUPPORTED_COMPOSITE"] == gates["requiredUnsupportedChoiceCount"],
        "insufficient_choice_count": kinds["ABSTAIN"] == gates["requiredInsufficientChoiceCount"],
        "cell_count": population["cell_count"] == gates["requiredCellCount"],
        "records_per_cell": all(count == gates["requiredRecordsPerCell"] for count in population["cell_counts"].values()),
        "source_record_count": population["source_record_count"] == gates["requiredSourceRecordCount"],
        "missing_control_count": population["missing_control_count"] == gates["requiredMissingControlCount"],
        "fixture_count": population["fixture_count"] == gates["requiredFixtureCount"],
        "records_per_truth": all(count == gates["requiredRecordsPerTruthChoice"] for count in population["truth_choice_counts"].values()),
        "records_per_candidate": all(count == gates["requiredRecordsPerPresentedCandidate"] for count in population["presented_candidate_counts"].values()),
        "known_pair_coverage": population["known_pair_coverage"] == gates["requiredKnownPairCoverage"],
        "novel_domain_coverage": population["novel_domain_coverage"] == gates["requiredNovelDomainCoverage"],
        "unsupported_domain_coverage": population["unsupported_domain_coverage"] == gates["requiredUnsupportedDomainCoverage"],
        "excluded_identifier_disjointness": gates["requireExcludedIdentifierDisjointness"] and population["excluded_identifier_overlap_count"] == 0,
        "unique_source_identifiers": gates["requireUniqueSourceIdentifiers"] and population["unique_source_identifier_count"] == population["source_record_count"],
        "text_free": not catalog["contains_language"] and not population["contains_language"],
        "zero_language_model_execution": gates["maximumEmittedLanguageRecordCount"] == gates["maximumManualLanguageInspectionCount"] == gates["maximumModelLoadCount"] == gates["maximumModelGenerationCount"] == gates["maximumActualExecutionCount"] == 0,
    }


__all__ = ["build_catalog", "evaluate_gates", "excluded_identifiers", "select_population"]
