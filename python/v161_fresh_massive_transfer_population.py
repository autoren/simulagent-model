from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order


def _select_rows(
    pool: list[dict[str, Any]],
    *,
    count: int,
    scenario_minimum: int,
    salt: str,
    role: str,
    class_label: str,
) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_scenario[row["scenario"]].append(row)
    if scenario_minimum * len(by_scenario) > count:
        raise ValueError("scenario minima exceed the class quota")
    selected: dict[str, dict[str, Any]] = {}
    for scenario, rows in sorted(by_scenario.items()):
        if len(rows) < scenario_minimum:
            raise ValueError(f"insufficient {role}/{class_label}/{scenario} candidates")
        ordered = sorted(
            rows,
            key=lambda row: hash_order(
                salt,
                role,
                class_label,
                "scenario-minimum",
                scenario,
                row["candidate_id"],
            ),
        )
        selected.update((row["candidate_id"], row) for row in ordered[:scenario_minimum])
    remainder = sorted(
        (row for row in pool if row["candidate_id"] not in selected),
        key=lambda row: hash_order(salt, role, class_label, "fill", row["candidate_id"]),
    )
    needed = count - len(selected)
    if len(remainder) < needed:
        raise ValueError(f"insufficient {role}/{class_label} candidates")
    selected.update((row["candidate_id"], row) for row in remainder[:needed])
    return sorted(selected.values(), key=lambda row: row["candidate_id"])


def select_transfer_population(
    inventory: Any, excluded_population: Any, config: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(inventory, dict) or not isinstance(inventory.get("candidate_index"), list):
        raise ValueError("V100 candidate inventory is missing")
    if inventory.get("candidate_index_sha256") != config["sourceCandidateIndexSha256"]:
        raise ValueError("V100 candidate index identity mismatch")
    if not isinstance(excluded_population, dict) or not isinstance(
        excluded_population.get("selected_population"), list
    ):
        raise ValueError("V101 excluded population is missing")
    if excluded_population.get("selected_population_sha256") != config["excludedPopulationPayloadSha256"]:
        raise ValueError("V101 excluded population identity mismatch")
    rows = inventory["candidate_index"]
    excluded_rows = excluded_population["selected_population"]
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("candidate identifiers are not unique")
    excluded_ids = {row["candidate_id"] for row in excluded_rows}
    if len(excluded_ids) != config["populationGates"]["requiredExcludedCandidateCount"]:
        raise ValueError("excluded population candidate count mismatch")

    required_classes = tuple(config["requiredClasses"])
    selection = config["selection"]
    count = selection["selectedCandidateCountPerClassPerRole"]
    remaining_counts: dict[str, dict[str, int]] = defaultdict(dict)
    selected: list[dict[str, Any]] = []
    for role, role_spec in selection["roles"].items():
        partition = role_spec["sourcePartition"]
        for class_label in required_classes:
            pool = [
                row
                for row in rows
                if row["candidate_id"] not in excluded_ids
                and row["partition"] == partition
                and row["class_label"] == class_label
            ]
            remaining_counts[role][class_label] = len(pool)
            chosen = _select_rows(
                pool,
                count=count,
                scenario_minimum=selection["scenarioMinimumPerClass"][class_label],
                salt=selection["baseSalt"],
                role=role,
                class_label=class_label,
            )
            for row in chosen:
                selected.append(
                    {
                        "population_id": f"v161::{role}::{row['candidate_id']}",
                        "candidate_id": row["candidate_id"],
                        "source_id": row["source_id"],
                        "role": role,
                        "source_partition": partition,
                        "class_label": class_label,
                        "scenario": row["scenario"],
                        "intent": row["intent"],
                        "current_utterance_intent_overlap_count": row[
                            "current_utterance_intent_overlap_count"
                        ],
                        "slot_type_count": row["slot_type_count"],
                    }
                )
    selected.sort(key=lambda row: row["population_id"])
    forbidden = {
        "utt",
        "utterance",
        "annot_utt",
        "annotated_utterance",
        "tokens",
        "slot_values",
        "values",
        "text",
        "prompt",
    }
    keys = set().union(*(row.keys() for row in selected)) if selected else set()
    if keys & forbidden:
        raise AssertionError("language leaked into V161 population")

    role_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_class_scenarios: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    role_class_intents: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in selected:
        role_class_counts[row["role"]][row["class_label"]] += 1
        role_class_scenarios[row["role"]][row["class_label"]].add(row["scenario"])
        role_class_intents[row["role"]][row["class_label"]].add(row["intent"])
    role_ids = {
        role: {row["candidate_id"] for row in selected if row["role"] == role}
        for role in selection["roles"]
    }
    selected_ids = {row["candidate_id"] for row in selected}
    return {
        "selected_candidate_count": len(selected),
        "excluded_candidate_count": len(excluded_ids),
        "excluded_population_overlap_count": len(selected_ids & excluded_ids),
        "remaining_pool_counts": {
            role: dict(sorted(counts.items())) for role, counts in sorted(remaining_counts.items())
        },
        "role_counts": dict(sorted(Counter(row["role"] for row in selected).items())),
        "role_class_counts": {
            role: dict(sorted(counts.items())) for role, counts in sorted(role_class_counts.items())
        },
        "role_class_scenario_counts": {
            role: {label: len(values) for label, values in sorted(classes.items())}
            for role, classes in sorted(role_class_scenarios.items())
        },
        "role_class_intent_counts": {
            role: {label: len(values) for label, values in sorted(classes.items())}
            for role, classes in sorted(role_class_intents.items())
        },
        "source_partition_counts": dict(
            sorted(Counter(row["source_partition"] for row in selected).items())
        ),
        "role_identifiers_are_disjoint": not (
            role_ids.get("development_transfer", set())
            & role_ids.get("protected_transfer", set())
        ),
        "selected_population_sha256": canonical_sha256(selected),
        "selected_population": selected,
        "contains_language_tokens_slot_values_or_prompts": False,
    }


def evaluate_population_gates(
    population: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["populationGates"]
    required_classes = tuple(config["requiredClasses"])
    roles = tuple(config["selection"]["roles"])
    checks: dict[str, bool] = {
        "total_candidate_count": population["selected_candidate_count"]
        == gates["requiredTotalCandidateCount"],
        "excluded_candidate_count": population["excluded_candidate_count"]
        == gates["requiredExcludedCandidateCount"],
        "zero_overlap_with_excluded_population": population["excluded_population_overlap_count"]
        == gates["requiredOverlapWithExcludedPopulation"],
        "role_identifier_disjointness": bool(population["role_identifiers_are_disjoint"]),
        "zero_train_partition_candidates": population["source_partition_counts"].get("train", 0)
        <= gates["maximumTrainPartitionCandidateCount"],
        "text_free_population": not population["contains_language_tokens_slot_values_or_prompts"],
    }
    expected_coverage = {
        "known_familiar": gates["requiredKnownScenarioCoverage"],
        "known_unfamiliar": gates["requiredKnownScenarioCoverage"],
        "novel_valid": gates["requiredNovelScenarioCoverage"],
        "unsupported": gates["requiredUnsupportedScenarioCoverage"],
    }
    for role in roles:
        checks[f"{role}_candidate_count"] = (
            population["role_counts"].get(role, 0) == gates["requiredCandidateCountPerRole"]
        )
        for class_label in required_classes:
            prefix = f"{role}_{class_label}"
            checks[f"{prefix}_remaining_pool"] = (
                population["remaining_pool_counts"].get(role, {}).get(class_label, 0)
                >= gates["minimumRemainingCandidateCountPerClassPerRole"]
            )
            checks[f"{prefix}_candidate_count"] = (
                population["role_class_counts"].get(role, {}).get(class_label, 0)
                == gates["requiredCandidateCountPerClassPerRole"]
            )
            checks[f"{prefix}_scenario_coverage"] = (
                population["role_class_scenario_counts"].get(role, {}).get(class_label, 0)
                == expected_coverage[class_label]
            )
            checks[f"{prefix}_intent_coverage"] = (
                population["role_class_intent_counts"].get(role, {}).get(class_label, 0)
                >= gates["minimumIntentCoveragePerClass"][class_label]
            )
    return checks


__all__ = ["evaluate_population_gates", "select_transfer_population"]
