from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v93_open_set_source import canonical_sha256, normalized_tokens
from v104_massive_language_extraction import parse_annotated_slots


def build_selected_language_artifacts(
    population_artifact: Any,
    source_inventory: Any,
    source_records: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(population_artifact, dict) or not isinstance(
        population_artifact.get("selected_population"), list
    ):
        raise ValueError("selected population is invalid")
    if not isinstance(source_inventory, dict) or not isinstance(
        source_inventory.get("candidate_index"), list
    ):
        raise ValueError("source inventory is invalid")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("source records are invalid")

    selected = population_artifact["selected_population"]
    candidate_by_id = {
        row["candidate_id"]: row for row in source_inventory["candidate_index"]
    }
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_records:
        if not isinstance(row, dict):
            raise ValueError("source row must be an object")
        identifier = str(row.get("id"))
        if not identifier or identifier in source_by_id:
            raise ValueError("source identifiers must be unique")
        source_by_id[identifier] = row

    declared = frozenset(source_inventory["declared_intents"])
    hidden = frozenset(source_inventory["hidden_intents"])
    unsupported = frozenset(source_inventory["unsupported_scenarios"])
    configured_roles = frozenset(config["roles"])
    role_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_structure = True
    exact_familiarity = True
    exact_slot_counts = True

    for selected_row in selected:
        role = selected_row["role"]
        if role not in configured_roles:
            raise ValueError("selected role is not configured")
        candidate = candidate_by_id.get(selected_row["candidate_id"])
        source = source_by_id.get(selected_row["source_id"])
        if candidate is None or source is None:
            raise ValueError("selected source or candidate identifier is missing")
        canonical_partition = config["canonicalSourcePartitionMap"].get(source["partition"])
        exact_structure = exact_structure and bool(
            str(source["id"]) == selected_row["source_id"]
            and candidate["source_id"] == selected_row["source_id"]
            and candidate["partition"] == selected_row["source_partition"]
            and canonical_partition == selected_row["source_partition"]
            and candidate["scenario"] == selected_row["scenario"] == source["scenario"]
            and candidate["intent"] == selected_row["intent"] == source["intent"]
            and candidate["class_label"] == selected_row["class_label"]
        )

        computed_overlap = len(
            normalized_tokens(source["utt"]) & normalized_tokens(source["intent"])
        )
        if selected_row["class_label"] == "known_familiar":
            exact_familiarity = exact_familiarity and bool(
                computed_overlap == selected_row["current_utterance_intent_overlap_count"]
                and computed_overlap >= 1
            )
        elif selected_row["class_label"] == "known_unfamiliar":
            exact_familiarity = exact_familiarity and bool(
                computed_overlap
                == selected_row["current_utterance_intent_overlap_count"]
                == 0
            )

        slots = parse_annotated_slots(source["annot_utt"])
        exact_slot_counts = exact_slot_counts and bool(
            len({slot["slot_type"] for slot in slots}) == selected_row["slot_type_count"]
        )
        pair = f"{source['scenario']}::{source['intent']}"
        if selected_row["class_label"] in {"known_familiar", "known_unfamiliar"}:
            visibility = "declared_known"
            exact_structure = exact_structure and pair in declared
        elif selected_row["class_label"] == "novel_valid":
            visibility = "hidden_valid"
            exact_structure = exact_structure and pair in hidden
        elif selected_row["class_label"] == "unsupported":
            visibility = "withheld_scenario"
            exact_structure = exact_structure and source["scenario"] in unsupported
        else:
            raise ValueError("unexpected selected class")

        role_records[role].append(
            {
                "record_id": selected_row["population_id"],
                "candidate_id": selected_row["candidate_id"],
                "source_id": selected_row["source_id"],
                "role": role,
                "source_partition": selected_row["source_partition"],
                "class_label": selected_row["class_label"],
                "schema_visibility": visibility,
                "scenario": source["scenario"],
                "intent": source["intent"],
                "utterance": source["utt"],
                "annotated_utterance": source["annot_utt"],
                "slots": slots,
                "current_utterance_intent_overlap_count": selected_row[
                    "current_utterance_intent_overlap_count"
                ],
            }
        )

    for role in configured_roles:
        role_records[role].sort(key=lambda row: row["record_id"])
    all_records = [
        row for role in sorted(configured_roles) for row in role_records[role]
    ]
    selected_ids = {row["candidate_id"] for row in selected}
    emitted_ids = {row["candidate_id"] for row in all_records}
    role_counts = Counter(row["role"] for row in all_records)
    role_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_records:
        role_class_counts[row["role"]][row["class_label"]] += 1
    development_ids = {
        row["candidate_id"] for row in role_records["development_transfer"]
    }
    protected_ids = {
        row["candidate_id"] for row in role_records["protected_transfer"]
    }

    return {
        "total_record_count": len(all_records),
        "role_record_counts": dict(sorted(role_counts.items())),
        "role_class_record_counts": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_class_counts.items())
        },
        "exact_selected_identifier_set": bool(
            len(all_records) == len(selected)
            and len(emitted_ids) == len(all_records)
            and emitted_ids == selected_ids
        ),
        "exact_structural_ground_truth_match": exact_structure,
        "exact_familiarity_reconstruction": exact_familiarity,
        "exact_slot_type_count_reconstruction": exact_slot_counts,
        "development_protected_role_disjoint": not (development_ids & protected_ids),
        "unselected_language_record_count": 0,
        "role_payload_sha256": {
            role: canonical_sha256(role_records[role]) for role in sorted(configured_roles)
        },
        "role_records": dict(role_records),
    }


def evaluate_extraction_gates(
    artifacts: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["extractionGates"]
    checks = {
        "total_record_count": (
            artifacts["total_record_count"] == gates["requiredTotalRecordCount"]
        ),
        "exact_selected_identifier_set": artifacts["exact_selected_identifier_set"],
        "exact_structural_ground_truth_match": artifacts[
            "exact_structural_ground_truth_match"
        ],
        "exact_familiarity_reconstruction": artifacts[
            "exact_familiarity_reconstruction"
        ],
        "exact_slot_type_count_reconstruction": artifacts[
            "exact_slot_type_count_reconstruction"
        ],
        "development_protected_role_disjointness": artifacts[
            "development_protected_role_disjoint"
        ],
        "zero_unselected_language_records": (
            artifacts["unselected_language_record_count"]
            <= gates["maximumUnselectedLanguageRecordCount"]
        ),
    }
    for role in config["roles"]:
        checks[f"{role}_record_count"] = (
            artifacts["role_record_counts"].get(role, 0)
            == gates["requiredRecordCountPerRole"]
        )
        for class_label in config["requiredClasses"]:
            checks[f"{role}_{class_label}_record_count"] = (
                artifacts["role_class_record_counts"].get(role, {}).get(class_label, 0)
                == gates["requiredRecordCountPerClassPerRole"]
            )
    return checks


__all__ = ["build_selected_language_artifacts", "evaluate_extraction_gates"]
