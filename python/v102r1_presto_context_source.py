from __future__ import annotations

from collections import Counter
from typing import Any

from v102_presto_context_source import (
    TARGET_ROOT_PATTERN,
    contains_phrase,
    evaluate_presto_source_gates,
    normalized_tokens,
    parse_presto_archive,
    target_arguments,
)
from v93_open_set_source import canonical_sha256


def _optional_list(value: Any, field: str) -> tuple[list[Any], int]:
    if value is None:
        return [], 1
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list or null")
    return value, 0


def tolerant_context_surfaces(metadata: dict[str, Any]) -> tuple[list[tuple[str, str]], int]:
    surfaces: list[tuple[str, str]] = []
    ignored = 0
    previous_turns, count = _optional_list(metadata["previous_turns"], "previous_turns")
    ignored += count
    for turn in previous_turns:
        if not isinstance(turn, dict):
            raise ValueError("previous turn must be an object")
        for field, kind in (
            ("user_query", "previous_turn_user_query"),
            ("response_text", "previous_turn_response_text"),
        ):
            value = turn.get(field)
            if isinstance(value, str):
                surfaces.append((kind, value))
            else:
                ignored += 1
    seeded_lists, count = _optional_list(metadata["seeded_lists"], "seeded_lists")
    ignored += count
    for item in seeded_lists:
        if not isinstance(item, dict):
            raise ValueError("seeded list must be an object")
        name = item.get("name")
        if isinstance(name, str):
            surfaces.append(("seeded_list_name", name))
        else:
            ignored += 1
        values, count = _optional_list(item.get("items"), "seeded list items")
        ignored += count
        for value in values:
            if isinstance(value, str):
                surfaces.append(("seeded_list_item", value))
            else:
                ignored += 1
    seeded_notes, count = _optional_list(metadata["seeded_notes"], "seeded_notes")
    ignored += count
    for note in seeded_notes:
        if not isinstance(note, dict):
            raise ValueError("seeded note must be an object")
        for field, kind in (("name", "seeded_note_name"), ("text", "seeded_note_text")):
            value = note.get(field)
            if isinstance(value, str):
                surfaces.append((kind, value))
            else:
                ignored += 1
    seeded_contacts, count = _optional_list(metadata["seeded_contacts"], "seeded_contacts")
    ignored += count
    for value in seeded_contacts:
        if isinstance(value, str):
            surfaces.append(("seeded_contact", value))
        else:
            ignored += 1
    return surfaces, ignored


def build_repaired_presto_context_inventory(
    source_records: Any, scientific_config: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("PRESTO records must be a non-empty list")
    required_record = frozenset(scientific_config["requiredRecordFields"])
    required_metadata = frozenset(scientific_config["requiredMetadataFields"])
    allowed_splits = frozenset(scientific_config["allowedSourceSplits"])
    split_map = scientific_config["canonicalSplitMap"]
    rule = scientific_config["dependencyRule"]
    eligible_sources = frozenset(rule["eligibleContextSources"])
    seen: set[str] = set()
    candidates = []
    member_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    en_us_human_counts: Counter[str] = Counter()
    semantic_roots: set[str] = set()
    ignored_optional_leaves = 0
    for entry in source_records:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("PRESTO source entry must contain member and record")
        member, row = entry
        if not isinstance(member, str) or not isinstance(row, dict) or not required_record <= row.keys():
            raise ValueError("PRESTO record lacks required fields")
        inputs = row["inputs"]
        target = row["targets"]
        metadata = row["metadata"]
        if not isinstance(inputs, str) or not isinstance(target, str) or not isinstance(metadata, dict):
            raise ValueError("PRESTO input, target, or metadata is invalid")
        if not required_metadata <= metadata.keys():
            raise ValueError("PRESTO metadata lacks required fields")
        identifier = str(metadata["example_id"])
        source_split = metadata["split"]
        if not identifier or identifier in seen or source_split not in allowed_splits:
            raise ValueError("PRESTO identifier or split is invalid")
        expected_member = f"presto_{source_split}.jsonl"
        if member != expected_member:
            raise ValueError("PRESTO member and metadata split disagree")
        seen.add(identifier)
        member_counts[member] += 1
        split_counts[source_split] += 1
        if metadata["locale"] != scientific_config["locale"]:
            continue
        if metadata["context"] != scientific_config["requiredContextProvenance"]:
            continue
        en_us_human_counts[source_split] += 1
        surfaces, ignored = tolerant_context_surfaces(metadata)
        ignored_optional_leaves += ignored
        matches: list[tuple[str, str]] = []
        for argument in target_arguments(target, rule):
            if contains_phrase(inputs, argument):
                continue
            for source_kind, surface in surfaces:
                if source_kind in eligible_sources and contains_phrase(surface, argument):
                    matches.append((argument, source_kind))
        if not matches:
            continue
        source_kinds = sorted({kind for _, kind in matches})
        root_match = TARGET_ROOT_PATTERN.match(target)
        if root_match:
            semantic_roots.add(root_match.group(1))
        role = split_map[source_split]
        candidates.append({
            "candidate_id": f"presto::{identifier}",
            "source_id": identifier,
            "source_member": member,
            "source_split": source_split,
            "role": role,
            "dependency_source_kinds": source_kinds,
            "dependency_argument_count": len({
                " ".join(normalized_tokens(argument)) for argument, _ in matches
            }),
            "previous_turn_count": len(metadata["previous_turns"] or []),
            "seeded_list_count": len(metadata["seeded_lists"] or []),
            "seeded_note_count": len(metadata["seeded_notes"] or []),
            "seeded_contact_count": len(metadata["seeded_contacts"] or []),
            "full_context_pair_id": f"presto::{identifier}::full",
            "ablated_context_pair_id": f"presto::{identifier}::ablated",
        })
    candidates.sort(key=lambda row: row["candidate_id"])
    forbidden = {
        "inputs", "targets", "input", "target", "argument", "context_text", "tokens",
        "seeded_values", "utterance", "text", "prompt",
    }
    keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    if keys & forbidden:
        raise AssertionError("PRESTO language leaked into repaired inventory")
    role_counts = Counter(row["role"] for row in candidates)
    dependency_counts = Counter(
        kind for row in candidates for kind in row["dependency_source_kinds"]
    )
    previous_turn_kinds = {"previous_turn_user_query", "previous_turn_response_text"}
    seeded_kinds = eligible_sources - previous_turn_kinds
    development_ids = {
        row["source_id"] for row in candidates if row["role"] == "development"
    }
    test_ids = {
        row["source_id"] for row in candidates if row["role"] == "protected_test"
    }
    return {
        "source_record_count": len(source_records),
        "source_member_record_counts": dict(sorted(member_counts.items())),
        "source_split_record_counts": dict(sorted(split_counts.items())),
        "en_us_human_context_record_counts": dict(sorted(en_us_human_counts.items())),
        "eligible_candidate_count": len(candidates),
        "role_candidate_counts": dict(sorted(role_counts.items())),
        "dependency_source_kind_counts": dict(sorted(dependency_counts.items())),
        "dependency_source_kind_count": len(dependency_counts),
        "previous_turn_dependent_candidate_count": sum(
            bool(set(row["dependency_source_kinds"]) & previous_turn_kinds) for row in candidates
        ),
        "seeded_state_dependent_candidate_count": sum(
            bool(set(row["dependency_source_kinds"]) & seeded_kinds) for row in candidates
        ),
        "semantic_root_function_count": len(semantic_roots),
        "synthetic_context_candidate_count": 0,
        "ignored_non_string_optional_context_leaf_count": ignored_optional_leaves,
        "development_test_identifiers_are_disjoint": not (development_ids & test_ids),
        "candidate_index_sha256": canonical_sha256(candidates),
        "candidate_index": candidates,
        "pairs_share_source_id_input_and_target_by_construction": True,
        "contains_input_target_argument_context_tokens_seeded_values_or_prompts": False,
    }


def evaluate_repaired_presto_source_gates(
    inventory: dict[str, Any], scientific_config: dict[str, Any]
) -> dict[str, bool]:
    return evaluate_presto_source_gates(inventory, scientific_config)


__all__ = [
    "build_repaired_presto_context_inventory", "evaluate_repaired_presto_source_gates",
    "parse_presto_archive", "tolerant_context_surfaces",
]
