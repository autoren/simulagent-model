from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

from v102_presto_context_source import TARGET_ROOT_PATTERN, contains_phrase, normalized_tokens
from v102r1_presto_context_source import parse_presto_archive, tolerant_context_surfaces


LITERAL_PATTERNS = {
    "guillemet": re.compile(r"«\s*(.*?)\s*»"),
    "single_guillemet": re.compile(r"‹\s*(.*?)\s*›"),
    "ascii_double_quote": re.compile(r'"\s*([^"\n]*?)\s*"'),
    "curly_double_quote": re.compile(r"“\s*(.*?)\s*”"),
    "ascii_single_quote": re.compile(r"'\s*([^'\n]*?)\s*'"),
    "square_bracket": re.compile(r"\[\s*([^\[\]\n]*?)\s*\]"),
}

STRUCTURAL_CHARACTERS = {
    "left_parenthesis": "(", "right_parenthesis": ")", "colon": ":", "equals": "=",
    "left_square_bracket": "[", "left_guillemet": "«", "left_single_guillemet": "‹",
    "ascii_double_quote": '"', "curly_double_quote": "“",
}


def _quality_literals(target: str, family: str, rule: dict[str, Any]) -> list[str]:
    prohibited = frozenset(rule["prohibitedNormalizedArguments"])
    values = []
    for match in LITERAL_PATTERNS[family].finditer(target):
        value = match.group(1).strip()
        tokens = normalized_tokens(value)
        normalized = " ".join(tokens)
        if (
            len(normalized.replace(" ", "")) >= rule["minimumNormalizedArgumentCharacterCount"]
            and len(tokens) <= rule["maximumNormalizedArgumentTokenCount"]
            and normalized not in prohibited
        ):
            values.append(value)
    return values


def build_target_syntax_census(
    source_records: Any, scientific_config: dict[str, Any], diagnostic_config: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("PRESTO source records are missing")
    families = tuple(diagnostic_config["literalFamilies"])
    if set(families) != set(LITERAL_PATTERNS):
        raise ValueError("literal-family code and config disagree")
    eligible_families = frozenset(diagnostic_config["candidateEligibleLiteralFamilies"])
    rule = scientific_config["dependencyRule"]
    source_kinds = frozenset(rule["eligibleContextSources"])
    previous_kinds = {"previous_turn_user_query", "previous_turn_response_text"}
    seeded_kinds = source_kinds - previous_kinds
    family_counts: dict[str, Counter[str]] = {family: Counter() for family in families}
    structural_counts: Counter[str] = Counter()
    human_split_counts: Counter[str] = Counter()
    union_candidates: dict[str, dict[str, Any]] = {}
    union_roots: set[str] = set()
    ignored_leaves = 0
    for member, row in source_records:
        metadata = row.get("metadata")
        target = row.get("targets")
        inputs = row.get("inputs")
        if not isinstance(metadata, dict) or not isinstance(target, str) or not isinstance(inputs, str):
            raise ValueError("PRESTO record structure is invalid")
        split = metadata.get("split")
        expected_member = f"presto_{split}.jsonl"
        if member != expected_member:
            raise ValueError("PRESTO member and split disagree")
        if (
            metadata.get("locale") != scientific_config["locale"]
            or metadata.get("context") != scientific_config["requiredContextProvenance"]
        ):
            continue
        role = scientific_config["canonicalSplitMap"][split]
        human_split_counts[role] += 1
        surfaces, ignored = tolerant_context_surfaces(metadata)
        ignored_leaves += ignored
        for feature, character in STRUCTURAL_CHARACTERS.items():
            if character in target:
                structural_counts[feature] += 1
        identifier = str(metadata.get("example_id"))
        union_kinds: set[str] = set()
        union_dependency = False
        for family in families:
            raw_matches = list(LITERAL_PATTERNS[family].finditer(target))
            if raw_matches:
                family_counts[family]["record_contains_literal_family"] += 1
            literals = _quality_literals(target, family, rule)
            if literals:
                family_counts[family]["quality_filtered_literal"] += 1
            absent = [value for value in literals if not contains_phrase(inputs, value)]
            if absent:
                family_counts[family]["literal_absent_from_current_input"] += 1
            context_present = []
            dependency_kinds: set[str] = set()
            for value in literals:
                matched_kinds = {
                    kind for kind, surface in surfaces
                    if kind in source_kinds and contains_phrase(surface, value)
                }
                if matched_kinds:
                    context_present.append(value)
                    dependency_kinds.update(matched_kinds)
            if context_present:
                family_counts[family]["literal_present_in_context"] += 1
            dependency_values = []
            for value in absent:
                matched_kinds = {
                    kind for kind, surface in surfaces
                    if kind in source_kinds and contains_phrase(surface, value)
                }
                if matched_kinds:
                    dependency_values.append(value)
                    dependency_kinds.update(matched_kinds)
            if dependency_values:
                family_counts[family]["literal_absent_from_input_and_present_in_context"] += 1
                if family in eligible_families:
                    union_dependency = True
                    union_kinds.update(dependency_kinds)
        if union_dependency:
            root_match = TARGET_ROOT_PATTERN.match(target)
            if root_match:
                union_roots.add(root_match.group(1))
            union_candidates[identifier] = {"role": role, "source_kinds": union_kinds}
    union_role_counts = Counter(value["role"] for value in union_candidates.values())
    union_kind_counts = Counter(
        kind for value in union_candidates.values() for kind in value["source_kinds"]
    )
    return {
        "source_record_count": len(source_records),
        "en_us_human_context_role_counts": dict(sorted(human_split_counts.items())),
        "literal_family_stage_record_counts": {
            family: {stage: counts.get(stage, 0) for stage in diagnostic_config["diagnosticStages"]}
            for family, counts in sorted(family_counts.items())
        },
        "structural_character_record_counts": {
            feature: structural_counts.get(feature, 0)
            for feature in diagnostic_config["structuralCharacterFeatures"]
        },
        "candidate_eligible_family_union_count": len(union_candidates),
        "candidate_eligible_family_union_role_counts": dict(sorted(union_role_counts.items())),
        "candidate_eligible_family_union_dependency_source_kind_counts": dict(
            sorted(union_kind_counts.items())
        ),
        "candidate_eligible_family_union_dependency_source_kind_count": len(union_kind_counts),
        "candidate_eligible_family_union_previous_turn_dependent_count": sum(
            bool(value["source_kinds"] & previous_kinds) for value in union_candidates.values()
        ),
        "candidate_eligible_family_union_seeded_state_dependent_count": sum(
            bool(value["source_kinds"] & seeded_kinds) for value in union_candidates.values()
        ),
        "candidate_eligible_family_union_semantic_root_function_count": len(union_roots),
        "ignored_non_string_optional_context_leaf_count": ignored_leaves,
        "emitted_candidate_identifier_count": 0,
        "contains_input_target_literal_context_tokens_identifiers_or_root_names": False,
    }


def evaluate_target_syntax_gates(
    census: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    gates = config["diagnosticGates"]
    return {
        "union_development_candidate_count": (
            census["candidate_eligible_family_union_role_counts"].get("development", 0)
            >= gates["minimumUnionDevelopmentCandidateCount"]
        ),
        "union_protected_test_candidate_count": (
            census["candidate_eligible_family_union_role_counts"].get("protected_test", 0)
            >= gates["minimumUnionProtectedTestCandidateCount"]
        ),
        "union_total_candidate_count": (
            census["candidate_eligible_family_union_count"]
            >= gates["minimumUnionTotalCandidateCount"]
        ),
        "union_previous_turn_dependency_count": (
            census["candidate_eligible_family_union_previous_turn_dependent_count"]
            >= gates["minimumUnionPreviousTurnDependentCandidateCount"]
        ),
        "union_seeded_state_dependency_count": (
            census["candidate_eligible_family_union_seeded_state_dependent_count"]
            >= gates["minimumUnionSeededStateDependentCandidateCount"]
        ),
        "union_dependency_source_kind_count": (
            census["candidate_eligible_family_union_dependency_source_kind_count"]
            >= gates["minimumUnionDependencySourceKindCount"]
        ),
        "union_semantic_root_function_count": (
            census["candidate_eligible_family_union_semantic_root_function_count"]
            >= gates["minimumUnionSemanticRootFunctionCount"]
        ),
        "zero_emitted_candidate_identifiers": (
            census["emitted_candidate_identifier_count"]
            <= gates["maximumEmittedCandidateIdentifierCount"]
        ),
        "text_and_identifier_free_census": not (
            census["contains_input_target_literal_context_tokens_identifiers_or_root_names"]
        ),
    }


__all__ = ["build_target_syntax_census", "evaluate_target_syntax_gates"]
