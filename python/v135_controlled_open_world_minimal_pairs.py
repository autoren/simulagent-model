from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {
    "truth_choice_id",
    "phase",
    "possible_choice_ids",
    "decisive_cue",
    "left_choice_id",
    "right_choice_id",
    "family_id",
    "group_id",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _opaque_id(salt: str, *parts: str) -> str:
    material = "::".join((salt, *parts)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:20]


def _render(template: str, slots: dict[str, str]) -> str:
    return template.format(**slots)


def _contains(text: str, cue: str) -> bool:
    return cue.casefold() in text.casefold()


def build_catalog(config: dict[str, Any]) -> dict[str, Any]:
    choices = config["catalog"]["choices"]
    ids = [row["choice_id"] for row in choices]
    if len(ids) != len(set(ids)):
        raise ValueError("V135 catalog choice identifiers must be unique")
    kinds = Counter(row["kind"] for row in choices)
    known_ids = {row["choice_id"] for row in choices if row["kind"] == "KNOWN"}
    for row in choices:
        if not row["visible_definition"].strip():
            raise ValueError(f"V135 choice lacks visible definition: {row['choice_id']}")
        if row["kind"] == "KNOWN" and (not row.get("includes") or not row.get("excludes")):
            raise ValueError(f"V135 known choice lacks explicit boundary: {row['choice_id']}")
    public_choices = [
        {
            "choice_id": row["choice_id"],
            "kind": row["kind"],
            "name": row["name"],
            "visible_definition": row["visible_definition"],
            **({"includes": row["includes"], "excludes": row["excludes"]} if row["kind"] == "KNOWN" else {}),
        }
        for row in choices
    ]
    return {
        "choice_count": len(choices),
        "kind_counts": dict(sorted(kinds.items())),
        "known_choice_ids": sorted(known_ids),
        "complete_safe_hypothesis_universe": True,
        "choices": public_choices,
        "catalog_sha256": canonical_sha256(public_choices),
    }


def _fixture(
    config: dict[str, Any],
    family: dict[str, Any],
    split: str,
    variant_index: int,
    phase: str,
    conversation: list[dict[str, str]],
    truth_choice_id: str,
    possible_choice_ids: list[str],
    decisive_cue: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    salt = config["generation"]["baseSalt"]
    group_id = f"v135-group::{_opaque_id(salt, family['family_id'], str(variant_index))}"
    fixture_id = f"v135::{_opaque_id(salt, family['family_id'], str(variant_index), phase)}"
    public = {
        "fixture_id": fixture_id,
        "split": split,
        "presented_candidate_choice_id": family["presented_candidate_choice_id"],
        "conversation": conversation,
    }
    hidden = {
        **public,
        "group_id": group_id,
        "family_id": family["family_id"],
        "variant_index": variant_index,
        "phase": phase,
        "truth_choice_id": truth_choice_id,
        "possible_choice_ids": possible_choice_ids,
        "decisive_cue": decisive_cue,
        "observation_sufficient": phase != "ambiguous",
    }
    return public, hidden


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    catalog_ids = {row["choice_id"] for row in config["catalog"]["choices"]}
    known_ids = {row["choice_id"] for row in config["catalog"]["choices"] if row["kind"] == "KNOWN"}
    split_for_variant = {
        index: split
        for split, indices in config["generation"]["splits"].items()
        for index in indices
    }
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    cue_checks: list[bool] = []
    clarification_checks: list[bool] = []

    for family in config["families"]:
        left = family["left_choice_id"]
        right = family["right_choice_id"]
        candidate = family["presented_candidate_choice_id"]
        if left not in catalog_ids or right not in catalog_ids or candidate not in known_ids:
            raise ValueError(f"V135 family references invalid choices: {family['family_id']}")
        if candidate != left:
            raise ValueError(f"V135 presented candidate must be the known left choice: {family['family_id']}")
        if left not in known_ids or right in known_ids:
            raise ValueError(f"V135 families must pair known-left with non-known-right: {family['family_id']}")
        if len(family["surfaces"]) != 4 or len(family["slot_variants"]) != 8:
            raise ValueError(f"V135 family must have four surfaces and eight variants: {family['family_id']}")

        for variant_index, slots in enumerate(family["slot_variants"]):
            split = split_for_variant.get(variant_index)
            if split is None:
                raise ValueError(f"V135 variant lacks split: {family['family_id']}::{variant_index}")
            surface = family["surfaces"][variant_index % len(family["surfaces"])]
            left_text = _render(surface["left"], slots)
            right_text = _render(surface["right"], slots)
            ambiguous_text = _render(surface["ambiguous"], slots)
            left_cue = _render(surface["left_cue"], slots)
            right_cue = _render(surface["right_cue"], slots)
            cue_checks.extend(
                [
                    _contains(left_text, left_cue) and not _contains(left_text, right_cue),
                    _contains(right_text, right_cue) and not _contains(right_text, left_cue),
                    not _contains(ambiguous_text, left_cue) and not _contains(ambiguous_text, right_cue),
                    _contains(family["left_answer"], family["left_answer_cue"])
                    and not _contains(family["left_answer"], family["right_answer_cue"]),
                    _contains(family["right_answer"], family["right_answer_cue"])
                    and not _contains(family["right_answer"], family["left_answer_cue"]),
                ]
            )

            stages = [
                (
                    "clear_left",
                    [{"role": "user", "content": left_text}],
                    left,
                    [left],
                    left_cue,
                ),
                (
                    "clear_right",
                    [{"role": "user", "content": right_text}],
                    right,
                    [right],
                    right_cue,
                ),
                (
                    "ambiguous",
                    [{"role": "user", "content": ambiguous_text}],
                    "A00",
                    sorted([left, right]),
                    None,
                ),
                (
                    "clarified_left",
                    [
                        {"role": "user", "content": ambiguous_text},
                        {"role": "assistant", "content": family["clarification_question"]},
                        {"role": "user", "content": family["left_answer"]},
                    ],
                    left,
                    [left],
                    family["left_answer_cue"],
                ),
                (
                    "clarified_right",
                    [
                        {"role": "user", "content": ambiguous_text},
                        {"role": "assistant", "content": family["clarification_question"]},
                        {"role": "user", "content": family["right_answer"]},
                    ],
                    right,
                    [right],
                    family["right_answer_cue"],
                ),
            ]
            for phase, conversation, truth, possible, cue in stages:
                public, hidden = _fixture(
                    config,
                    family,
                    split,
                    variant_index,
                    phase,
                    conversation,
                    truth,
                    possible,
                    cue,
                )
                public_rows.append(public)
                hidden_rows.append(hidden)
            clarification_checks.extend(
                [
                    stages[3][2] == left and stages[3][1][-1]["content"] == family["left_answer"],
                    stages[4][2] == right and stages[4][1][-1]["content"] == family["right_answer"],
                ]
            )

    public_rows.sort(key=lambda row: row["fixture_id"])
    hidden_rows.sort(key=lambda row: row["fixture_id"])
    return {
        "fixture_count": len(hidden_rows),
        "group_count": len({row["group_id"] for row in hidden_rows}),
        "family_count": len({row["family_id"] for row in hidden_rows}),
        "split_counts": dict(sorted(Counter(row["split"] for row in hidden_rows).items())),
        "initial_split_counts": dict(
            sorted(Counter(row["split"] for row in hidden_rows if not row["phase"].startswith("clarified_")).items())
        ),
        "clarified_split_counts": dict(
            sorted(Counter(row["split"] for row in hidden_rows if row["phase"].startswith("clarified_")).items())
        ),
        "ambiguous_split_counts": dict(
            sorted(Counter(row["split"] for row in hidden_rows if row["phase"] == "ambiguous").items())
        ),
        "choice_coverage_by_split": {
            split: sorted({row["truth_choice_id"] for row in hidden_rows if row["split"] == split})
            for split in sorted({row["split"] for row in hidden_rows})
        },
        "phase_counts": dict(sorted(Counter(row["phase"] for row in hidden_rows).items())),
        "truth_counts": dict(sorted(Counter(row["truth_choice_id"] for row in hidden_rows).items())),
        "candidate_counts": dict(sorted(Counter(row["presented_candidate_choice_id"] for row in hidden_rows).items())),
        "cue_check_count": len(cue_checks),
        "cue_validation_rate": sum(cue_checks) / len(cue_checks),
        "clarification_check_count": len(clarification_checks),
        "clarification_resolution_rate": sum(clarification_checks) / len(clarification_checks),
        "public_fixtures_sha256": canonical_sha256(public_rows),
        "hidden_fixtures_sha256": canonical_sha256(hidden_rows),
        "public_fixtures": public_rows,
        "hidden_fixtures": hidden_rows,
    }


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return list(value) + [key for child in value.values() for key in _walk_keys(child)]
    if isinstance(value, list):
        return [key for child in value for key in _walk_keys(child)]
    return []


def evaluate_gates(catalog: dict[str, Any], population: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["gates"]
    kinds = catalog["kind_counts"]
    hidden = population["hidden_fixtures"]
    group_sizes = Counter(row["group_id"] for row in hidden)
    variants_by_family = {
        family: len({row["variant_index"] for row in hidden if row["family_id"] == family})
        for family in {row["family_id"] for row in hidden}
    }
    truth_derived = all(
        (row["phase"] == "ambiguous" and row["truth_choice_id"] == "A00" and not row["observation_sufficient"])
        or (row["phase"] != "ambiguous" and row["truth_choice_id"] != "A00" and row["observation_sufficient"])
        for row in hidden
    )
    public_keys = _walk_keys(population["public_fixtures"])
    return {
        "choice_count": catalog["choice_count"] == gates["requiredChoiceCount"],
        "known_choice_count": kinds.get("KNOWN", 0) == gates["requiredKnownChoiceCount"],
        "novel_choice_count": kinds.get("VALID_UNDECLARED", 0) == gates["requiredNovelChoiceCount"],
        "unsupported_choice_count": kinds.get("UNSUPPORTED", 0) == gates["requiredUnsupportedChoiceCount"],
        "insufficient_choice_count": kinds.get("INSUFFICIENT_EVIDENCE", 0) == gates["requiredInsufficientChoiceCount"],
        "family_count": population["family_count"] == gates["requiredFamilyCount"],
        "variant_count_per_family": set(variants_by_family.values()) == {gates["requiredVariantCountPerFamily"]},
        "group_count": population["group_count"] == gates["requiredGroupCount"],
        "stages_per_group": set(group_sizes.values()) == {gates["requiredStagesPerGroup"]},
        "fixture_count": population["fixture_count"] == gates["requiredFixtureCount"],
        "fixtures_per_split": set(population["split_counts"].values()) == {gates["requiredFixtureCountPerSplit"]},
        "initial_per_split": set(population["initial_split_counts"].values()) == {gates["requiredInitialFixtureCountPerSplit"]},
        "clarified_per_split": set(population["clarified_split_counts"].values()) == {gates["requiredClarifiedFixtureCountPerSplit"]},
        "ambiguous_per_split": set(population["ambiguous_split_counts"].values()) == {gates["requiredAmbiguousFixtureCountPerSplit"]},
        "cue_validation": population["cue_validation_rate"] == gates["requiredCueValidationRate"],
        "truth_derivation": truth_derived and gates["requiredTruthDerivationRate"] == 1.0,
        "clarification_resolution": population["clarification_resolution_rate"] == gates["requiredClarificationResolutionRate"],
        "public_gold_leakage": sum(key in FORBIDDEN_PUBLIC_KEYS for key in public_keys) == gates["requiredPublicGoldLeakageCount"],
        "choice_coverage_per_split": all(
            len(choices) == gates["requiredChoiceCoveragePerSplit"]
            for choices in population["choice_coverage_by_split"].values()
        ),
        "zero_external_or_model_access": all(
            gates[key] == 0
            for key in (
                "maximumV134LanguageReadCount",
                "maximumModelLoadCount",
                "maximumModelGenerationCount",
                "maximumAPICallCount",
                "maximumTrainingRunCount",
                "maximumActualExecutionCount",
            )
        ),
    }


__all__ = ["FORBIDDEN_PUBLIC_KEYS", "build_catalog", "build_population", "canonical_sha256", "evaluate_gates"]
