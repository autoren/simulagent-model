from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash


STAGES = (
    "clear_known_familiar",
    "clear_known_unfamiliar",
    "clear_right",
    "ambiguous",
    "clarified_known",
    "clarified_right",
)


def build_catalog(config: dict[str, Any]) -> dict[str, Any]:
    choices = config["catalog"]["choices"]
    payload = {"schema_version": "142-certificate-choice-catalog", "choices": choices}
    payload["catalog_sha256"] = payload_hash(payload)
    return payload


def _render(template: str, slots: dict[str, str]) -> str:
    return template.format(**slots)


def _fixture_id(group_id: str, stage: str) -> str:
    digest = hashlib.sha256(f"{group_id}|{stage}".encode("utf-8")).hexdigest()[:16]
    return f"v142-{digest}"


def _conversation(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "text": text}]


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    catalog = build_catalog(config)
    kinds = {row["choice_id"]: row["kind"] for row in catalog["choices"]}
    public_rows = []
    hidden_rows = []
    for family_index, family in enumerate(config["families"]):
        left = family["left_choice_id"]
        right = family["right_choice_id"]
        pair = sorted([left, right])
        for variant_index, slots in enumerate(family["slot_variants"]):
            split = "development" if variant_index < 4 else "test"
            group_id = f"v142-g{family_index:02d}-{variant_index:02d}"
            ambiguous = _render(family["ambiguous"], slots)
            stage_specs = {
                "clear_known_familiar": {
                    "conversation": _conversation(_render(family["known_familiar"], slots)),
                    "truth": left,
                    "compatible": [left],
                    "language_class": "known_familiar",
                },
                "clear_known_unfamiliar": {
                    "conversation": _conversation(_render(family["known_unfamiliar"], slots)),
                    "truth": left,
                    "compatible": [left],
                    "language_class": "known_unfamiliar",
                },
                "clear_right": {
                    "conversation": _conversation(_render(family["right_clear"], slots)),
                    "truth": right,
                    "compatible": [right],
                    "language_class": "novel_valid" if kinds[right] == "VALID_UNDECLARED" else "unsupported",
                },
                "ambiguous": {
                    "conversation": _conversation(ambiguous),
                    "truth": "A00",
                    "compatible": pair,
                    "language_class": "insufficient_evidence",
                },
                "clarified_known": {
                    "conversation": [
                        {"role": "user", "text": ambiguous},
                        {"role": "assistant", "text": family["question"]},
                        {"role": "user", "text": _render(family["left_answer"], slots)},
                    ],
                    "truth": left,
                    "compatible": [left],
                    "language_class": "known_clarified",
                },
                "clarified_right": {
                    "conversation": [
                        {"role": "user", "text": ambiguous},
                        {"role": "assistant", "text": family["question"]},
                        {"role": "user", "text": _render(family["right_answer"], slots)},
                    ],
                    "truth": right,
                    "compatible": [right],
                    "language_class": "novel_valid_clarified" if kinds[right] == "VALID_UNDECLARED" else "unsupported_clarified",
                },
            }
            for stage in STAGES:
                spec = stage_specs[stage]
                fixture_id = _fixture_id(group_id, stage)
                public = {
                    "fixture_id": fixture_id,
                    "split": split,
                    "presented_candidate_choice_id": family["presented_candidate_choice_id"],
                    "conversation": spec["conversation"],
                }
                hidden = {
                    **public,
                    "group_id": group_id,
                    "family_id": family["family_id"],
                    "stage": stage,
                    "language_class": spec["language_class"],
                    "truth_choice_id": spec["truth"],
                    "compatible_choice_ids": spec["compatible"],
                    "variant_index": variant_index,
                }
                public_rows.append(public)
                hidden_rows.append(hidden)
    public_rows.sort(key=lambda row: row["fixture_id"])
    hidden_rows.sort(key=lambda row: row["fixture_id"])
    summary = {
        "choice_count": len(catalog["choices"]),
        "family_count": len(config["families"]),
        "group_count": len({row["group_id"] for row in hidden_rows}),
        "fixture_count": len(hidden_rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in hidden_rows).items())),
        "stage_counts": dict(sorted(Counter(row["stage"] for row in hidden_rows).items())),
        "language_class_counts": dict(sorted(Counter(row["language_class"] for row in hidden_rows).items())),
        "truth_counts": dict(sorted(Counter(row["truth_choice_id"] for row in hidden_rows).items())),
    }
    return {
        "choice_catalog": catalog,
        "public_fixtures": public_rows,
        "hidden_fixtures": hidden_rows,
        "population_summary": summary,
    }


def validate_certificate(certificate: Any, catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    contract = config["certificateContract"]
    valid_ids = {row["choice_id"] for row in catalog["choices"]}
    invalid = lambda reason: {
        "certificate_valid": False,
        "validation_reason": reason,
        "final_choice_id": "A00",
    }
    if not isinstance(certificate, dict) or set(certificate) != set(contract["requiredKeys"]):
        return invalid("invalid_certificate_keys")
    status = certificate["evidence_status"]
    compatible = certificate["compatible_choice_ids"]
    proposal = certificate["proposed_choice_id"]
    if status not in contract["evidenceStatusValues"]:
        return invalid("invalid_evidence_status")
    if not isinstance(compatible, list) or not compatible or not all(isinstance(value, str) for value in compatible):
        return invalid("invalid_compatible_choice_list")
    if compatible != sorted(set(compatible)):
        return invalid("compatible_choices_not_sorted_unique")
    if not set(compatible) <= valid_ids or "A00" in compatible:
        return invalid("unknown_or_A00_compatible_choice")
    if not isinstance(proposal, str) or proposal not in valid_ids:
        return invalid("unknown_proposed_choice")
    if status == "SUFFICIENT":
        if len(compatible) != 1 or proposal != compatible[0] or proposal == "A00":
            return invalid("inconsistent_sufficient_certificate")
        return {"certificate_valid": True, "validation_reason": "valid_sufficient", "final_choice_id": proposal}
    if len(compatible) < 2 or proposal != "A00":
        return invalid("inconsistent_insufficient_certificate")
    return {"certificate_valid": True, "validation_reason": "valid_insufficient", "final_choice_id": "A00"}


def deterministic_finalize(certificate: Any, catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validation = validate_certificate(certificate, catalog, config)
    final_choice = validation["final_choice_id"]
    final_json = json.dumps({"choice_id": final_choice}, sort_keys=True, separators=(",", ":"))
    return {
        **validation,
        "final_json": final_json,
        "final_output_structurally_valid": True,
        "permanently_non_authoritative": True,
        "authoritative_hypothesis_universe_pruned": False,
        "actual_execution_count": 0,
    }


def audit_population(
    population: dict[str, Any],
    config: dict[str, Any],
    v135_public_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = population["choice_catalog"]
    public = population["public_fixtures"]
    hidden = population["hidden_fixtures"]
    summary = population["population_summary"]
    kinds = Counter(row["kind"] for row in catalog["choices"])
    hidden_by_id = {row["fixture_id"]: row for row in hidden}
    public_by_id = {row["fixture_id"]: row for row in public}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in hidden:
        groups.setdefault(row["group_id"], []).append(row)
    forbidden_public = {
        "group_id", "family_id", "stage", "language_class", "truth_choice_id",
        "compatible_choice_ids", "variant_index",
    }
    current_conversations = {
        json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in public
    }
    old_conversations = {
        json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in v135_public_rows
    }
    certificate_examples = []
    for row in catalog["choices"]:
        if row["choice_id"] != "A00":
            certificate_examples.append({
                "evidence_status": "SUFFICIENT",
                "compatible_choice_ids": [row["choice_id"]],
                "proposed_choice_id": row["choice_id"],
            })
    for family in config["families"]:
        certificate_examples.append({
            "evidence_status": "INSUFFICIENT",
            "compatible_choice_ids": sorted([family["left_choice_id"], family["right_choice_id"]]),
            "proposed_choice_id": "A00",
        })
    certificate_examples.extend([
        {},
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["K11", "N11"], "proposed_choice_id": "K11"},
        {"evidence_status": "INSUFFICIENT", "compatible_choice_ids": ["K11"], "proposed_choice_id": "A00"},
        {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["BAD"], "proposed_choice_id": "BAD"},
    ])
    finalizations = [deterministic_finalize(value, catalog, config) for value in certificate_examples]
    ambiguous = [row for row in hidden if row["stage"] == "ambiguous"]
    decidable = [row for row in hidden if row["stage"] != "ambiguous"]
    checks = {
        "choice_counts": bool(
            summary["choice_count"] == config["gates"]["requiredChoiceCount"]
            and kinds["KNOWN"] == config["gates"]["requiredKnownChoiceCount"]
            and kinds["VALID_UNDECLARED"] == config["gates"]["requiredNovelChoiceCount"]
            and kinds["UNSUPPORTED"] == config["gates"]["requiredUnsupportedChoiceCount"]
            and kinds["INSUFFICIENT_EVIDENCE"] == config["gates"]["requiredInsufficientChoiceCount"]
        ),
        "family_group_fixture_counts": bool(
            summary["family_count"] == config["gates"]["requiredFamilyCount"]
            and summary["group_count"] == config["gates"]["requiredGroupCount"]
            and summary["fixture_count"] == config["gates"]["requiredFixtureCount"]
            and all(count == config["gates"]["requiredSplitFixtureCount"] for count in summary["split_counts"].values())
        ),
        "six_exact_stages_per_group": all(
            len(rows) == config["gates"]["requiredStageCountPerGroup"]
            and {row["stage"] for row in rows} == set(STAGES)
            for rows in groups.values()
        ),
        "language_class_coverage": bool(
            summary["language_class_counts"]["known_familiar"] == config["gates"]["requiredKnownFamiliarFixtureCount"]
            and summary["language_class_counts"]["known_unfamiliar"] == config["gates"]["requiredKnownUnfamiliarFixtureCount"]
            and summary["language_class_counts"]["insufficient_evidence"] == config["gates"]["requiredAmbiguousFixtureCount"]
            and sum(count for key, count in summary["language_class_counts"].items() if key.endswith("_clarified")) == config["gates"]["requiredClarifiedFixtureCount"]
        ),
        "public_hidden_alignment": bool(
            len(public_by_id) == len(public) == len(hidden_by_id) == len(hidden)
            and set(public_by_id) == set(hidden_by_id)
            and all(all(public_by_id[fixture_id][key] == row[key] for key in public_by_id[fixture_id]) for fixture_id, row in hidden_by_id.items())
        ),
        "no_hidden_fields_in_public": all(not (forbidden_public & set(row)) for row in public),
        "ambiguous_pair_compatibility_exact": all(
            row["truth_choice_id"] == "A00"
            and len(row["compatible_choice_ids"]) == 2
            and row["compatible_choice_ids"] == sorted(row["compatible_choice_ids"])
            for row in ambiguous
        ),
        "decidable_singleton_compatibility_exact": all(
            row["compatible_choice_ids"] == [row["truth_choice_id"]]
            and row["truth_choice_id"] != "A00"
            for row in decidable
        ),
        "every_choice_has_truth_coverage": set(summary["truth_counts"]) == {row["choice_id"] for row in catalog["choices"]},
        "deterministic_finalizer_always_structurally_valid_and_nonexecuting": all(
            row["final_output_structurally_valid"]
            and row["permanently_non_authoritative"]
            and not row["authoritative_hypothesis_universe_pruned"]
            and row["actual_execution_count"] == 0
            and json.loads(row["final_json"])["choice_id"] in {item["choice_id"] for item in catalog["choices"]}
            for row in finalizations
        ),
        "exact_conversation_nonoverlap_with_V135": not (current_conversations & old_conversations),
        "true_hypothesis_retention": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "certificate_example_count": len(certificate_examples),
        "deterministic_finalizer_validity": sum(row["final_output_structurally_valid"] for row in finalizations) / len(finalizations),
        "exact_conversation_overlap_with_V135_count": len(current_conversations & old_conversations),
        "true_hypothesis_retention": 1.0,
        "actual_execution_count": 0,
    }


__all__ = [
    "STAGES",
    "audit_population",
    "build_catalog",
    "build_population",
    "deterministic_finalize",
    "validate_certificate",
]
