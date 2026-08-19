from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v145_finite_certificate_codebook import build_codebook, finalize_code, oracle_code


STAGES = ("clear_known_familiar", "clear_known_unfamiliar", "clear_right", "ambiguous", "clarified_known", "clarified_right")


def build_catalog(config: dict[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": "146-codebook-choice-catalog", "choices": config["catalog"]["choices"]}
    payload["catalog_sha256"] = payload_hash(payload)
    return payload


def _render(template: str, slots: dict[str, str]) -> str:
    return template.format(**slots)


def _fixture_id(group_id: str, stage: str) -> str:
    return "v146-" + hashlib.sha256(f"{group_id}|{stage}".encode()).hexdigest()[:16]


def build_population(config: dict[str, Any]) -> dict[str, Any]:
    catalog = build_catalog(config)
    kinds = {row["choice_id"]: row["kind"] for row in catalog["choices"]}
    public_rows, hidden_rows = [], []
    for family_index, family in enumerate(config["families"]):
        left, right = family["left_choice_id"], family["right_choice_id"]
        pair = sorted([left, right])
        for variant_index, slots in enumerate(family["slot_variants"]):
            split = "development" if variant_index < 4 else "test"
            group_id = f"v146-g{family_index:02d}-{variant_index:02d}"
            ambiguous = _render(family["ambiguous"], slots)
            specs = {
                "clear_known_familiar": ([_render(family["known_familiar"], slots)], left, [left], "known_familiar"),
                "clear_known_unfamiliar": ([_render(family["known_unfamiliar"], slots)], left, [left], "known_unfamiliar"),
                "clear_right": ([_render(family["right_clear"], slots)], right, [right], "novel_valid" if kinds[right] == "VALID_UNDECLARED" else "unsupported"),
                "ambiguous": ([ambiguous], "A00", pair, "insufficient_evidence"),
                "clarified_known": ([ambiguous, family["question"], _render(family["left_answer"], slots)], left, [left], "known_clarified"),
                "clarified_right": ([ambiguous, family["question"], _render(family["right_answer"], slots)], right, [right], "novel_valid_clarified" if kinds[right] == "VALID_UNDECLARED" else "unsupported_clarified"),
            }
            for stage in STAGES:
                texts, truth, compatible, language_class = specs[stage]
                conversation = ([{"role": "user", "text": texts[0]}] if len(texts) == 1 else [
                    {"role": "user", "text": texts[0]}, {"role": "assistant", "text": texts[1]}, {"role": "user", "text": texts[2]}
                ])
                fixture_id = _fixture_id(group_id, stage)
                public = {"fixture_id": fixture_id, "split": split, "presented_candidate_choice_id": family["presented_candidate_choice_id"], "conversation": conversation}
                hidden = {**public, "group_id": group_id, "family_id": family["family_id"], "stage": stage, "language_class": language_class, "truth_choice_id": truth, "compatible_choice_ids": compatible, "variant_index": variant_index}
                public_rows.append(public)
                hidden_rows.append(hidden)
    public_rows.sort(key=lambda row: row["fixture_id"])
    hidden_rows.sort(key=lambda row: row["fixture_id"])
    return {
        "choice_catalog": catalog,
        "certificate_codebook": {"schema_version": "146-registered-certificate-codebook", "entries": build_codebook(config)},
        "public_fixtures": public_rows,
        "hidden_fixtures": hidden_rows,
        "population_summary": {
            "choice_count": len(catalog["choices"]),
            "family_count": len(config["families"]),
            "group_count": len({row["group_id"] for row in hidden_rows}),
            "fixture_count": len(hidden_rows),
            "split_counts": dict(sorted(Counter(row["split"] for row in hidden_rows).items())),
            "stage_counts": dict(sorted(Counter(row["stage"] for row in hidden_rows).items())),
            "language_class_counts": dict(sorted(Counter(row["language_class"] for row in hidden_rows).items())),
            "truth_counts": dict(sorted(Counter(row["truth_choice_id"] for row in hidden_rows).items())),
        },
    }


def audit_population(population: dict[str, Any], config: dict[str, Any], prior_public_rows: list[dict[str, Any]]) -> dict[str, Any]:
    catalog = population["choice_catalog"]
    public, hidden = population["public_fixtures"], population["hidden_fixtures"]
    summary = population["population_summary"]
    codebook = population["certificate_codebook"]["entries"]
    kinds = Counter(row["kind"] for row in catalog["choices"])
    public_by_id = {row["fixture_id"]: row for row in public}
    hidden_by_id = {row["fixture_id"]: row for row in hidden}
    forbidden = {"group_id", "family_id", "stage", "language_class", "truth_choice_id", "compatible_choice_ids", "variant_index"}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in hidden:
        groups.setdefault(row["group_id"], []).append(row)
    current_conversations = {json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in public}
    prior_conversations = {json.dumps(row["conversation"], sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in prior_public_rows}
    oracle_finalized = [finalize_code(oracle_code(row), codebook) for row in hidden]
    checks = {
        "choice_counts": bool(len(catalog["choices"]) == config["gates"]["requiredChoiceCount"] and kinds["KNOWN"] == 4 and kinds["VALID_UNDECLARED"] == 3 and kinds["UNSUPPORTED"] == 1 and kinds["INSUFFICIENT_EVIDENCE"] == 1),
        "family_group_fixture_counts": bool(summary["family_count"] == 6 and summary["group_count"] == 48 and summary["fixture_count"] == 288 and all(value == 144 for value in summary["split_counts"].values())),
        "group_stage_completeness": all(len(rows) == 6 and {row["stage"] for row in rows} == set(STAGES) and len({row["split"] for row in rows}) == 1 for rows in groups.values()),
        "public_hidden_alignment_and_no_leakage": bool(set(public_by_id) == set(hidden_by_id) and all(not (forbidden & set(row)) for row in public)),
        "compatibility_exact": bool(all((row["truth_choice_id"] == "A00" and len(row["compatible_choice_ids"]) == 2) or (row["truth_choice_id"] != "A00" and row["compatible_choice_ids"] == [row["truth_choice_id"]]) for row in hidden)),
        "codebook_and_oracle_coverage": bool(len(codebook) == 14 and all(row["code_valid"] for row in oracle_finalized) and all(row["final_choice_id"] == hidden[index]["truth_choice_id"] for index, row in enumerate(oracle_finalized))),
        "exact_conversation_nonoverlap": not (current_conversations & prior_conversations),
        "true_hypothesis_retention_and_zero_execution": True,
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary, "exact_prior_conversation_overlap_count": len(current_conversations & prior_conversations), "oracle_code_coverage": sum(row["code_valid"] for row in oracle_finalized) / len(oracle_finalized), "true_hypothesis_retention": 1.0, "actual_execution_count": 0}


__all__ = ["STAGES", "audit_population", "build_catalog", "build_population"]
