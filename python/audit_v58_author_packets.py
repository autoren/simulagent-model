#!/usr/bin/env python3
"""Audit canonical or in-memory V58 text-free author packets."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from generate_v58_author_packets import build_packets, read_jsonl
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


PACKET_FIELDS = {
    "schema_version",
    "experiment",
    "packet_id",
    "anonymous_writer_id",
    "writer_cohort",
    "slot_status",
    "release_authorized",
    "identity_mapping_in_repository",
    "submission_license",
    "prompts",
}


def audit_packet_objects(
    packets: list[dict[str, Any]],
    protocol: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    source_by_id = {row["id"]: row for row in source_rows}
    prompts = [prompt for packet in packets for prompt in packet.get("prompts", [])]
    if len(packets) != 14:
        errors.append("packet_census")
    if any(set(packet) != PACKET_FIELDS for packet in packets):
        errors.append("packet_fields")
    if any(
        packet["release_authorized"]
        or packet["identity_mapping_in_repository"]
        for packet in packets
    ):
        errors.append("release_or_identity_mapping")
    if len({packet["packet_id"] for packet in packets}) != len(packets):
        errors.append("duplicate_packet_id")
    if len({packet["anonymous_writer_id"] for packet in packets}) != len(packets):
        errors.append("duplicate_writer_slot")
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        errors.append("duplicate_prompt_id")
    if len({row["source_record_id"] for row in prompts}) != len(prompts):
        errors.append("duplicate_source_record")

    prompt_fields = {
        "packet_id", "prompt_id", "anonymous_writer_id", "writer_cohort",
        "collection_round", "construction_family", "stratum",
        "abstention_condition", "source_record_id", "entity_legend",
        "known_ontology_glossary", "intended_semantics",
        "writing_instructions",
    }
    pack_counts = Counter()
    family_counts = Counter()
    for packet in packets:
        expected_prompt_count = 60 if packet["writer_cohort"] == "pilot" else 70
        if len(packet["prompts"]) != expected_prompt_count:
            errors.append("per_packet_prompt_census")
        for prompt in packet["prompts"]:
            if set(prompt) != prompt_fields:
                errors.append("prompt_fields")
                continue
            if (
                prompt["packet_id"] != packet["packet_id"]
                or prompt["anonymous_writer_id"]
                != packet["anonymous_writer_id"]
                or prompt["writer_cohort"] != packet["writer_cohort"]
            ):
                errors.append("packet_prompt_binding")
            source = source_by_id.get(prompt["source_record_id"])
            if source is None:
                errors.append("unknown_source_record")
                continue
            from generate_v58_author_packets import family_matches
            if not family_matches(source, prompt["construction_family"]):
                errors.append("construction_source_mismatch")
            if prompt["entity_legend"] != source["agent_input"]["entities"]:
                errors.append("entity_legend_mismatch")
            if prompt["stratum"] == "primary":
                if (
                    prompt["intended_semantics"] != source["target"]["parse"]
                    or prompt["abstention_condition"] is not None
                ):
                    errors.append("primary_target_mismatch")
            elif prompt["stratum"] == "abstention":
                if (
                    prompt["intended_semantics"] is not None
                    or prompt["abstention_condition"] not in protocol[
                        "abstentionPrompt"
                    ]["conditions"]
                ):
                    errors.append("abstention_target_mismatch")
            else:
                errors.append("unknown_stratum")
            serialized = json.dumps(prompt, sort_keys=True)
            forbidden_strings = [
                source["agent_input"]["evidence_text"],
                source["oracle_metadata"]["focus_text"],
                source["oracle_metadata"]["decoy_text"],
                source["oracle_metadata"]["cue"],
            ]
            ontology = source["agent_input"]["predicate_ontology"]
            forbidden_strings.extend(
                value
                for relation in ontology["relations"]
                for key, value in relation.items() if key.endswith("_form")
            )
            forbidden_strings.extend(
                value
                for predicate in ontology["unary_predicates"]
                for key, value in predicate.items() if key.endswith("_form")
            )
            if (
                "evidence_text" in serialized
                or "oracle_metadata" in serialized
                or "submitted_text" in serialized
                or any(value in serialized for value in forbidden_strings)
            ):
                errors.append("reference_surface_or_human_text_leak")
            pack_counts[source["ontology_pack"]] += 1
            family_counts[(
                packet["writer_cohort"], packet["slot_status"],
                prompt["construction_family"], prompt["stratum"],
            )] += 1

    split = protocol["constructionSplit"]
    families = split["pilotExposedFamilies"] + split["evaluationOnlyFamilies"]
    for family in split["pilotExposedFamilies"]:
        if family_counts[("pilot", "active", family, "primary")] != 24:
            errors.append("pilot_family_balance")
    for family in split["evaluationOnlyFamilies"]:
        if family_counts[("pilot", "active", family, "primary")] != 0:
            errors.append("pilot_holdout_leak")
    for family in families:
        if family_counts[("evaluation", "active", family, "primary")] != 60:
            errors.append("evaluation_primary_family_balance")
        if family_counts[("evaluation", "active", family, "abstention")] != 10:
            errors.append("evaluation_abstention_family_balance")
        if family_counts[("evaluation", "reserve", family, "primary")] != 12:
            errors.append("reserve_primary_family_balance")
        if family_counts[("evaluation", "reserve", family, "abstention")] != 2:
            errors.append("reserve_abstention_family_balance")
    packet_groups = Counter(
        (packet["writer_cohort"], packet["slot_status"])
        for packet in packets
    )
    census_ok = (
        packet_groups == {
            ("pilot", "active"): 2,
            ("evaluation", "active"): 10,
            ("evaluation", "reserve"): 2,
        }
        and len(prompts) == 960
        and len(pack_counts) == 12
        and max(pack_counts.values()) - min(pack_counts.values()) <= 4
    )
    if not census_ok:
        errors.append("packet_group_prompt_or_pack_balance")
    return {
        "passed": not errors,
        "errors": sorted(Counter(errors).items()),
        "metrics": {
            "packets": len(packets),
            "prompts": len(prompts),
            "unique_prompt_ids": len({row["prompt_id"] for row in prompts}),
            "unique_source_records": len({
                row["source_record_id"] for row in prompts
            }),
            "packet_groups": {
                f"{cohort}_{status}": count
                for (cohort, status), count in sorted(packet_groups.items())
            },
            "ontology_pack_counts": dict(sorted(pack_counts.items())),
            "human_text_fields": sum(
                "submitted_text" in row for row in prompts
            ),
            "released_packets": sum(
                packet["release_authorized"] for packet in packets
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-dir",
        default="data/v58-human-authored-known-ontology-language/author-packets",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v58-human-authored-known-ontology-language/"
            "author-packet-audit.json"
        ),
    )
    args = parser.parse_args()
    packet_dir = (PROJECT_ROOT / args.packet_dir).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    manifest_path = packet_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lock_path = PROJECT_ROOT / manifest["generator_lock"]
    lock = json.loads(lock_path.read_text())
    protocol_path = PROJECT_ROOT / manifest["protocol"]
    protocol = json.loads(protocol_path.read_text())
    source_path = PROJECT_ROOT / manifest["v40_core"]
    source_rows = read_jsonl(source_path)
    errors: list[str] = []

    bindings_ok = (
        file_sha256(lock_path) == manifest["generator_lock_sha256"]
        and file_sha256(protocol_path) == manifest["protocol_sha256"]
        and file_sha256(source_path) == manifest["v40_core_sha256"]
        and manifest["release"] == {
            "pilot_packets_released": 0,
            "evaluation_packets_released": 0,
            "reserve_packets_activated": 0,
        }
        and manifest["human_text"] == {
            "collected": 0,
            "accessed": 0,
            "fields_present": 0,
        }
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in manifest["artifacts"]
        )
    )
    if not bindings_ok:
        errors.append("V58 author packet manifest or artifacts are not intact")

    packets = [
        json.loads((PROJECT_ROOT / row["path"]).read_text())
        for row in manifest["artifacts"]
    ]
    object_audit = audit_packet_objects(packets, protocol, source_rows)
    if not object_audit["passed"]:
        errors.append("V58 author packet content audit failed")

    expected_packets = build_packets(
        protocol, source_rows, manifest["generation_seed"]
    )
    deterministic_ok = packets == expected_packets
    if not deterministic_ok:
        errors.append("V58 author packets do not reproduce deterministically")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v58-author-packet-seal.json",
            "data/v58-human-authored-known-ontology-language/pilot-submissions",
            "data/v58-human-authored-known-ontology-language/evaluation-submissions",
            "configs/v58-pilot-population-seal.json",
            "configs/v58-candidate-lock.json",
        )
    )
    if not downstream_absent:
        errors.append("V58 packet seal, human text, or candidate artifact already exists")

    audit = {
        "schema_version": 58,
        "experiment": "v58_author_packet_audit",
        "passed": not errors,
        "decision": (
            "authorize_v58_author_packet_seal"
            if not errors else "repair_v58_author_packets"
        ),
        "errors": errors,
        "packet_directory": str(packet_dir.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "generator_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "generator_lock_sha256": file_sha256(lock_path),
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "v40_core": str(source_path.relative_to(PROJECT_ROOT)),
        "v40_core_sha256": file_sha256(source_path),
        "checks": {
            "manifest_source_lock_and_artifact_bindings": bindings_ok,
            "packet_content_census_balance_and_no_surface_leak": object_audit[
                "passed"
            ],
            "deterministic_packet_reproduction": deterministic_ok,
            "packet_seal_human_text_and_candidate_absent": downstream_absent,
        },
        "packet_metrics": object_audit["metrics"],
        "packet_content_errors": object_audit["errors"],
        "data_access": {
            "human_authored_records_collected": 0,
            "human_authored_text_accessed": 0,
            "packets_released": 0,
            "candidate_evaluation_runs": 0,
            "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
