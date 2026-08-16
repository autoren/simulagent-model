#!/usr/bin/env python3
"""Generate text-free, blinded V58 author packets from sealed V40 meanings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


OPERATION_MEANINGS = {
    "assert": "present the embedded proposition as accepted",
    "deny": "reject the embedded proposition",
    "double_deny": "reject a denial of the embedded proposition",
    "contrast_select": "select the embedded proposition over a contrasted alternative",
    "unresolved": "report that the embedded proposition remains unsettled",
}
SIGN_MEANINGS = {
    "positive": "the embedded relation holds",
    "negative": "the embedded relation does not hold",
}
FAMILY_INSTRUCTIONS = {
    "plain_assertion": "Use a natural affirmative assertion without meta-language.",
    "lexical_negation": "Express the relation itself negatively, not merely uncertainty.",
    "denial": "Naturally reject the embedded proposition while keeping its internal sign.",
    "double_denial": "Use a genuine denial-of-denial construction without changing the target roles.",
    "contrastive_focus": "Contrast at least two possibilities and make the target proposition the selected focus.",
    "unresolved_status": "Make clear that the target proposition is unsettled rather than true or false.",
    "direct_relation": "Mention the semantic source before the semantic target.",
    "inverse_relation": "Use an inverse or passive realization while preserving canonical source and target roles.",
    "argument_reversal": "Place the target entity before the source entity on the surface while preserving their semantic roles.",
    "distractor_scope": "Include a plausible non-target fact but unambiguously scope the requested operation over the target proposition.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def family_matches(row: dict[str, Any], family: str) -> bool:
    meta = row["oracle_metadata"]
    return {
        "plain_assertion": (
            meta["operation"] == "assert" and meta["sign"] == "positive"
        ),
        "lexical_negation": (
            meta["operation"] == "assert" and meta["sign"] == "negative"
        ),
        "denial": meta["operation"] == "deny" and meta["sign"] == "positive",
        "double_denial": (
            meta["operation"] == "double_deny" and meta["sign"] == "positive"
        ),
        "contrastive_focus": meta["operation"] == "contrast_select",
        "unresolved_status": meta["operation"] == "unresolved",
        "direct_relation": meta["orientation"] == "direct",
        "inverse_relation": meta["orientation"] == "inverse",
        "argument_reversal": meta["argument_reversal"] is True,
        "distractor_scope": meta["decoy_kind"] != "non_state_distractor",
    }[family]


def _relation(row: dict[str, Any]) -> dict[str, Any]:
    predicate = row["target"]["parse"]["predicate"]
    return next(
        relation
        for relation in row["agent_input"]["predicate_ontology"]["relations"]
        if relation["id"] == predicate
    )


def known_ontology_glossary(row: dict[str, Any]) -> dict[str, Any]:
    ontology = row["agent_input"]["predicate_ontology"]
    target = row["target"]["parse"]
    predicates = []
    for relation in ontology["relations"]:
        predicates.append({
            "predicate_id": relation["id"],
            "kind": "directed_relation",
            "neutral_label": relation["id"].replace("_", " "),
            "source_type": relation["source_type"],
            "target_type": relation["target_type"],
        })
    for predicate in ontology["unary_predicates"]:
        predicates.append({
            "predicate_id": predicate["id"],
            "kind": "unary_predicate",
            "neutral_label": predicate["id"].replace("_", " "),
            "entity_type": predicate["entity_type"],
        })
    return {
        "known_predicates": predicates,
        "target_predicate_id": target["predicate"],
        "target_operation": {
            "id": target["outer_operation"],
            "meaning": OPERATION_MEANINGS[target["outer_operation"]],
        },
        "target_lexical_sign": {
            "id": target["lexical_sign"],
            "meaning": SIGN_MEANINGS[target["lexical_sign"]],
        },
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _select_source(
    rows: list[dict[str, Any]],
    used: set[str],
    family: str,
    desired_pack: str,
    key: str,
) -> dict[str, Any]:
    candidates = [
        row for row in rows if row["id"] not in used and family_matches(row, family)
    ]
    preferred = [row for row in candidates if row["ontology_pack"] == desired_pack]
    pool = preferred or candidates
    if not pool:
        raise RuntimeError(f"no unused V40 source remains for {family}")
    selected = min(pool, key=lambda row: _digest(f"{key}|{row['id']}"))
    used.add(selected["id"])
    return selected


def _prompt(
    row: dict[str, Any],
    packet_id: str,
    writer_id: str,
    cohort: str,
    family: str,
    stratum: str,
    ordinal: int,
    seed: str,
) -> dict[str, Any]:
    prompt_id = "v58p_" + _digest(
        f"{seed}|{writer_id}|{family}|{stratum}|{ordinal}|{row['id']}"
    )[:20]
    abstention_condition = None
    intended = row["target"]["parse"] if stratum == "primary" else None
    instructions = [
        "Write exactly one natural utterance.",
        "Do not copy an example or expose AST field names.",
        "Do not use a language model or other generative writing tool.",
    ]
    if stratum == "primary":
        instructions.extend([
            "Express the intended semantics uniquely.",
            "Preserve lexical sign, outer operation, and source/target roles.",
            FAMILY_INSTRUCTIONS[family],
        ])
    else:
        abstention_condition = (
            "genuinely_ambiguous_scope_or_referent"
            if ordinal % 2 == 0 else "unsupported_or_unknown_reference"
        )
        instructions.extend([
            (
                "Write an utterance with genuinely ambiguous scope or reference; "
                "do not announce the ambiguity."
                if abstention_condition.startswith("genuinely") else
                "Write an utterance containing a natural unsupported or unknown reference; do not announce that it is unsupported."
            ),
            "Ensure that no unique supported canonical AST can be assigned.",
            FAMILY_INSTRUCTIONS[family],
        ])
    return {
        "packet_id": packet_id,
        "prompt_id": prompt_id,
        "anonymous_writer_id": writer_id,
        "writer_cohort": cohort,
        "collection_round": "pilot_01" if cohort == "pilot" else "evaluation_01",
        "construction_family": family,
        "stratum": stratum,
        "abstention_condition": abstention_condition,
        "source_record_id": row["id"],
        "entity_legend": row["agent_input"]["entities"],
        "known_ontology_glossary": known_ontology_glossary(row),
        "intended_semantics": intended,
        "writing_instructions": instructions,
    }


def build_packets(
    protocol: dict[str, Any],
    source_rows: list[dict[str, Any]],
    seed: str,
    evaluation_reserve_slots: int = 2,
) -> list[dict[str, Any]]:
    split = protocol["constructionSplit"]
    pilot_families = split["pilotExposedFamilies"]
    evaluation_families = pilot_families + split["evaluationOnlyFamilies"]
    packs = sorted({row["ontology_pack"] for row in source_rows})
    used: set[str] = set()
    global_index = 0
    packets = []

    def add_packet(
        writer_id: str,
        cohort: str,
        status: str,
        families: list[str],
        primary_per_family: int,
        include_abstention: bool,
    ) -> None:
        nonlocal global_index
        packet_id = "v58packet_" + _digest(f"{seed}|{writer_id}")[:20]
        prompts = []
        for family in families:
            for ordinal in range(primary_per_family):
                desired_pack = packs[global_index % len(packs)]
                key = f"{seed}|{writer_id}|{family}|primary|{ordinal}"
                row = _select_source(
                    source_rows, used, family, desired_pack, key
                )
                prompts.append(_prompt(
                    row, packet_id, writer_id, cohort, family, "primary",
                    ordinal, seed,
                ))
                global_index += 1
            if include_abstention:
                desired_pack = packs[global_index % len(packs)]
                key = f"{seed}|{writer_id}|{family}|abstention|0"
                row = _select_source(
                    source_rows, used, family, desired_pack, key
                )
                prompts.append(_prompt(
                    row, packet_id, writer_id, cohort, family, "abstention",
                    families.index(family), seed,
                ))
                global_index += 1
        packets.append({
            "schema_version": 58,
            "experiment": "v58_blinded_text_free_author_packet",
            "packet_id": packet_id,
            "anonymous_writer_id": writer_id,
            "writer_cohort": cohort,
            "slot_status": status,
            "release_authorized": False,
            "identity_mapping_in_repository": False,
            "submission_license": protocol["submissionSchema"]["attestation"][
                "datasetLicense"
            ],
            "prompts": prompts,
        })

    for index in range(protocol["quotas"]["pilotAuthors"]):
        add_packet(
            f"pilot_writer_slot_{index:02d}",
            "pilot",
            "active",
            pilot_families,
            protocol["quotas"][
                "acceptedPrimaryPerPilotExposedFamilyPerPilotAuthor"
            ],
            False,
        )
    for index in range(protocol["quotas"]["minimumEvaluationAuthors"]):
        add_packet(
            f"evaluation_writer_slot_{index:02d}",
            "evaluation",
            "active",
            evaluation_families,
            protocol["quotas"]["acceptedPrimaryPerFamilyPerEvaluationAuthor"],
            True,
        )
    for index in range(evaluation_reserve_slots):
        add_packet(
            f"evaluation_reserve_slot_{index:02d}",
            "evaluation",
            "reserve",
            evaluation_families,
            protocol["quotas"]["acceptedPrimaryPerFamilyPerEvaluationAuthor"],
            True,
        )
    return packets


def packet_manifest(packets: list[dict[str, Any]], seed: str) -> dict[str, Any]:
    return {
        "schema_version": 58,
        "experiment": "v58_author_packet_manifest",
        "generation_seed": seed,
        "packets": [
            {
                "packet_id": packet["packet_id"],
                "anonymous_writer_id": packet["anonymous_writer_id"],
                "writer_cohort": packet["writer_cohort"],
                "slot_status": packet["slot_status"],
                "prompts": len(packet["prompts"]),
                "primary_prompts": sum(
                    row["stratum"] == "primary" for row in packet["prompts"]
                ),
                "abstention_prompts": sum(
                    row["stratum"] == "abstention" for row in packet["prompts"]
                ),
            }
            for packet in packets
        ],
        "counts": {
            "packets": len(packets),
            "pilot_active_packets": sum(
                row["writer_cohort"] == "pilot" for row in packets
            ),
            "evaluation_active_packets": sum(
                row["writer_cohort"] == "evaluation"
                and row["slot_status"] == "active" for row in packets
            ),
            "evaluation_reserve_packets": sum(
                row["writer_cohort"] == "evaluation"
                and row["slot_status"] == "reserve" for row in packets
            ),
            "prompts": sum(len(row["prompts"]) for row in packets),
            "unique_source_records": len({
                prompt["source_record_id"]
                for packet in packets for prompt in packet["prompts"]
            }),
        },
        "release": {
            "pilot_packets_released": 0,
            "evaluation_packets_released": 0,
            "reserve_packets_activated": 0,
        },
        "human_text": {
            "collected": 0,
            "accessed": 0,
            "fields_present": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator-lock", default="configs/v58-author-packet-generator-lock.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "data/v58-human-authored-known-ontology-language/author-packets"
        ),
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.generator_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 author packet target already exists")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["generate_v58_blinded_author_packets_once"]:
        raise RuntimeError("V58 author packet generator lock does not authorize generation")
    for path, digest in lock["generator_files_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != digest:
            raise RuntimeError(f"V58 frozen packet generator changed: {path}")
    protocol_path = PROJECT_ROOT / lock["protocol"]
    if file_sha256(protocol_path) != lock["protocol_sha256"]:
        raise RuntimeError("V58 frozen collection protocol changed")
    protocol = json.loads(protocol_path.read_text())
    source_path = PROJECT_ROOT / protocol["knownOntologySource"]["corePopulation"]
    if file_sha256(source_path) != lock["v40_core_sha256"]:
        raise RuntimeError("V58 sealed V40 source changed")
    seed = lock["packet_generation_seed"]
    packets = build_packets(protocol, read_jsonl(source_path), seed)
    output.mkdir(parents=True)
    artifacts = []
    for packet in packets:
        path = output / f"{packet['anonymous_writer_id']}.json"
        path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        artifacts.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "packet_id": packet["packet_id"],
            "anonymous_writer_id": packet["anonymous_writer_id"],
        })
    manifest = packet_manifest(packets, seed)
    manifest.update({
        "generator_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "generator_lock_sha256": file_sha256(lock_path),
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "v40_core": str(source_path.relative_to(PROJECT_ROOT)),
        "v40_core_sha256": file_sha256(source_path),
        "artifacts": artifacts,
    })
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "counts": manifest["counts"],
        "release": manifest["release"],
        "human_text": manifest["human_text"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
