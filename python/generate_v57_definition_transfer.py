#!/usr/bin/env python3
"""Independent V57 population generator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v57_definition_compiler import compiled_truth, render_controlled_definition


FAMILIES = ("signature_first", "meaning_first", "example_first")
OPERATIONS = ("assert", "deny", "double_deny", "contrast_select", "unresolved")
OPERATION_CUES = {
    "assert": "support",
    "deny": "rejection",
    "double_deny": "denial-blocking",
    "contrast_select": "selection",
    "unresolved": "undecided",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _concept(
    pack: int, ordinal: int, kind: str, signature: dict[str, str],
    family: str, seed: int,
) -> dict[str, Any]:
    token = digest(f"v57|{seed}|{pack}|{ordinal}|{kind}")[:7]
    identifier = "sym_" + digest(f"opaque|{seed}|{pack}|{ordinal}")[:10]
    word = "z" + token
    if kind == "unary_predicate":
        forms = {
            "positive": f"{{entity}} is {word}",
            "negative": f"{{entity}} is not {word}",
        }
        primary = forms["positive"]
    elif kind == "binary_relation":
        forms = {
            "direct_positive": f"{{source}} {word}s {{target}}",
            "direct_negative": f"{{source}} does not {word} {{target}}",
            "inverse_positive": f"{{target}} is {word}ed by {{source}}",
            "inverse_negative": f"{{target}} is not {word}ed by {{source}}",
        }
        primary = forms["direct_positive"]
    else:
        forms = {"command": f"{word} {{actor}} toward {{target}}"}
        primary = forms["command"]
    return {
        "opaque_id": identifier,
        "kind": kind,
        "typed_signature": signature,
        "controlled_definition": render_controlled_definition(
            identifier, kind, signature, forms, family
        ),
        "positive_or_command_form": primary,
        "lexical_forms": forms,
        "definition_template_family": family,
    }


def ontology_pack(
    pack: int, target_family: str = "signature_first", seed: int = 5707
) -> dict[str, Any]:
    carrier = f"pack{pack:02d}_carrier"
    station = f"pack{pack:02d}_station"
    entities = [
        {"id": f"c{pack:02d}_0", "entity_type": carrier},
        {"id": f"c{pack:02d}_1", "entity_type": carrier},
        {"id": f"s{pack:02d}_0", "entity_type": station},
        {"id": f"s{pack:02d}_1", "entity_type": station},
    ]
    specs = (
        ("unary_predicate", {"entity": carrier}),
        ("unary_predicate", {"entity": station}),
        ("binary_relation", {"source": carrier, "target": carrier}),
        ("binary_relation", {"source": station, "target": carrier}),
        ("bound_action", {"actor": carrier, "target": station}),
        ("bound_action", {"actor": station, "target": carrier}),
    )
    concepts = [
        _concept(pack, index, kind, signature, target_family, seed)
        for index, (kind, signature) in enumerate(specs)
    ]
    return {"entities": entities, "concepts": concepts}


def _format(concept: dict[str, Any], key: str, binding: dict[str, str]) -> str:
    return concept["lexical_forms"][key].format(**binding)


def _truth(parse: dict[str, Any]) -> str | None:
    return compiled_truth(parse)


def build_core(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seed = config["population"]["generatorSeed"]
    for pack in range(config["population"]["ontologyPacks"]):
        for ordinal in range(config["population"]["coreRecordsPerPack"]):
            kind_index = ordinal // 40
            concept_within_kind = ordinal % 2
            concept_index = kind_index * 2 + concept_within_kind
            family = FAMILIES[ordinal % len(FAMILIES)]
            ontology = ontology_pack(pack, family, seed)
            concept = ontology["concepts"][concept_index]
            entities = ontology["entities"]
            operation = OPERATIONS[(ordinal // 2) % len(OPERATIONS)]
            sign = "positive" if ordinal % 2 == 0 else "negative"
            if concept["kind"] == "unary_predicate":
                entity_type = concept["typed_signature"]["entity"]
                choices = [row["id"] for row in entities if row["entity_type"] == entity_type]
                binding = {"entity": choices[(ordinal // 3) % len(choices)]}
                focus = _format(concept, sign, binding)
                parse = {
                    "kind": "unary_predicate", "symbol": concept["opaque_id"],
                    "arguments": [binding["entity"]], "lexical_sign": sign,
                    "orientation": None, "outer_operation": operation,
                }
                evidence = (
                    f"Focal report: {focus}; Operation cue: {OPERATION_CUES[operation]}; "
                    "Context only: maintenance note."
                )
                orientation = None
            elif concept["kind"] == "binary_relation":
                signature = concept["typed_signature"]
                sources = [row["id"] for row in entities if row["entity_type"] == signature["source"]]
                targets = [row["id"] for row in entities if row["entity_type"] == signature["target"]]
                source = sources[(ordinal // 3) % len(sources)]
                target = targets[(ordinal // 5) % len(targets)]
                if source == target:
                    target = targets[(targets.index(target) + 1) % len(targets)]
                orientation = "direct" if (ordinal // 2) % 2 == 0 else "inverse"
                key = f"{orientation}_{sign}"
                focus = _format(concept, key, {"source": source, "target": target})
                parse = {
                    "kind": "binary_relation", "symbol": concept["opaque_id"],
                    "arguments": [source, target], "lexical_sign": sign,
                    "orientation": orientation, "outer_operation": operation,
                }
                evidence = (
                    f"Focal report: {focus}; Operation cue: {OPERATION_CUES[operation]}; "
                    "Context only: maintenance note."
                )
            else:
                signature = concept["typed_signature"]
                actors = [row["id"] for row in entities if row["entity_type"] == signature["actor"]]
                targets = [row["id"] for row in entities if row["entity_type"] == signature["target"]]
                actor = actors[(ordinal // 3) % len(actors)]
                target = targets[(ordinal // 5) % len(targets)]
                command = _format(concept, "command", {"actor": actor, "target": target})
                parse = {
                    "kind": "bound_action", "symbol": concept["opaque_id"],
                    "arguments": [actor, target], "lexical_sign": None,
                    "orientation": None, "outer_operation": None,
                }
                evidence = f"Action request: {command}."
                orientation = None
            token = f"{pack}|{ordinal}|{concept['opaque_id']}|{family}"
            rows.append({
                "id": "v57_" + digest(token)[:16],
                "schema_version": 57,
                "split": "core",
                "ontology_pack": f"pack_{pack:02d}",
                "public": {
                    "entities": entities,
                    "concept_definitions": ontology["concepts"],
                    "evidence_text": evidence,
                },
                "target": {"parse": parse, "compiled_truth": _truth(parse)},
                "oracle_metadata": {
                    "concept_kind": concept["kind"],
                    "definition_template_family": family,
                    "lexical_sign": parse["lexical_sign"],
                    "relation_orientation": orientation,
                    "outer_operation": parse["outer_operation"],
                    "target_concept_index": concept_index,
                },
            })
    expected = config["population"]["coreRecords"]
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise RuntimeError("V57 core census or IDs are invalid")
    return sorted(rows, key=lambda row: row["id"])


def _replace_target_concept(row: dict[str, Any], update) -> dict[str, Any]:
    result = copy.deepcopy(row)
    target_id = result["target"]["parse"]["symbol"]
    for index, concept in enumerate(result["public"]["concept_definitions"]):
        if concept["opaque_id"] == target_id:
            result["public"]["concept_definitions"][index] = update(concept)
            break
    return result


def safety_case(base: dict[str, Any], condition: str, ordinal: int) -> dict[str, Any]:
    row = copy.deepcopy(base)
    row["id"] = "v57s_" + digest(f"{base['id']}|{condition}|{ordinal}")[:16]
    row["split"] = "safety"
    target_id = row["target"]["parse"]["symbol"]
    if condition == "missing_definition":
        row["public"]["concept_definitions"] = [
            concept for concept in row["public"]["concept_definitions"]
            if concept["opaque_id"] != target_id
        ]
    elif condition == "unknown_lexeme":
        row["public"]["evidence_text"] = "Action request: unknownlex c00_0 toward s00_0."
    elif condition == "duplicate_lexeme":
        target = next(
            concept for concept in row["public"]["concept_definitions"]
            if concept["opaque_id"] == target_id
        )
        duplicate = copy.deepcopy(target)
        duplicate["opaque_id"] = "sym_" + digest(row["id"])[-10:]
        duplicate["controlled_definition"] = render_controlled_definition(
            duplicate["opaque_id"], duplicate["kind"], duplicate["typed_signature"],
            duplicate["lexical_forms"], duplicate["definition_template_family"],
        )
        row["public"]["concept_definitions"].append(duplicate)
    elif condition == "incomplete_signature":
        row = _replace_target_concept(row, lambda concept: {
            **concept, "typed_signature": {},
        })
    elif condition == "type_mismatch":
        entities = row["public"]["entities"]
        target = next(
            concept for concept in row["public"]["concept_definitions"]
            if concept["opaque_id"] == target_id
        )
        required_types = set(target["typed_signature"].values())
        wrong_row = next((
            entity for entity in entities
            if entity["entity_type"] not in required_types
        ), None)
        if wrong_row is None:
            wrong_row = {"id": f"alien_{ordinal}", "entity_type": "alien_type"}
            row["public"]["entities"].append(wrong_row)
        wrong = wrong_row["id"]
        evidence = row["public"]["evidence_text"]
        identifiers = [entity["id"] for entity in entities]
        original = next((identifier for identifier in identifiers if identifier in evidence), None)
        if original:
            row["public"]["evidence_text"] = evidence.replace(original, wrong, 1)
    elif condition == "ambiguous_role_order":
        row = _replace_target_concept(row, lambda concept: {
            **concept,
            "typed_signature": {"participant_a": "unknown", "participant_b": "unknown"},
        })
    elif condition == "contradictory_definition":
        def corrupt(concept):
            parsed_forms = dict(concept["lexical_forms"])
            first = next(iter(parsed_forms))
            parsed_forms[first] += " contradicted"
            return {
                **concept,
                "controlled_definition": render_controlled_definition(
                    concept["opaque_id"], concept["kind"],
                    concept["typed_signature"], parsed_forms,
                    concept["definition_template_family"],
                ),
            }
        row = _replace_target_concept(row, corrupt)
    else:
        raise ValueError(condition)
    row.pop("target")
    row["expected"] = {"statuses": ["abstain", "ambiguous"], "condition": condition}
    row["oracle_metadata"] = {"condition": condition, "source_id": base["id"]}
    return row


def build_safety(core: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    conditions = config["safetyPopulation"]["conditions"]
    for pack in range(config["population"]["ontologyPacks"]):
        pack_rows = [row for row in core if row["ontology_pack"] == f"pack_{pack:02d}"]
        for condition_index, condition in enumerate(conditions):
            for ordinal in range(config["safetyPopulation"]["recordsPerConditionPerPack"]):
                rows.append(safety_case(
                    pack_rows[condition_index * 3 + ordinal], condition, ordinal
                ))
    if len(rows) != config["safetyPopulation"]["records"]:
        raise RuntimeError("V57 safety census is invalid")
    return sorted(rows, key=lambda row: row["id"])


def build_populations(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    core = build_core(config)
    return {"core": core, "safety": build_safety(core, config)}


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return digest(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v57-implementation-lock.json")
    parser.add_argument("--output", default="data/v57-definition-augmented-ontology-transfer")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V57 population target already exists")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_v57_population"]:
        raise RuntimeError("V57 implementation lock does not authorize construction")
    for path, expected in lock["implementation_files_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V57 locked implementation changed: {path}")
    populations = build_populations(lock["config_payload"])
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in populations.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ))
        artifacts[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "records": len(rows), "sha256": file_sha256(path),
        }
    manifest = {
        "schema_version": 57,
        "experiment": "v57_population_manifest",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "truth_access_count": 0,
        "evaluation_runs": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
