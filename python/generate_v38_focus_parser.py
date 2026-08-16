#!/usr/bin/env python3
"""Construct the exact in-memory/materialized V38 anti-shortcut populations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import ENTITY_ALIASES, canonical_json, deterministic_shuffle, opaque_id, sha256_text
from v32_language import compile_truth
from v38_focus_parser import (
    NON_STATE_DECOYS, SURFACE_TEMPLATES, deterministic_focus_index,
    extract_literal_candidates, normalized_template, ontology_with_lexical_forms,
    render_form, render_surface,
)


def make_entities(count: int, token: str, seed: int):
    aliases = deterministic_shuffle(ENTITY_ALIASES, f"v38|{seed}|{token}|aliases")
    types = ["unit", "unit", "hub"]
    while len(types) < count:
        types.append("unit" if len(types) % 2 else "hub")
    return deterministic_shuffle([
        {"id": aliases[index], "entity_type": entity_type}
        for index, entity_type in enumerate(types)
    ], f"v38|{seed}|{token}|order")


def relation_spec(config, predicate):
    return next(row for row in config["ontology"]["relations"] if row["id"] == predicate)


def relation_text(config, predicate, arguments, sign, orientation):
    spec = relation_spec(config, predicate)
    key = f"{orientation}{'True' if sign == 'positive' else 'False'}Form"
    return render_form(spec[key], arguments)


def build_population(config: dict[str, Any], v32_config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    seed = config["developmentPopulation"]["generatorSeed"]
    registry_name = "fit" if split == "ontology_focus_fit" else "validation"
    records = []
    for operation_index, operation in enumerate(SURFACE_TEMPLATES[registry_name]):
        for surface_index, surface in enumerate(SURFACE_TEMPLATES[registry_name][operation]):
            for sign_index, sign in enumerate(config["developmentPopulation"]["lexicalSignsPerOperation"]):
                for order_index, focus_order in enumerate(config["developmentPopulation"]["focusOrders"]):
                    for decoy_index, decoy_kind in enumerate(config["developmentPopulation"]["decoyKinds"]):
                        for orientation_index, orientation in enumerate(config["developmentPopulation"]["relationOrientations"]):
                            count = config["developmentPopulation"]["entityCounts"][
                                (surface_index + sign_index + decoy_index + orientation_index) % 3
                            ]
                            token = f"{registry_name}|{operation}|{surface}|{sign}|{focus_order}|{decoy_kind}|{orientation}"
                            entities = make_entities(count, token, seed)
                            units = [row["id"] for row in entities if row["entity_type"] == "unit"]
                            hubs = [row["id"] for row in entities if row["entity_type"] == "hub"]
                            predicate = "linked" if (operation_index + surface_index) % 2 == 0 else "feeds"
                            arguments = [units[0], units[1]] if predicate == "linked" else [hubs[0], units[0]]
                            reversed_arguments = predicate == "linked" and (order_index + decoy_index) % 2 == 1
                            if reversed_arguments:
                                arguments = list(reversed(arguments))
                            focus_text = relation_text(v32_config, predicate, arguments, sign, orientation)
                            if decoy_kind == "exact_opposite":
                                decoy_text = relation_text(
                                    v32_config, predicate, arguments,
                                    "negative" if sign == "positive" else "positive", orientation,
                                )
                            elif decoy_kind == "different_grounded_atom":
                                if predicate == "linked":
                                    decoy_text = relation_text(v32_config, "feeds", [hubs[0], units[0]], sign, orientation)
                                else:
                                    decoy_text = relation_text(v32_config, "linked", [units[0], units[1]], sign, orientation)
                            else:
                                decoy_text = NON_STATE_DECOYS[focus_order]
                            # Surface A puts focus first; B puts it second. Cross the registered order
                            # factor by swapping which surface realization is used within each pair.
                            realized_surface = surface
                            template = SURFACE_TEMPLATES[registry_name][operation][realized_surface]
                            template_focus_first = template.index("{focus}") < template.index("{decoy}")
                            desired_focus_first = focus_order == "focus_first"
                            if template_focus_first != desired_focus_first:
                                other = "focus_b" if realized_surface == "focus_a" else "focus_a"
                                realized_surface = other
                            evidence = render_surface(registry_name, operation, realized_surface, focus_text, decoy_text)
                            record_id = opaque_id("v38", token)
                            row = {
                                "id": record_id,
                                "schema_version": 38,
                                "split": split,
                                "scene_id": opaque_id("scene", token),
                                "agent_input": {
                                    "entities": entities,
                                    "predicate_ontology": ontology_with_lexical_forms(v32_config),
                                    "evidence_text": evidence,
                                },
                                "target": {
                                    "focus": {
                                        "predicate": predicate,
                                        "arguments": arguments,
                                        "lexical_sign": sign,
                                        "orientation": orientation,
                                        "text": focus_text,
                                    },
                                    "outer_operation": operation,
                                    "truth_status": compile_truth(sign, operation, v32_config),
                                },
                                "oracle_metadata": {
                                    "surface_name": realized_surface,
                                    "surface_family": f"{operation}.{realized_surface}",
                                    "registry_split": registry_name,
                                    "normalized_template": normalized_template(registry_name, operation, realized_surface),
                                    "focus_order": focus_order,
                                    "decoy_kind": decoy_kind,
                                    "orientation": orientation,
                                    "argument_reversal": reversed_arguments,
                                    "entity_count": count,
                                },
                            }
                            candidates = extract_literal_candidates(row)
                            focus_index = deterministic_focus_index(row, candidates)
                            focus = candidates[focus_index]
                            if (focus.predicate, list(focus.arguments), focus.sign, focus.orientation, focus.text) != (
                                predicate, arguments, sign, orientation, focus_text
                            ):
                                raise ValueError("V38 deterministic grammar selected the wrong focus")
                            row["oracle_metadata"]["grounded_literal_candidates"] = len(candidates)
                            row["target"]["focus_candidate_index"] = focus_index
                            records.append(row)
    if len(records) != 240 or len({row["id"] for row in records}) != 240:
        raise ValueError(f"V38 {split} population is not exactly 240 records")
    return sorted(records, key=lambda row: row["id"])


def corpus_hash(rows):
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v38-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_corpus"]:
        raise RuntimeError("V38 implementation lock does not authorize construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V38 locked implementation changed: {path}")
    output = PROJECT_ROOT / "data/v38-ontology-anchored-focus-parser"
    if output.exists():
        raise RuntimeError("V38 corpus already exists")
    config, v32_config = lock["config_payload"], lock["v32_config_payload"]
    populations = {
        "ontology_focus_fit": build_population(config, v32_config, "ontology_focus_fit"),
        "ontology_focus_validation": build_population(config, v32_config, "ontology_focus_validation"),
    }
    for name, rows in populations.items():
        if corpus_hash(rows) != lock["expected_corpus_sha256"][name]:
            raise RuntimeError(f"V38 {name} differs from the implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in populations.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        artifacts[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(rows), "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 38, "experiment": config["experiment"],
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)), "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "counts": {
            name: {
                "focus_order": dict(Counter(row["oracle_metadata"]["focus_order"] for row in rows)),
                "decoy_kind": dict(Counter(row["oracle_metadata"]["decoy_kind"] for row in rows)),
                "orientation": dict(Counter(row["oracle_metadata"]["orientation"] for row in rows)),
                "candidate_count": dict(Counter(str(row["oracle_metadata"]["grounded_literal_candidates"]) for row in rows)),
            } for name, rows in populations.items()
        },
        "data_access": {"model_forward_passes": 0, "fit_runs": 0, "validation_evaluations": 0, "v32_evaluation_records_read": 0, "v28_runs": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
