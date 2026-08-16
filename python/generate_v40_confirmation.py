#!/usr/bin/env python3
"""Independent V40 generator; intentionally does not import V39 rendering code."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import canonical_json, opaque_id, sha256_text


PACK_NAMES = ("amber", "birch", "cobalt", "dune", "ember", "flint", "grove", "harbor", "indigo", "juniper", "kelp", "lumen")
LINK_WORDS = (
    ("tethered to", "tether"), ("paired with", "pairing"), ("bridged to", "bridge"),
    ("chained to", "chain"), ("spliced with", "splice"), ("lashed to", "lashing"),
    ("joined with", "joining"), ("fastened to", "fastening"), ("coupled with", "coupling"),
    ("bound to", "binding"), ("meshed with", "mesh"), ("latched to", "latch"),
)
ROUTE_WORDS = (
    ("supplies", "supplied"), ("powers", "powered"), ("provisions", "provisioned"),
    ("energizes", "energized"), ("routes into", "routed into"), ("streams to", "streamed to"),
    ("channels to", "channeled to"), ("relays to", "relayed to"), ("serves", "served"),
    ("drives", "driven"), ("dispatches to", "dispatched to"), ("transmits to", "transmitted to"),
)
OPERATOR_ROOTS = {
    "assert": "verified acceptance",
    "deny": "registered rejection",
    "double_deny": "rejection reversal",
    "contrast_select": "preferred alternative",
    "unresolved": "pending verdict",
}
SEPARATORS = {"period": ". ", "semicolon": "; ", "em_dash": " — "}


def ontology_pack(index: int) -> tuple[dict[str, Any], dict[str, str]]:
    name = PACK_NAMES[index]
    unit_type, hub_type = f"{name}_carrier", f"{name}_station"
    link_word, link_noun = LINK_WORDS[index]
    route_word, routed_word = ROUTE_WORDS[index]
    link_id, route_id = f"{name}_bond", f"{name}_transfer"
    ontology = {
        "entity_types": [unit_type, hub_type],
        "unary_predicates": [
            {"id": f"{name}_ready", "entity_type": unit_type, "positive_form": "{entity} is primed", "negative_form": "{entity} is unprimed"},
            {"id": f"{name}_awake", "entity_type": hub_type, "positive_form": "{entity} is awake", "negative_form": "{entity} is dormant"},
        ],
        "relations": [
            {
                "id": link_id, "source_type": unit_type, "target_type": unit_type,
                "direct_positive_form": f"{{source}} is {link_word} {{target}}",
                "direct_negative_form": f"{{source}} is not {link_word} {{target}}",
                "inverse_positive_form": f"{{target}} has a {link_noun} from {{source}}",
                "inverse_negative_form": f"{{target}} has no {link_noun} from {{source}}",
            },
            {
                "id": route_id, "source_type": hub_type, "target_type": unit_type,
                "direct_positive_form": f"{{source}} {route_word} {{target}}",
                "direct_negative_form": f"{{source}} does not {route_word} {{target}}",
                "inverse_positive_form": f"{{target}} is {routed_word} by {{source}}",
                "inverse_negative_form": f"{{target}} is not {routed_word} by {{source}}",
            },
        ],
    }
    cues = {operation: f"{name} {root}" for operation, root in OPERATOR_ROOTS.items()}
    return ontology, cues


def operator_ontology(cues: dict[str, str]) -> dict[str, Any]:
    return {
        "operations": [{"id": operation, "cues": [cue]} for operation, cue in cues.items()],
        "grammar": {
            "roles": {"focus": "Focal report:", "operation": "Operation cue:", "context": "Context only:"},
            "productions": [
                "{focus_label}{literal}{separator}{operation_label}{cue}{separator}{context_label}{aside}.",
                "{context_label}{aside}{separator}{operation_label}{cue}{separator}{focus_label}{literal}.",
            ],
            "separators": SEPARATORS,
            "constraint": "exactly one focus, operation, and context field; one separator realization per record",
        },
    }


def make_entities(pack_index: int, count: int, token: str, ontology: dict[str, Any]) -> list[dict[str, str]]:
    unit_type, hub_type = ontology["entity_types"]
    prefix = PACK_NAMES[pack_index][:3]
    suffixes = ("aro", "bex", "cim", "dov", "eri")
    types = [unit_type, unit_type, hub_type, unit_type, hub_type][:count]
    shift = int(sha256_text(token)[:4], 16) % len(suffixes)
    return [
        {"id": f"{prefix}{pack_index + 1:02d}_{suffixes[(shift + position) % len(suffixes)]}", "entity_type": entity_type}
        for position, entity_type in enumerate(types)
    ]


def render_literal(spec: dict[str, Any], arguments: list[str], sign: str, orientation: str) -> str:
    template = spec[f"{orientation}_{sign}_form"]
    return template.format(source=arguments[0], target=arguments[1])


def render_evidence(focus: str, decoy: str, cue: str, focus_order: str, punctuation: str) -> str:
    separator = SEPARATORS[punctuation]
    if focus_order == "focus_first":
        return f"Focal report: {focus}{separator}Operation cue: {cue}{separator}Context only: {decoy}."
    return f"Context only: {decoy}{separator}Operation cue: {cue}{separator}Focal report: {focus}."


def compile_reference_truth(sign: str, operation: str) -> str:
    table = {
        "assert": {"positive": "true", "negative": "false"},
        "deny": {"positive": "false", "negative": "true"},
        "double_deny": {"positive": "true", "negative": "false"},
        "contrast_select": {"positive": "true", "negative": "false"},
        "unresolved": {"positive": "unknown", "negative": "unknown"},
    }
    return table[operation][sign]


def build_core(config: dict[str, Any]) -> list[dict[str, Any]]:
    factors = config["confirmationPopulation"]["coreFactors"]
    rows = []
    for pack_index in range(config["confirmationPopulation"]["ontologyPacks"]):
        ontology, cues = ontology_pack(pack_index)
        for op_i, operation in enumerate(factors["outerOperations"]):
            for order_i, focus_order in enumerate(factors["literalPositions"]):
                for decoy_i, decoy_kind in enumerate(factors["decoyKinds"]):
                    for orient_i, orientation in enumerate(factors["relationOrientations"]):
                        for sign_i, sign in enumerate(factors["lexicalSigns"]):
                            token = f"{pack_index}|{operation}|{focus_order}|{decoy_kind}|{orientation}|{sign}"
                            count = config["confirmationPopulation"]["entityCounts"][(pack_index + op_i + order_i + decoy_i + orient_i + sign_i) % 3]
                            punctuation = config["confirmationPopulation"]["punctuationRealizations"][(pack_index + op_i * 2 + order_i + decoy_i + orient_i + sign_i) % 3]
                            entities = make_entities(pack_index, count, token, ontology)
                            unit_type, hub_type = ontology["entity_types"]
                            units = [row["id"] for row in entities if row["entity_type"] == unit_type]
                            hubs = [row["id"] for row in entities if row["entity_type"] == hub_type]
                            relation_index = (pack_index + op_i + decoy_i + orient_i) % 2
                            relation = ontology["relations"][relation_index]
                            arguments = [units[0], units[1]] if relation_index == 0 else [hubs[0], units[0]]
                            reversed_arguments = relation_index == 0 and (pack_index + order_i + decoy_i + sign_i) % 2 == 1
                            if reversed_arguments:
                                arguments = list(reversed(arguments))
                            focus = render_literal(relation, arguments, sign, orientation)
                            if decoy_kind == "exact_opposite":
                                decoy = render_literal(relation, arguments, "negative" if sign == "positive" else "positive", orientation)
                            elif decoy_kind == "different_grounded_atom":
                                other = ontology["relations"][1 - relation_index]
                                other_args = [hubs[0], units[0]] if relation_index == 0 else [units[0], units[1]]
                                decoy = render_literal(other, other_args, sign, orientation)
                            else:
                                decoy = f"a {PACK_NAMES[pack_index]} maintenance note lists spare washers"
                            row = {
                                "id": opaque_id("v40", token),
                                "schema_version": 40,
                                "split": "independent_confirmation",
                                "ontology_pack": PACK_NAMES[pack_index],
                                "agent_input": {
                                    "entities": entities,
                                    "predicate_ontology": ontology,
                                    "operator_ontology": operator_ontology(cues),
                                    "evidence_text": render_evidence(focus, decoy, cues[operation], focus_order, punctuation),
                                },
                                "target": {
                                    "parse": {"predicate": relation["id"], "arguments": arguments, "lexical_sign": sign, "outer_operation": operation},
                                    "truth_status": compile_reference_truth(sign, operation),
                                },
                                "oracle_metadata": {
                                    "operation": operation, "sign": sign, "focus_order": focus_order,
                                    "decoy_kind": decoy_kind, "orientation": orientation, "punctuation": punctuation,
                                    "entity_count": count, "argument_reversal": reversed_arguments,
                                    "focus_text": focus, "decoy_text": decoy, "cue": cues[operation],
                                },
                            }
                            rows.append(row)
    expected = config["confirmationPopulation"]["ontologyPacks"] * config["confirmationPopulation"]["coreRecordsPerPack"]
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError("V40 core population size or IDs are invalid")
    return sorted(rows, key=lambda row: row["id"])


def safety_case(base: dict[str, Any], condition: str, ordinal: int) -> dict[str, Any]:
    row = copy.deepcopy(base)
    metadata = row.pop("oracle_metadata")
    row.pop("target")
    row["id"] = opaque_id("v40-safety", f"{base['id']}|{condition}|{ordinal}")
    row["split"] = "independent_safety"
    if condition == "mixed_separators":
        row["agent_input"]["evidence_text"] = f"Focal report: {metadata['focus_text']}. Operation cue: {metadata['cue']}; Context only: {metadata['decoy_text']}."
        statuses = ["abstain"]
    elif condition == "two_complete_focus_fields":
        row["agent_input"]["evidence_text"] = f"Focal report: {metadata['focus_text']}; Focal report: {metadata['decoy_text']}; Operation cue: {metadata['cue']}; Context only: duplicated."
        statuses = ["ambiguous", "abstain"]
    elif condition == "unknown_predicate_lexeme":
        row["agent_input"]["evidence_text"] = row["agent_input"]["evidence_text"].replace(metadata["focus_text"], "quorim shadows velan", 1)
        statuses = ["abstain"]
    elif condition == "unknown_operator_cue":
        row["agent_input"]["evidence_text"] = row["agent_input"]["evidence_text"].replace(metadata["cue"], "unregistered disposition", 1)
        statuses = ["abstain"]
    elif condition == "duplicate_operation_fields":
        separator = SEPARATORS[metadata["punctuation"]]
        row["agent_input"]["evidence_text"] = row["agent_input"]["evidence_text"].replace(
            f"Operation cue: {metadata['cue']}{separator}",
            f"Operation cue: {metadata['cue']}{separator}Operation cue: {metadata['cue']}{separator}", 1,
        )
        statuses = ["abstain"]
    else:
        raise ValueError(condition)
    row["expected"] = {"condition": condition, "statuses": statuses}
    row["oracle_metadata"] = {"source_id": base["id"], "condition": condition}
    return row


def build_populations(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    core = build_core(config)
    safety = []
    conditions = config["safetyPopulation"]["conditions"]
    for pack_index, pack in enumerate(PACK_NAMES):
        pack_rows = [row for row in core if row["ontology_pack"] == pack]
        for condition_index, condition in enumerate(conditions):
            for ordinal in range(config["safetyPopulation"]["recordsPerConditionPerPack"]):
                base = pack_rows[condition_index * 2 + ordinal]
                safety.append(safety_case(base, condition, ordinal))
    if len(safety) != config["safetyPopulation"]["totalRecords"]:
        raise ValueError("V40 safety population size is invalid")
    return {"independent_confirmation": core, "independent_safety": sorted(safety, key=lambda row: row["id"])}


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v40-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_confirmation"]:
        raise RuntimeError("V40 implementation lock does not authorize construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V40 locked implementation changed: {path}")
    output = PROJECT_ROOT / "data/v40-independent-compiler-confirmation"
    if output.exists():
        raise RuntimeError("V40 confirmation corpus already exists")
    populations = build_populations(lock["config_payload"])
    for name, rows in populations.items():
        if corpus_hash(rows) != lock["expected_corpus_sha256"][name]:
            raise RuntimeError(f"V40 {name} differs from implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in populations.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        artifacts[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(rows), "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 40,
        "experiment": lock["config_payload"]["experiment"],
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "counts": {
            "ontology_packs": dict(Counter(row["ontology_pack"] for row in populations["independent_confirmation"])),
            "safety_conditions": dict(Counter(row["expected"]["condition"] for row in populations["independent_safety"])),
        },
        "data_access": {"confirmation_scoring_runs": 0, "model_forward_passes": 0, "v32_evaluation_records_read": 0, "v28_runs": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
