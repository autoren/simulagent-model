#!/usr/bin/env python3
"""Construct the locked V39 development, held-out, and safety populations."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import canonical_json, opaque_id, sha256_text
from v32_language import compile_truth
from generate_v38_focus_parser import make_entities, relation_text
from v38_focus_parser import NON_STATE_DECOYS, ontology_with_lexical_forms
from v39_compiler import OPERATOR_CUES, declared_operator_ontology, render_declared_evidence


def _base_records(config: dict[str, Any], v32: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grammar = config["supportedGrammar"]
    populations = {"compiler_development": [], "supported_evaluation": []}
    seed = 3911
    for op_i, operation in enumerate(grammar["outerOperations"]):
        for cue_i, cue in enumerate(OPERATOR_CUES[operation]):
            for order_i, focus_order in enumerate(grammar["literalPositions"]):
                for decoy_i, decoy_kind in enumerate(grammar["decoyKinds"]):
                    for orient_i, orientation in enumerate(grammar["relationOrientations"]):
                        for sign_i, sign in enumerate(grammar["lexicalSigns"]):
                            for punct_i, punctuation in enumerate(grammar["punctuationRealizations"]):
                                split = "compiler_development" if (cue_i + punct_i) % 2 == 0 else "supported_evaluation"
                                token = f"{operation}|{cue_i}|{focus_order}|{decoy_kind}|{orientation}|{sign}|{punctuation}"
                                entity_count = grammar["entityCounts"][(op_i + order_i + decoy_i + orient_i + sign_i + cue_i + punct_i) % 3]
                                entities = make_entities(entity_count, f"v39|{token}", seed)
                                units = [row["id"] for row in entities if row["entity_type"] == "unit"]
                                hubs = [row["id"] for row in entities if row["entity_type"] == "hub"]
                                predicate = "linked" if (op_i + decoy_i + punct_i) % 2 == 0 else "feeds"
                                arguments = [units[0], units[1]] if predicate == "linked" else [hubs[0], units[0]]
                                reversed_arguments = predicate == "linked" and (order_i + decoy_i + cue_i) % 2 == 1
                                if reversed_arguments:
                                    arguments = list(reversed(arguments))
                                focus_text = relation_text(v32, predicate, arguments, sign, orientation)
                                if decoy_kind == "exact_opposite":
                                    opposite = "negative" if sign == "positive" else "positive"
                                    decoy = relation_text(v32, predicate, arguments, opposite, orientation)
                                elif decoy_kind == "different_grounded_atom":
                                    if predicate == "linked":
                                        decoy = relation_text(v32, "feeds", [hubs[0], units[0]], sign, orientation)
                                    else:
                                        decoy = relation_text(v32, "linked", [units[0], units[1]], sign, orientation)
                                else:
                                    decoy = NON_STATE_DECOYS[focus_order]
                                evidence = render_declared_evidence(focus_text, decoy, cue, focus_order, punctuation)
                                cell = f"{operation}|{focus_order}|{decoy_kind}|{orientation}|{sign}"
                                row = {
                                    "id": opaque_id("v39", f"{split}|{token}"),
                                    "schema_version": 39,
                                    "split": split,
                                    "agent_input": {
                                        "entities": entities,
                                        "predicate_ontology": ontology_with_lexical_forms(v32),
                                        "operator_ontology": declared_operator_ontology(),
                                        "evidence_text": evidence,
                                    },
                                    "target": {
                                        "parse": {
                                            "predicate": predicate,
                                            "arguments": arguments,
                                            "lexical_sign": sign,
                                            "outer_operation": operation,
                                        },
                                        "truth_status": compile_truth(sign, operation, v32),
                                    },
                                    "oracle_metadata": {
                                        "cue_index": cue_i,
                                        "focus_order": focus_order,
                                        "decoy_kind": decoy_kind,
                                        "orientation": orientation,
                                        "punctuation": punctuation,
                                        "entity_count": entity_count,
                                        "argument_reversal": reversed_arguments,
                                        "composition_cell": cell,
                                        "focus_text": focus_text,
                                        "decoy_text": decoy,
                                        "cue": cue,
                                    },
                                }
                                populations[split].append(row)
    for name, rows in populations.items():
        if len(rows) != 360 or len({row["id"] for row in rows}) != 360:
            raise ValueError(f"V39 {name} population is not exactly 360 records")
        populations[name] = sorted(rows, key=lambda row: row["id"])
    return populations


def _mutated_challenge(base: dict[str, Any], kind: str, index: int) -> dict[str, Any]:
    row = copy.deepcopy(base)
    metadata = row.pop("oracle_metadata")
    row.pop("target")
    row["id"] = opaque_id("v39-challenge", f"{kind}|{base['id']}|{index}")
    row["split"] = "compiler_safety"
    evidence = row["agent_input"]["evidence_text"]
    if kind == "malformed_declared_grammar":
        evidence = evidence.replace("Operation cue: ", "Operator clue: ", 1)
        expected = ["abstain"]
    elif kind == "two_equally_marked_focus_literals":
        evidence = f"Focal report: {metadata['focus_text']}; Focal report: {metadata['decoy_text']}; Operation cue: {metadata['cue']}; Context only: duplicate focus fields."
        expected = ["ambiguous", "abstain"]
    elif kind == "unknown_predicate_lexeme":
        evidence = evidence.replace(metadata["focus_text"], "nexa glimmers beside pavo", 1)
        expected = ["abstain"]
    elif kind == "unknown_operator_cue":
        evidence = evidence.replace(metadata["cue"], "speculation", 1)
        expected = ["abstain"]
    else:
        raise ValueError(kind)
    row["agent_input"]["evidence_text"] = evidence
    row["expected"] = {"statuses": expected, "challenge_kind": kind}
    row["oracle_metadata"] = {"source_id": base["id"], "challenge_kind": kind}
    return row


def _novel_paraphrase(base: dict[str, Any], index: int) -> dict[str, Any]:
    row = copy.deepcopy(base)
    metadata = row.pop("oracle_metadata")
    target = row.pop("target")
    operation = target["parse"]["outer_operation"]
    templates = {
        "assert": "Inspection supports {focus}, while {decoy} is merely contextual.",
        "deny": "Inspection rejects {focus}; the mention of {decoy} is incidental.",
        "double_deny": "It would be wrong to reject {focus}, irrespective of {decoy}.",
        "contrast_select": "Prefer {focus} over the contextual alternative {decoy}.",
        "unresolved": "Inspection leaves {focus} undecided; {decoy} settles nothing.",
    }
    row["id"] = opaque_id("v39-paraphrase", f"{base['id']}|{index}")
    row["split"] = "novel_paraphrase_diagnostic"
    row["agent_input"]["evidence_text"] = templates[operation].format(focus=metadata["focus_text"], decoy=metadata["decoy_text"])
    row["expected"] = {"scope": "unsupported_non_gating", "reference_parse": target["parse"]}
    row["oracle_metadata"] = {"source_id": base["id"], "operation": operation}
    return row


def build_populations(config: dict[str, Any], v32: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    populations = _base_records(config, v32)
    evaluation = populations["supported_evaluation"]
    challenge_kinds = (
        "malformed_declared_grammar", "two_equally_marked_focus_literals",
        "unknown_predicate_lexeme", "unknown_operator_cue",
    )
    safety = []
    for kind_i, kind in enumerate(challenge_kinds):
        selected = evaluation[kind_i::6][:60]
        if len(selected) != 60:
            raise ValueError("V39 safety sampling failed")
        safety.extend(_mutated_challenge(row, kind, index) for index, row in enumerate(selected))
    populations["compiler_safety"] = sorted(safety, key=lambda row: row["id"])
    populations["novel_paraphrase_diagnostic"] = sorted(
        [_novel_paraphrase(row, index) for index, row in enumerate(evaluation[:50])],
        key=lambda row: row["id"],
    )
    return populations


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v39-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_evaluation"]:
        raise RuntimeError("V39 implementation lock does not authorize construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V39 locked implementation changed: {path}")
    output = PROJECT_ROOT / "data/v39-declared-language-compiler"
    if output.exists():
        raise RuntimeError("V39 corpus already exists")
    populations = build_populations(lock["config_payload"], lock["v32_config_payload"])
    for name, rows in populations.items():
        if corpus_hash(rows) != lock["expected_corpus_sha256"][name]:
            raise RuntimeError(f"V39 {name} differs from implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in populations.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        artifacts[name] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(rows), "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 39,
        "experiment": lock["config_payload"]["experiment"],
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "counts": {
            name: dict(Counter(row.get("oracle_metadata", {}).get("challenge_kind", "supported") for row in rows))
            for name, rows in populations.items()
        },
        "data_access": {"model_forward_passes": 0, "fit_runs": 0, "evaluation_scoring_runs": 0, "v32_evaluation_records_read": 0, "v28_runs": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
