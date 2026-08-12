"""Generate the development-only V18 episodic schema-induction benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from v18_schema import (
    BOOLEAN_VALUES,
    ProgramHypothesis,
    all_assignments,
    enumerate_program_hypotheses,
    execute_query,
    greedy_distinguishing_support,
    program_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHALLOW_FAMILIES = frozenset({"var", "not", "and", "or", "xor", "and_not"})
STRUCTURAL_FAMILIES = frozenset({"or_of_and", "and_of_or", "xor_of_and", "majority"})
DEEP_FAMILIES = frozenset({"mux", "deep_xor"})

LEXICONS: dict[str, list[dict[str, str]]] = {
    "industrial": [
        {"id": "relay_charged", "label": "relay charge", "active": "the relay carries charge", "inactive": "the relay is discharged"},
        {"id": "valve_open", "label": "valve position", "active": "the valve stands open", "inactive": "the valve is shut"},
        {"id": "rotor_stable", "label": "rotor stability", "active": "the rotor turns steadily", "inactive": "the rotor wobbles"},
        {"id": "coolant_flowing", "label": "coolant flow", "active": "coolant is circulating", "inactive": "coolant flow has stopped"},
    ],
    "astronomical": [
        {"id": "lens_aligned", "label": "lens alignment", "active": "the lens is aligned", "inactive": "the lens is out of alignment"},
        {"id": "dome_open", "label": "dome position", "active": "the dome aperture is open", "inactive": "the dome aperture is closed"},
        {"id": "tracker_locked", "label": "tracker lock", "active": "the tracker holds its lock", "inactive": "the tracker has lost lock"},
        {"id": "clock_synced", "label": "clock synchronization", "active": "the clock is synchronized", "inactive": "the clock is unsynchronized"},
    ],
    "archival": [
        {"id": "seal_intact", "label": "document seal", "active": "the document seal is intact", "inactive": "the document seal is broken"},
        {"id": "index_current", "label": "index currency", "active": "the index is current", "inactive": "the index is outdated"},
        {"id": "drawer_unlocked", "label": "drawer lock", "active": "the drawer is unlocked", "inactive": "the drawer is locked"},
        {"id": "lamp_lit", "label": "reading lamp", "active": "the reading lamp is lit", "inactive": "the reading lamp is dark"},
    ],
    "maritime": [
        {"id": "bilge_clear", "label": "bilge condition", "active": "the bilge is clear", "inactive": "the bilge is flooded"},
        {"id": "anchor_raised", "label": "anchor position", "active": "the anchor is raised", "inactive": "the anchor is lowered"},
        {"id": "compass_true", "label": "compass reading", "active": "the compass reads true", "inactive": "the compass is deflected"},
        {"id": "hatch_dogged", "label": "hatch fastening", "active": "the hatch is dogged", "inactive": "the hatch is unsecured"},
    ],
}

TRAIN_FAMILY_PAIRS = (
    ("and", "or"),
    ("xor", "and_not"),
    ("and_not", "xor"),
    ("or", "and"),
    ("xor", "or"),
    ("and_not", "and"),
    ("var", "not"),
    ("not", "var"),
)
CALIBRATION_FAMILY_PAIRS = (
    ("var", "and"),
    ("not", "xor"),
    ("or", "var"),
    ("and", "not"),
)
RECOMBINATION_FAMILY_PAIRS = (
    ("and", "xor"),
    ("or", "and_not"),
    ("xor", "and"),
    ("and_not", "or"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_candidate(
    hypotheses: Sequence[ProgramHypothesis],
    predicate: Callable[[ProgramHypothesis], bool],
    excluded_signatures: set[tuple[str, ...]],
    selector: str,
) -> ProgramHypothesis:
    candidates = [
        value for value in hypotheses
        if value.signature not in excluded_signatures and predicate(value)
    ]
    candidates.sort(key=lambda value: sha256_text(f"{selector}|{'|'.join(value.signature)}"))
    for candidate in candidates:
        support = greedy_distinguishing_support(candidate.program, hypotheses)
        if len(support) < 16:
            return candidate
    raise ValueError(f"No eligible V18 program for selector {selector}")


def render_observation(
    ontology: Sequence[dict[str, str]],
    assignment: dict[str, bool],
    unknown: str | None,
    style: int,
) -> str:
    sentences = []
    for index, concept in enumerate(ontology):
        identifier = concept["id"]
        if identifier == unknown:
            sentence = (
                f"No current reading establishes either state of {concept['label']}."
            )
        else:
            clause = concept["active"] if assignment[identifier] else concept["inactive"]
            if style % 3 == 0:
                sentence = f"A current inspection confirms that {clause}."
            elif style % 3 == 1:
                sentence = f"The latest operator log records that {clause}."
            else:
                opposite = concept["inactive"] if assignment[identifier] else concept["active"]
                sentence = f"A signed present-state check rules out the claim that {opposite}."
        sentences.append(sentence)
    rotation = style % len(sentences)
    return " ".join(sentences[rotation:] + sentences[:rotation])


def allowed_values_for(
    determinant_ids: Sequence[str],
    assignment: dict[str, bool],
    unknown: str | None,
) -> list[dict[str, Any]]:
    values = []
    for identifier in determinant_ids:
        allowed = list(BOOLEAN_VALUES) if identifier == unknown else [
            "active" if assignment[identifier] else "inactive"
        ]
        values.append({"determinant_id": identifier, "allowed_values": allowed})
    return values


def build_episode(
    split: str,
    axis: str,
    lexicon_name: str,
    ordinal: int,
    target: ProgramHypothesis,
    hypotheses: Sequence[ProgramHypothesis],
) -> dict[str, Any]:
    ontology = LEXICONS[lexicon_name]
    determinant_ids = tuple(value["id"] for value in ontology)
    support = greedy_distinguishing_support(target.program, hypotheses)
    episode_key = sha256_text(f"{split}|{axis}|{lexicon_name}|{ordinal}|{target.signature}")[:20]
    mechanic = f"v18_{axis}_{ordinal:02d}_{episode_key[:8]}"
    action = f"run diagnostic action {episode_key[:10]}"

    agent_support = []
    support_grounding = []
    support_assignments = set()
    for index, trace in enumerate(support):
        trace_id = f"{episode_key}:support:{index:02d}"
        assignment = trace["assignment"]
        support_assignments.add(tuple(assignment[value] for value in determinant_ids))
        agent_support.append({
            "trace_id": trace_id,
            "observation": render_observation(ontology, assignment, None, index),
            "observed_transition_code": trace["transition_code"],
        })
        support_grounding.append({"trace_id": trace_id, "assignment": assignment})

    agent_queries = []
    query_targets = []
    query_index = 0
    for assignment in all_assignments(determinant_ids):
        assignment_key = tuple(assignment[value] for value in determinant_ids)
        if assignment_key in support_assignments:
            continue
        query_id = f"{episode_key}:query:{query_index:03d}"
        allowed = allowed_values_for(determinant_ids, assignment, None)
        answer = execute_query(target.program, allowed)
        agent_queries.append({
            "query_id": query_id,
            "observation": render_observation(ontology, assignment, None, query_index + 1),
        })
        query_targets.append({
            "query_id": query_id,
            "allowed_values": allowed,
            "unknown_determinant": None,
            "unknown_effect": "fully_observed",
            **answer,
        })
        query_index += 1

    for unknown in determinant_ids:
        other_ids = tuple(value for value in determinant_ids if value != unknown)
        for partial in all_assignments(other_ids):
            assignment = {**partial, unknown: False}
            query_id = f"{episode_key}:query:{query_index:03d}"
            allowed = allowed_values_for(determinant_ids, assignment, unknown)
            answer = execute_query(target.program, allowed)
            effect = "outcome_invariant" if answer["identifiable"] else "outcome_sensitive"
            agent_queries.append({
                "query_id": query_id,
                "observation": render_observation(ontology, assignment, unknown, query_index + 2),
            })
            query_targets.append({
                "query_id": query_id,
                "allowed_values": allowed,
                "unknown_determinant": unknown,
                "unknown_effect": effect,
                **answer,
            })
            query_index += 1

    return {
        "id": f"v18:{episode_key}",
        "schema_version": 18,
        "split": split,
        "generalization_axis": axis,
        "mechanic": mechanic,
        "lexicon_family": lexicon_name,
        "program_metadata": {
            "component_families": list(target.component_families),
            "maximum_depth": target.max_depth,
            "support_traces": len(support),
            "visible_outcomes": sorted(set(target.signature)),
        },
        "agent_input": {
            "task": "induce_transition_schema_and_answer_queries",
            "candidate_action": action,
            "determinant_ontology": [
                {"id": value["id"], "label": value["label"], "type": "boolean"}
                for value in ontology
            ],
            "dsl_contract": {
                "value_type": "boolean",
                "operators": ["not", "and", "or", "xor"],
                "outcome_bits": len(target.program["output_bits"]),
            },
            "support_traces": agent_support,
            "queries": agent_queries,
            "output_instruction": (
                "Infer an executable transition rule from the support traces. For each query, "
                "return every possible visible transition code and whether the outcome is identifiable."
            ),
        },
        "oracle_grounding": {
            "condition": "schema_induction_with_oracle_language_grounding",
            "support": support_grounding,
            "queries": query_targets,
        },
        "target": {
            "executable_schema": target.program,
            "behavioral_signature": list(program_signature(target.program)),
            "relevant_determinants": list(target.relevant_determinants),
        },
        "source": {
            "kind": "procedural_v18_development_simulator",
            "v17_records_read": 0,
            "v17_model_results_read": 0,
        },
    }


def candidate_with_signature(
    hypotheses: Sequence[ProgramHypothesis], signature: tuple[str, ...]
) -> ProgramHypothesis:
    for value in hypotheses:
        if value.signature == signature:
            return value
    raise ValueError("Lexicon-specific grammar omitted a selected behavioral signature")


def generate(config: dict[str, Any]) -> list[dict[str, Any]]:
    known_lexicons = config["knownLexicons"]
    held_out_lexicon = config["heldOutLexicon"]
    hypotheses_by_lexicon = {
        name: enumerate_program_hypotheses(tuple(value["id"] for value in LEXICONS[name]), 2)
        for name in [*known_lexicons, held_out_lexicon]
    }
    used: set[tuple[str, ...]] = set()
    records: list[dict[str, Any]] = []
    training_signatures: list[tuple[str, ...]] = []

    for index in range(config["trainingEpisodes"]):
        lexicon = known_lexicons[index % len(known_lexicons)]
        hypotheses = hypotheses_by_lexicon[lexicon]
        family_pair = TRAIN_FAMILY_PAIRS[index % len(TRAIN_FAMILY_PAIRS)]
        target = select_candidate(
            hypotheses,
            lambda value, pair=family_pair: value.component_families == pair,
            used,
            f"train:{index}",
        )
        used.add(target.signature)
        training_signatures.append(target.signature)
        records.append(build_episode("train", "training_components", lexicon, index, target, hypotheses))

    for index in range(config["calibrationEpisodes"]):
        lexicon = known_lexicons[index % len(known_lexicons)]
        hypotheses = hypotheses_by_lexicon[lexicon]
        family_pair = CALIBRATION_FAMILY_PAIRS[index % len(CALIBRATION_FAMILY_PAIRS)]
        target = select_candidate(
            hypotheses,
            lambda value, pair=family_pair: value.component_families == pair,
            used,
            f"calibration:{index}",
        )
        used.add(target.signature)
        records.append(build_episode("calibration", "known_component_calibration", lexicon, index, target, hypotheses))

    count = config["episodesPerDevelopmentAxis"]
    for index in range(count):
        lexicon = known_lexicons[index % len(known_lexicons)]
        hypotheses = hypotheses_by_lexicon[lexicon]
        pair = RECOMBINATION_FAMILY_PAIRS[index % len(RECOMBINATION_FAMILY_PAIRS)]
        target = select_candidate(
            hypotheses,
            lambda value, pair=pair: value.component_families == pair,
            used,
            f"recombination:{index}",
        )
        used.add(target.signature)
        records.append(build_episode("development", "known_primitive_recombination", lexicon, index, target, hypotheses))

    for index in range(count):
        lexicon = known_lexicons[index % len(known_lexicons)]
        hypotheses = hypotheses_by_lexicon[lexicon]
        target = select_candidate(
            hypotheses,
            lambda value: (
                bool(set(value.component_families) & STRUCTURAL_FAMILIES)
                and not bool(set(value.component_families) & DEEP_FAMILIES)
                and value.max_depth == 2
                and len(value.relevant_determinants) == 4
                and len(set(value.signature)) >= 3
            ),
            used,
            f"structure:{index}",
        )
        used.add(target.signature)
        records.append(build_episode("development", "structural_composition", lexicon, index, target, hypotheses))

    maritime_hypotheses = hypotheses_by_lexicon[held_out_lexicon]
    for index in range(count):
        signature = training_signatures[index]
        target = candidate_with_signature(maritime_hypotheses, signature)
        records.append(build_episode("development", "determinant_vocabulary", held_out_lexicon, index, target, maritime_hypotheses))

    for index in range(count):
        lexicon = known_lexicons[index % len(known_lexicons)]
        hypotheses = hypotheses_by_lexicon[lexicon]
        target = select_candidate(
            hypotheses,
            lambda value: (
                bool(set(value.component_families) & DEEP_FAMILIES)
                and value.max_depth >= 3
                and len(value.relevant_determinants) == 4
                and len(set(value.signature)) >= 3
            ),
            used,
            f"depth:{index}",
        )
        used.add(target.signature)
        records.append(build_episode("development", "composition_depth", lexicon, index, target, hypotheses))

    for index in range(count):
        lexicon = known_lexicons[index % len(known_lexicons)]
        hypotheses = hypotheses_by_lexicon[lexicon]
        target = select_candidate(
            hypotheses,
            lambda value: (
                set(value.component_families).issubset(SHALLOW_FAMILIES)
                and
                len(value.relevant_determinants) <= 3
                and 1 < len(set(value.signature)) < len(value.signature)
            ),
            used,
            f"invariance:{index}",
        )
        used.add(target.signature)
        records.append(build_episode("development", "outcome_invariance", lexicon, index, target, hypotheses))

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v18.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config_text = config_path.read_text()
    config = json.loads(config_text)
    records = generate(config)

    output_dir = (PROJECT_ROOT / config["outputDir"]).resolve()
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes: dict[str, str] = {}
    dataset_parts = []
    for split in ("train", "calibration", "development"):
        values = [value for value in records if value["split"] == split]
        content = "".join(canonical_json(value) + "\n" for value in values)
        path = records_dir / f"{split}.jsonl"
        path.write_text(content)
        relative = f"records/{split}.jsonl"
        artifact_hashes[relative] = sha256_text(content)
        dataset_parts.append(f"{relative}\n{content}")

    implementation_paths = [
        "python/generate_v18_schema_benchmark.py",
        "python/v18_schema.py",
    ]
    manifest = {
        "schema_version": 18,
        "experiment": "v18_executable_transition_schema_induction_development",
        "config": config,
        "config_sha256": sha256_text(config_text),
        "records": len(records),
        "split_counts": {
            split: sum(value["split"] == split for value in records)
            for split in ("train", "calibration", "development")
        },
        "development_axis_counts": {
            axis: sum(value["generalization_axis"] == axis for value in records)
            for axis in config["developmentAxes"]
        },
        "implementation_sha256": {
            path: sha256_text((PROJECT_ROOT / path).read_text()) for path in implementation_paths
        },
        "artifact_sha256": artifact_hashes,
        "dataset_sha256": sha256_text("".join(dataset_parts)),
        "data_access": {
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "fresh_final_mechanic_created": False,
            "adapter_training_runs": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir.relative_to(PROJECT_ROOT)), **manifest}, indent=2))


if __name__ == "__main__":
    main()
