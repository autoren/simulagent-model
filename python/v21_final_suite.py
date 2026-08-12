"""Locked-design generator and language renderer for the sealed V21 final suite."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Sequence

from generate_v18_schema_benchmark import allowed_values_for
from v18_schema import (
    ProgramHypothesis,
    all_assignments,
    enumerate_program_hypotheses,
    execute_query,
    greedy_distinguishing_support,
    program_signature,
)


DETERMINANT_IDS = ("state_a", "state_b", "state_c", "state_d")

SURFACES = (
    "direct_assertion",
    "present_confirmation",
    "current_observation",
    "explicit_negation",
    "denied_claim",
    "scoped_rejection",
    "contrastive_correction",
    "contrastive_verification",
    "contrastive_resolution",
)

SEMANTIC_OPERATOR = {
    "direct_assertion": "affirmative_gold",
    "present_confirmation": "affirmative_gold",
    "current_observation": "affirmative_gold",
    "explicit_negation": "negated_opposite",
    "denied_claim": "negated_opposite",
    "scoped_rejection": "negated_opposite",
    "contrastive_correction": "contrastive_both",
    "contrastive_verification": "contrastive_both",
    "contrastive_resolution": "contrastive_both",
}

SUPPORTED_CONCEPTS = (
    {
        "id": "generator_stable", "label": "the generator rhythm is stable",
        "active": "the generator rhythm is even",
        "inactive": "the generator output surges unevenly",
    },
    {
        "id": "mirror_seated", "label": "the mirror shard is seated",
        "active": "the mirror shard sits flush in its socket",
        "inactive": "the mirror socket is empty",
    },
    {
        "id": "fork_calibrated", "label": "the carried tuning fork is calibrated",
        "active": "the fork tone matches the reference pitch",
        "inactive": "the fork tone falls away from the reference pitch",
    },
    {
        "id": "hatch_unlocked", "label": "the observatory hatch is unlocked",
        "active": "the observatory hatch stands unlatched",
        "inactive": "the observatory hatch remains latched",
    },
)

NOVEL_LEXICONS: dict[str, tuple[dict[str, str], ...]] = {
    "primitive_one_bit": (
        {"id": "brine_warm", "label": "brine temperature", "active": "the brine runs warm", "inactive": "the brine remains chilled"},
        {"id": "vent_drawing", "label": "vent draft", "active": "the vent draws upward", "inactive": "the vent air is still"},
        {"id": "crystal_clear", "label": "crystal clarity", "active": "the crystal is transparent", "inactive": "the crystal has clouded"},
        {"id": "flask_sealed", "label": "flask seal", "active": "the flask seal holds", "inactive": "the flask seal is leaking"},
    ),
    "composed_one_bit": (
        {"id": "bud_open", "label": "bud state", "active": "the bud has opened", "inactive": "the bud remains closed"},
        {"id": "stem_rigid", "label": "stem rigidity", "active": "the stem stands rigid", "inactive": "the stem bends freely"},
        {"id": "root_damp", "label": "root moisture", "active": "the roots are damp", "inactive": "the roots are dry"},
        {"id": "spore_viable", "label": "spore viability", "active": "the spores remain viable", "inactive": "the spores are inert"},
    ),
    "factorized_two_bit": (
        {"id": "signal_green", "label": "signal aspect", "active": "the signal shows green", "inactive": "the signal shows red"},
        {"id": "points_aligned", "label": "points alignment", "active": "the points align to the branch", "inactive": "the points align to the main line"},
        {"id": "circuit_clear", "label": "track circuit", "active": "the track circuit is clear", "inactive": "the track circuit is occupied"},
        {"id": "brake_released", "label": "brake state", "active": "the brake is released", "inactive": "the brake is applied"},
    ),
    "nested_two_bit": (
        {"id": "ram_extended", "label": "hydraulic ram", "active": "the ram is extended", "inactive": "the ram is retracted"},
        {"id": "bypass_open", "label": "bypass state", "active": "the bypass is open", "inactive": "the bypass is closed"},
        {"id": "reservoir_full", "label": "reservoir level", "active": "the reservoir is full", "inactive": "the reservoir is depleted"},
        {"id": "pump_primed", "label": "pump prime", "active": "the pump is primed", "inactive": "the pump has lost prime"},
    ),
}

ACTIONS = {
    "primitive_one_bit": "take the geothermal sampler reading",
    "composed_one_bit": "run the greenhouse viability check",
    "factorized_two_bit": "request the railway interlock indication",
    "nested_two_bit": "cycle the hydraulic diagnostic",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def injective_on_relevant_projection(hypothesis: ProgramHypothesis) -> bool:
    relevant = tuple(hypothesis.relevant_determinants)
    if not relevant:
        return True
    codes = set()
    for assignment in all_assignments(relevant):
        complete = {identifier: assignment.get(identifier, False) for identifier in DETERMINANT_IDS}
        index = sum(
            int(complete[identifier]) << (len(DETERMINANT_IDS) - position - 1)
            for position, identifier in enumerate(DETERMINANT_IDS)
        )
        codes.add(hypothesis.signature[index])
    return len(codes) == 2 ** len(relevant)


def family_eligible(family: str, hypothesis: ProgramHypothesis) -> bool:
    relevant = len(hypothesis.relevant_determinants)
    injective = injective_on_relevant_projection(hypothesis)
    if family == "primitive_one_bit":
        return relevant == 1 and hypothesis.max_depth <= 1 and injective
    if family == "composed_one_bit":
        return relevant >= 2 and hypothesis.max_depth >= 1 and not injective
    if family == "factorized_two_bit":
        return relevant == 2 and hypothesis.max_depth <= 1 and injective
    if family == "nested_two_bit":
        return relevant >= 3 and hypothesis.max_depth >= 2 and not injective
    raise ValueError(f"Unknown V21 construction family {family}")


def select_targets(
    config: dict[str, Any], seed: str,
    excluded: dict[int, set[tuple[str, ...]]] | None = None,
) -> list[tuple[str, int, ProgramHypothesis]]:
    excluded = excluded or {}
    result = []
    used: dict[int, set[tuple[str, ...]]] = {1: set(), 2: set()}
    for family, specification in config["constructionFamilies"].items():
        output_bits = int(specification["outcomeBits"])
        hypotheses = enumerate_program_hypotheses(DETERMINANT_IDS, output_bits)
        candidates = [
            value for value in hypotheses
            if family_eligible(family, value)
            and value.signature not in excluded.get(output_bits, set())
            and value.signature not in used[output_bits]
        ]
        candidates.sort(key=lambda value: sha256_text(
            f"{seed}|{family}|{'|'.join(value.signature)}"
        ))
        selected = []
        for candidate in candidates:
            if len(greedy_distinguishing_support(candidate.program, hypotheses)) >= 16:
                continue
            selected.append(candidate)
            used[output_bits].add(candidate.signature)
            if len(selected) == specification["episodes"]:
                break
        if len(selected) != specification["episodes"]:
            raise RuntimeError(f"Insufficient eligible programs for {family}")
        result.extend((family, index, value) for index, value in enumerate(selected))
    return result


def build_episode(
    family: str, ordinal: int, target: ProgramHypothesis,
    hypotheses: Sequence[ProgramHypothesis], seed: str,
) -> dict[str, Any]:
    support = greedy_distinguishing_support(target.program, hypotheses)
    key = sha256_text(f"{seed}|{family}|{ordinal}|{'|'.join(target.signature)}")[:20]
    support_rows = []
    oracle_support = []
    support_assignments = set()
    for index, trace in enumerate(support):
        trace_id = f"{key}:support:{index:02d}"
        assignment = trace["assignment"]
        support_assignments.add(tuple(assignment[value] for value in DETERMINANT_IDS))
        support_rows.append({
            "trace_id": trace_id,
            "observation_ref": trace_id,
            "observed_transition_code": trace["transition_code"],
        })
        oracle_support.append({"trace_id": trace_id, "assignment": assignment})
    query_rows = []
    oracle_queries = []
    query_index = 0
    for assignment in all_assignments(DETERMINANT_IDS):
        assignment_key = tuple(assignment[value] for value in DETERMINANT_IDS)
        if assignment_key in support_assignments:
            continue
        query_id = f"{key}:query:{query_index:03d}"
        allowed = allowed_values_for(DETERMINANT_IDS, assignment, None)
        answer = execute_query(target.program, allowed)
        query_rows.append({"query_id": query_id, "observation_ref": query_id})
        oracle_queries.append({
            "query_id": query_id, "allowed_values": allowed,
            "unknown_determinant": None, "unknown_effect": "fully_observed", **answer,
        })
        query_index += 1
    for unknown in DETERMINANT_IDS:
        others = tuple(value for value in DETERMINANT_IDS if value != unknown)
        for partial in all_assignments(others):
            assignment = {**partial, unknown: False}
            query_id = f"{key}:query:{query_index:03d}"
            allowed = allowed_values_for(DETERMINANT_IDS, assignment, unknown)
            answer = execute_query(target.program, allowed)
            query_rows.append({"query_id": query_id, "observation_ref": query_id})
            oracle_queries.append({
                "query_id": query_id, "allowed_values": allowed,
                "unknown_determinant": unknown,
                "unknown_effect": "outcome_invariant" if answer["identifiable"] else "outcome_sensitive",
                **answer,
            })
            query_index += 1
    return {
        "id": f"v21:{key}",
        "schema_version": 21,
        "split": "final",
        "construction_family": family,
        "generalization_axis": family,
        "mechanic": f"v21_{family}_{ordinal:02d}_{key[:8]}",
        "program_metadata": {
            "component_families": list(target.component_families),
            "maximum_depth": target.max_depth,
            "relevant_determinants": len(target.relevant_determinants),
            "support_traces": len(support),
            "outcome_bits": len(target.program["output_bits"]),
            "injective_on_relevant_projection": injective_on_relevant_projection(target),
            "visible_outcomes": sorted(set(target.signature)),
        },
        "agent_input": {
            "task": "induce_transition_schema_and_answer_queries",
            "candidate_action": ACTIONS[family],
            "determinant_ontology": [
                {"id": identifier, "label": f"latent Boolean position {index + 1}", "type": "boolean"}
                for index, identifier in enumerate(DETERMINANT_IDS)
            ],
            "dsl_contract": {
                "value_type": "boolean", "operators": ["not", "and", "or", "xor"],
                "outcome_bits": len(target.program["output_bits"]),
            },
            "support_traces": support_rows,
            "queries": query_rows,
            "output_instruction": (
                "Infer an executable transition rule from the paired language-grounded support "
                "traces and return all possible visible transition codes for every query."
            ),
        },
        "oracle_grounding": {"support": oracle_support, "queries": oracle_queries},
        "target": {
            "executable_schema": target.program,
            "behavioral_signature": list(program_signature(target.program)),
            "relevant_determinants": list(target.relevant_determinants),
        },
        "source": {
            "kind": "sealed_v21_delayed_seed_generator",
            "v17_records_read": 0, "v17_model_results_read": 0,
        },
    }


def grounding_code(allowed_values: Sequence[dict[str, Any]], item_kind: str) -> int:
    by_id = {value["determinant_id"]: value["allowed_values"] for value in allowed_values}
    code = 0 if item_kind == "support" else 97
    for index, identifier in enumerate(DETERMINANT_IDS):
        values = by_id[identifier]
        digit = 2 if len(values) == 2 else 1 if values == ["active"] else 0
        code = code * 3 + digit + index
    return code


def render_current(concept: dict[str, str], value: str, surface: str) -> str:
    actual = concept[value]
    opposite = concept["inactive" if value == "active" else "active"]
    templates = {
        "direct_assertion": f"A current verification establishes that {actual}.",
        "present_confirmation": f"The latest inspection confirms that {actual}.",
        "current_observation": f"The present reading shows that {actual}.",
        "explicit_negation": f"A current verification establishes that it is not true that {opposite}.",
        "denied_claim": f"The current auditor denies the claim that {opposite}.",
        "scoped_rejection": f"The report rejected by the current auditor is the one claiming that {opposite}.",
        "contrastive_correction": f"The current correction says that {opposite} is not the case; instead, {actual}.",
        "contrastive_verification": f"The present check rules out that {opposite} and confirms that {actual}.",
        "contrastive_resolution": f"Of the two possibilities, current evidence excludes that {opposite} and supports that {actual}.",
    }
    return templates[surface]


def render_unresolved(concept: dict[str, str], mode: str, code: int) -> str:
    if mode == "stale":
        earlier = concept["active" if code % 2 else "inactive"]
        return f"An archived reading said that {earlier}, but no present reading is available."
    if mode == "conflicting":
        return (
            f"Two equally current readings conflict: one says that {concept['active']}; "
            f"the other says that {concept['inactive']}."
        )
    return (
        f"No current evidence establishes either that {concept['active']} or that {concept['inactive']}."
    )


def build_scene(
    episode: dict[str, Any], view: str, item_kind: str, item_id: str,
    allowed_values: Sequence[dict[str, Any]], observed_transition_code: str | None,
) -> dict[str, Any]:
    family = episode["construction_family"]
    concepts = SUPPORTED_CONCEPTS if view == "supported" else NOVEL_LEXICONS[family]
    code = grounding_code(allowed_values, item_kind)
    family_index = list(NOVEL_LEXICONS).index(family)
    surface = SURFACES[(code + family_index) % len(SURFACES)]
    unresolved_mode = ("unknown", "stale", "conflicting")[(code + family_index) % 3]
    by_id = {value["determinant_id"]: value["allowed_values"] for value in allowed_values}
    rendered = []
    for position, (latent_id, concept) in enumerate(zip(DETERMINANT_IDS, concepts, strict=True)):
        allowed = by_id[latent_id]
        if len(allowed) == 2:
            text = render_unresolved(concept, unresolved_mode, code + position)
            temporal = {
                "unknown": "UNKNOWN_CURRENT", "stale": "STALE_ONLY",
                "conflicting": "CONFLICTING_CURRENT",
            }[unresolved_mode]
            current = None
            relations = ["UNKNOWN", "UNKNOWN"]
        else:
            current = allowed[0]
            text = render_current(concept, current, surface)
            temporal = "CURRENT"
            relations = ["ENTAILED", "CONTRADICTED"] if current == "active" else ["CONTRADICTED", "ENTAILED"]
        rendered.append({
            "position": position, "latent_id": latent_id, "concept": concept,
            "text": text, "temporal": temporal, "current": current,
            "relations": relations, "allowed_values": list(allowed),
            "order": sha256_text(f"{family}|{view}|{code}|{surface}|{position}")[:16],
        })
    rendered.sort(key=lambda value: value["order"])
    observation = ""
    evidence_units = []
    span_by_position = {}
    for value in rendered:
        if observation:
            observation += "\n"
        start = len(observation)
        observation += value["text"]
        evidence = {"start": start, "end": len(observation), "text": value["text"]}
        evidence_units.append(evidence)
        span_by_position[value["position"]] = evidence
    target = []
    for position, (latent_id, concept) in enumerate(zip(DETERMINANT_IDS, concepts, strict=True)):
        value = next(item for item in rendered if item["position"] == position)
        target.append({
            "determinant_id": concept["id"], "latent_determinant_id": latent_id,
            "temporal_status": value["temporal"], "current_value": value["current"],
            "hypothesis_relations": value["relations"],
            "allowed_values": value["allowed_values"], "evidence_span": span_by_position[position],
        })
    scene_key = sha256_text(f"{episode['id']}|{view}|{item_kind}|{item_id}")[:24]
    result = {
        "id": f"v21s:{scene_key}", "schema_version": 21, "split": "final",
        "construction_family": family, "generalization_axis": family,
        "episode_id": episode["id"], "view": view,
        "view_role": "primary" if view == "supported" else "non_gating_diagnostic",
        "item_kind": item_kind, "source_item_id": item_id,
        "surface_family": surface, "semantic_operator_family": SEMANTIC_OPERATOR[surface],
        "unresolved_mode": unresolved_mode if any(len(value["allowed_values"]) == 2 for value in allowed_values) else None,
        "agent_input": {
            "task": "ground_current_state_polarity",
            "candidate_action": ACTIONS[family],
            "transition_determinants": [{"id": value["id"], "label": value["label"]} for value in concepts],
            "state_hypotheses": [
                {"determinant_id": value["id"], "statements": [value["active"], value["inactive"]]}
                for value in concepts
            ],
            "observation": observation,
            "output_instruction": (
                "Match each determinant to one evidence unit, classify its temporal status, "
                "and compare reliable current evidence with both supplied state hypotheses."
            ),
        },
        "evidence_units": evidence_units,
        "target": {"determinant_grounding": target},
        "source": {"episode_id": episode["id"], "v17_records_read": 0, "v17_model_results_read": 0},
    }
    if observed_transition_code is not None:
        result["observed_transition_code"] = observed_transition_code
    return result


def build_scenes(episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    scenes = []
    for episode in episodes:
        observed = {
            value["trace_id"]: value["observed_transition_code"]
            for value in episode["agent_input"]["support_traces"]
        }
        for view in ("supported", "novel_ontology"):
            for grounding in episode["oracle_grounding"]["support"]:
                allowed = [
                    {
                        "determinant_id": identifier,
                        "allowed_values": ["active" if grounding["assignment"][identifier] else "inactive"],
                    }
                    for identifier in DETERMINANT_IDS
                ]
                scenes.append(build_scene(
                    episode, view, "support", grounding["trace_id"], allowed,
                    observed[grounding["trace_id"]],
                ))
            for query in episode["oracle_grounding"]["queries"]:
                scenes.append(build_scene(
                    episode, view, "query", query["query_id"], query["allowed_values"], None,
                ))
    return scenes


def generate_suite(
    config: dict[str, Any], seed: str,
    excluded: dict[int, set[tuple[str, ...]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = select_targets(config, seed, excluded)
    hypotheses = {
        bits: enumerate_program_hypotheses(DETERMINANT_IDS, bits) for bits in (1, 2)
    }
    episodes = [
        build_episode(family, ordinal, target, hypotheses[len(target.program["output_bits"])], seed)
        for family, ordinal, target in targets
    ]
    return episodes, build_scenes(episodes)


def structural_summary(
    episodes: Sequence[dict[str, Any]], scenes: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "episodes": len(episodes),
        "family_counts": dict(Counter(value["construction_family"] for value in episodes)),
        "outcome_bit_counts": dict(Counter(str(value["program_metadata"]["outcome_bits"]) for value in episodes)),
        "injectivity_counts": dict(Counter(
            "injective" if value["program_metadata"]["injective_on_relevant_projection"] else "non_injective"
            for value in episodes
        )),
        "scenes": len(scenes),
        "view_counts": dict(Counter(value["view"] for value in scenes)),
        "surface_counts": dict(Counter(value["surface_family"] for value in scenes)),
        "semantic_operator_counts": dict(Counter(value["semantic_operator_family"] for value in scenes)),
        "unresolved_mode_counts": dict(Counter(
            value["unresolved_mode"] for value in scenes if value["unresolved_mode"] is not None
        )),
    }
