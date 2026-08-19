from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import hashlib
import json
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe, valuations


def constraint_signature(constraints: list[dict[str, int]]) -> str:
    normalized = sorted((int(row["valuation_index"]), int(row["outcome"])) for row in constraints)
    return "|".join(f"V{index}={outcome}" for index, outcome in normalized)


def v165_ambiguous_signatures(hidden_records: list[dict[str, Any]]) -> set[str]:
    signatures: set[str] = set()
    domain = valuations()
    for record in hidden_records:
        if record["evidence_status"] != "ambiguous":
            continue
        names = record["registered_primitives"]
        constraints = []
        for observation in record["observations"]:
            assignment = observation["intervention"]["set_primitives"]
            bits = tuple(bool(assignment[name]) for name in names)
            constraints.append({
                "valuation_index": domain.index(bits),
                "outcome": int(bool(observation["observed_qualifies"])),
            })
        signatures.add(constraint_signature(constraints))
    return signatures


def all_source_states() -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(range(8), 2):
        for left_value, right_value in product((0, 1), repeat=2):
            constraints = [
                {"valuation_index": left, "outcome": left_value},
                {"valuation_index": right, "outcome": right_value},
            ]
            rows.append({"constraints": constraints, "constraint_signature": constraint_signature(constraints)})
    return rows


def exact_version_space(constraints: list[dict[str, int]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidate_universe()
        if all(int(candidate["truth_table"][row["valuation_index"]]) == row["outcome"] for row in constraints)
    ]


def _state_id(signature: str) -> str:
    return "v169-" + hashlib.sha256(signature.encode()).hexdigest()[:16]


def build_population(hidden_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    excluded = v165_ambiguous_signatures(hidden_records)
    source = all_source_states()
    selected = []
    for row in source:
        if row["constraint_signature"] in excluded:
            continue
        version = exact_version_space(row["constraints"])
        counts = Counter(candidate["expressibility_class"] for candidate in version)
        eligible = set(counts) == set(config["priorClasses"] if "priorClasses" in config else ["alias", "composition", "provisional_primitive"])
        selected.append({
            "state_id": _state_id(row["constraint_signature"]),
            "role": "fresh_confirmation_population",
            "constraints": row["constraints"],
            "constraint_signature": row["constraint_signature"],
            "candidate_ids": [candidate["candidate_id"] for candidate in version],
            "candidate_count": len(version),
            "candidate_class_counts": dict(sorted(counts.items())),
            "planner_eligible": eligible,
        })
    selected.sort(key=lambda row: row["state_id"])
    eligible_ids = [row["state_id"] for row in selected if row["planner_eligible"]]
    membership_hash = hashlib.sha256(
        json.dumps([row["state_id"] for row in selected], separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "source_state_count": len(source),
        "excluded_V165_signature_count": len(excluded),
        "selected_state_count": len(selected),
        "planner_eligible_state_count": len(eligible_ids),
        "planner_ineligible_state_count": len(selected) - len(eligible_ids),
        "selected_membership_sha256": membership_hash,
        "candidate_count_values": sorted({row["candidate_count"] for row in selected}),
        "class_coverage_counts": dict(sorted(Counter(len(row["candidate_class_counts"]) for row in selected).items())),
        "project_authored_procedural": True,
        "planner_policy_score_count": 0,
        "evaluation_record_count": 0,
    }
    return {
        "states": selected,
        "eligible_state_ids": eligible_ids,
        "excluded_V165_signatures": sorted(excluded),
        "summary": summary,
    }


def audit_population(population: dict[str, Any], hidden_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["populationGates"]
    states = population["states"]
    source = all_source_states()
    excluded = v165_ambiguous_signatures(hidden_records)
    signatures = [row["constraint_signature"] for row in states]
    exact = [
        row["candidate_ids"] == [candidate["candidate_id"] for candidate in exact_version_space(row["constraints"])]
        for row in states
    ]
    class_sums = [sum(row["candidate_class_counts"].values()) == row["candidate_count"] for row in states]
    nonoverlap_source = [row for row in source if row["constraint_signature"] not in excluded]
    checks = {
        "source_state_count": len(source) == gates["requiredSourceStateCount"],
        "excluded_signature_count": len(excluded) >= gates["minimumExcludedV165SignatureCount"],
        "zero_V165_signature_overlap": len(set(signatures) & excluded) == gates["requiredOverlapWithV165AmbiguousSignatures"],
        "candidate_count": set(row["candidate_count"] for row in states) == {gates["requiredCandidatesPerSelectedState"]},
        "selected_state_count": len(states) >= gates["minimumSelectedStateCount"],
        "eligible_state_count": len(population["eligible_state_ids"]) >= gates["minimumPlannerEligibleStateCount"],
        "signature_uniqueness": len(set(signatures)) / len(signatures) == gates["requiredConstraintSignatureUniqueness"],
        "version_space_exactness": sum(exact) / len(exact) == gates["requiredVersionSpaceExactness"],
        "class_count_sum_exactness": sum(class_sums) / len(class_sums) == gates["requiredClassCountSumExactness"],
        "all_nonoverlapping_states_retained": len(states) / len(nonoverlap_source) == gates["requiredAllNonoverlappingStatesRetained"],
        "zero_planner_policy_scores": population["summary"]["planner_policy_score_count"] <= gates["maximumPlannerPolicyScoreCount"],
        "zero_evaluation_records": population["summary"]["evaluation_record_count"] <= gates["maximumEvaluationRecordCount"],
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": population["summary"]}


__all__ = [
    "all_source_states", "audit_population", "build_population", "constraint_signature",
    "exact_version_space", "v165_ambiguous_signatures",
]
