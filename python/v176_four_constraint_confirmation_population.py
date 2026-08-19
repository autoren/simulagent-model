from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import hashlib
import json
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v169_fresh_constraint_state_population import constraint_signature, exact_version_space


PRIOR_CLASSES = ("alias", "composition", "provisional_primitive")


def all_source_states() -> list[dict[str, Any]]:
    rows = []
    for valuation_indices in combinations(range(8), 4):
        for outcomes in product((0, 1), repeat=4):
            constraints = [
                {"valuation_index": valuation_index, "outcome": outcome}
                for valuation_index, outcome in zip(valuation_indices, outcomes)
            ]
            rows.append(
                {
                    "constraints": constraints,
                    "constraint_signature": constraint_signature(constraints),
                }
            )
    return rows


def exact_target_context_signature(
    constraints: list[dict[str, int]], candidate: dict[str, Any]
) -> str:
    truth = "".join(str(int(value)) for value in candidate["truth_table"])
    return f"{constraint_signature(constraints)}||T={truth}"


def _state_id(signature: str) -> str:
    return "v176-" + hashlib.sha256(signature.encode()).hexdigest()[:16]


def _target_case_id(state_id: str, candidate_id: str) -> str:
    return "v176-target-" + hashlib.sha256(
        f"{state_id}|{candidate_id}".encode()
    ).hexdigest()[:16]


def _V172_target_context_signatures(
    V172_states: dict[str, Any], V172_targets: dict[str, Any]
) -> set[str]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    states = {row["state_id"]: row for row in V172_states["states"]}
    return {
        exact_target_context_signature(
            states[target["state_id"]]["constraints"],
            universe[target["target_candidate_id"]],
        )
        for target in V172_targets["target_cases"]
    }


def build_population(
    V172_states: dict[str, Any], V172_targets: dict[str, Any]
) -> dict[str, Any]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    old_signatures = _V172_target_context_signatures(V172_states, V172_targets)
    old_candidate_ids = {
        row["target_candidate_id"] for row in V172_targets["target_cases"]
    }
    states = []
    target_cases = []
    for source in all_source_states():
        version = exact_version_space(source["constraints"])
        class_counts = Counter(row["expressibility_class"] for row in version)
        eligible = set(class_counts) == set(PRIOR_CLASSES)
        state_id = _state_id(source["constraint_signature"])
        states.append(
            {
                "state_id": state_id,
                "role": "fresh_certification_aware_confirmation_population",
                "constraints": source["constraints"],
                "constraint_signature": source["constraint_signature"],
                "candidate_ids": [row["candidate_id"] for row in version],
                "candidate_count": len(version),
                "candidate_class_counts": dict(sorted(class_counts.items())),
                "confirmation_eligible": eligible,
            }
        )
        if eligible:
            for candidate in version:
                target_cases.append(
                    {
                        "target_case_id": _target_case_id(
                            state_id, candidate["candidate_id"]
                        ),
                        "state_id": state_id,
                        "target_candidate_id": candidate["candidate_id"],
                        "target_class": candidate["expressibility_class"],
                        "initial_version_space_count": len(version),
                        "class_count_within_state": class_counts[
                            candidate["expressibility_class"]
                        ],
                        "class_balanced_prior_weight": {
                            "numerator": 1,
                            "denominator": 3
                            * class_counts[candidate["expressibility_class"]],
                        },
                        "exact_target_context_signature": exact_target_context_signature(
                            source["constraints"], candidate
                        ),
                    }
                )
    states.sort(key=lambda row: row["state_id"])
    target_cases.sort(key=lambda row: row["target_case_id"])
    eligible_ids = [
        row["state_id"] for row in states if row["confirmation_eligible"]
    ]
    target_signatures = {
        row["exact_target_context_signature"] for row in target_cases
    }
    target_candidate_ids = {
        row["target_candidate_id"] for row in target_cases
    }
    state_membership = [row["state_id"] for row in states]
    target_membership = [row["target_case_id"] for row in target_cases]
    summary = {
        "source_state_count": len(all_source_states()),
        "selected_state_count": len(states),
        "confirmation_eligible_state_count": len(eligible_ids),
        "confirmation_ineligible_state_count": len(states) - len(eligible_ids),
        "target_case_count": len(target_cases),
        "target_class_counts": dict(
            sorted(Counter(row["target_class"] for row in target_cases).items())
        ),
        "candidate_count_values": sorted(
            {row["candidate_count"] for row in states}
        ),
        "class_coverage_counts": {
            str(key): value
            for key, value in sorted(
                Counter(
                    len(row["candidate_class_counts"]) for row in states
                ).items()
            )
        },
        "exact_target_context_signature_count": len(target_signatures),
        "exact_target_context_signature_overlap_with_V172": len(
            target_signatures & old_signatures
        ),
        "unique_target_candidate_count": len(target_candidate_ids),
        "candidate_identity_overlap_with_V172_count": len(
            target_candidate_ids & old_candidate_ids
        ),
        "state_membership_sha256": hashlib.sha256(
            json.dumps(state_membership, separators=(",", ":")).encode()
        ).hexdigest(),
        "target_membership_sha256": hashlib.sha256(
            json.dumps(target_membership, separators=(",", ":")).encode()
        ).hexdigest(),
        "target_context_membership_sha256": hashlib.sha256(
            json.dumps(sorted(target_signatures), separators=(",", ":")).encode()
        ).hexdigest(),
        "project_authored_procedural": True,
        "planner_policy_score_count": 0,
        "sandbox_transaction_count": 0,
        "evaluation_record_count": 0,
    }
    return {
        "states": states,
        "confirmation_eligible_state_ids": eligible_ids,
        "target_cases": target_cases,
        "summary": summary,
    }


def audit_population(
    population: dict[str, Any],
    V172_states: dict[str, Any],
    V172_targets: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    states = population["states"]
    target_cases = population["target_cases"]
    gates = config["populationGates"]
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    state_by_id = {row["state_id"]: row for row in states}
    old_signatures = _V172_target_context_signatures(V172_states, V172_targets)
    exact_versions = [
        row["candidate_ids"]
        == [
            candidate["candidate_id"]
            for candidate in exact_version_space(row["constraints"])
        ]
        for row in states
    ]
    target_exact = [
        row["state_id"] in state_by_id
        and state_by_id[row["state_id"]]["confirmation_eligible"]
        and row["target_candidate_id"]
        in state_by_id[row["state_id"]]["candidate_ids"]
        and row["exact_target_context_signature"]
        == exact_target_context_signature(
            state_by_id[row["state_id"]]["constraints"],
            universe[row["target_candidate_id"]],
        )
        for row in target_cases
    ]
    target_signatures = [
        row["exact_target_context_signature"] for row in target_cases
    ]
    expected_target_count = sum(
        row["candidate_count"]
        for row in states
        if row["confirmation_eligible"]
    )
    checks = {
        "source_and_selected_state_count": len(states)
        == len(all_source_states())
        == gates["requiredSourceStateCount"]
        == gates["requiredSelectedStateCount"],
        "candidate_count": {row["candidate_count"] for row in states}
        == {gates["requiredCandidatesPerState"]},
        "eligible_state_count": len(population["confirmation_eligible_state_ids"])
        == gates["requiredEligibleStateCount"],
        "ineligible_states_retained": len(states)
        - len(population["confirmation_eligible_state_ids"])
        == gates["requiredIneligibleStateCount"],
        "target_case_count": len(target_cases)
        == expected_target_count
        == gates["requiredTargetCaseCount"],
        "target_class_counts": population["summary"]["target_class_counts"]
        == gates["requiredTargetClassCounts"],
        "class_coverage_counts": population["summary"]["class_coverage_counts"]
        == gates["requiredClassCoverageCounts"],
        "constraint_signature_uniqueness": len(
            {row["constraint_signature"] for row in states}
        )
        / len(states)
        == gates["requiredConstraintSignatureUniqueness"],
        "version_space_exactness": sum(exact_versions) / len(exact_versions)
        == gates["requiredVersionSpaceExactness"],
        "target_membership_exactness": sum(target_exact) / len(target_exact)
        == gates["requiredTargetMembershipExactness"],
        "class_balanced_prior_weights_normalize_per_state": all(
            sum(
                Fraction(
                    row["class_balanced_prior_weight"]["numerator"],
                    row["class_balanced_prior_weight"]["denominator"],
                )
                for row in target_cases
                if row["state_id"] == state_id
            )
            == 1
            for state_id in population["confirmation_eligible_state_ids"]
        ),
        "exact_target_context_signatures_are_unique": len(set(target_signatures))
        / len(target_signatures)
        == gates["requiredExactTargetContextSignatureUniqueness"],
        "zero_exact_target_context_signature_overlap_with_V172": len(
            set(target_signatures) & old_signatures
        )
        == gates["requiredExactTargetContextSignatureOverlapWithV172"],
        "zero_policy_transaction_or_evaluation_access": (
            population["summary"]["planner_policy_score_count"]
            <= gates["maximumPlannerPolicyScoreCount"]
            and population["summary"]["sandbox_transaction_count"]
            <= gates["maximumSandboxTransactionCount"]
            and population["summary"]["evaluation_record_count"]
            <= gates["maximumEvaluationRecordCount"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": population["summary"],
    }


__all__ = [
    "PRIOR_CLASSES",
    "all_source_states",
    "audit_population",
    "build_population",
    "exact_target_context_signature",
]
