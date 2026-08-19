from __future__ import annotations

from collections import Counter
from typing import Any

from v148_typed_witness_firewall import finalize_witness
from v149_fresh_closed_interaction_population import witness_from_answer_event


def decision_cost(truth: str, final: str, known_ids: set[str], config: dict[str, Any]) -> float:
    costs = config["decisionCosts"]
    if final == truth:
        return costs["correct"]
    if final == "A00":
        return costs["safeAbstentionOnDecidableTruth"]
    if final in known_ids and truth not in known_ids:
        return costs["wrongKnownOnNonKnownTruth"]
    if final in known_ids and truth in known_ids:
        return costs["wrongKnownOnKnownTruth"]
    return costs["otherWrongState"]


def build_episodes(development_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in development_metadata:
        groups.setdefault(row["group_id"], {})[row["stage"]] = row
    episodes = []
    for group_id, stages in sorted(groups.items()):
        ambiguous = stages["request_ambiguous"]
        for side, answer_stage in (("left", "closed_answer_known"), ("right", "closed_answer_right")):
            answered = stages[answer_stage]
            episodes.append(
                {
                    "episode_id": f"{group_id}::{side}",
                    "group_id": group_id,
                    "family_id": answered["family_id"],
                    "side": side,
                    "truth_state_id": answered["truth_state_id"],
                    "compatible_state_ids": ambiguous["compatible_state_ids"],
                    "oracle_query_id": ambiguous["oracle_query_id"],
                    "closed_answer_event": answered["closed_answer_event"],
                }
            )
    return episodes


def exact_query_plan(episode: dict[str, Any], catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    known_ids = set(config["knownIds"])
    pair = episode["compatible_state_ids"]
    no_query_cost = sum(decision_cost(truth, "A00", known_ids, config) for truth in pair) / len(pair)
    values = {"NO_QUERY": no_query_cost}
    for query in catalog["queries"]:
        if query["query_id"] == episode["oracle_query_id"]:
            residual = 0.0
        else:
            residual = no_query_cost
        values[query["query_id"]] = config["policy"]["queryCost"] + residual
    selected = min(values, key=lambda action: (values[action], action))
    return {
        "selected_action": selected,
        "action_values": values,
        "no_query_cost": no_query_cost,
        "selected_expected_cost": values[selected],
    }


def evaluate(
    development_metadata: list[dict[str, Any]],
    catalog: dict[str, Any],
    witness_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    episodes = build_episodes(development_metadata)
    query_proposals = ["NONE"] + [row["query_id"] for row in catalog["queries"]]
    candidate_proposals = witness_config["outputIds"]
    records = []
    for episode in episodes:
        plan = exact_query_plan(episode, catalog, witness_config | config)
        for candidate_proposal in candidate_proposals:
            for query_proposal in query_proposals:
                if plan["selected_action"] == episode["oracle_query_id"]:
                    witness = witness_from_answer_event(episode["closed_answer_event"], catalog)
                else:
                    witness = None
                output = finalize_witness(witness, candidate_proposal, witness_config)
                final_state = output["final_state_id"]
                records.append(
                    {
                        "episode_id": episode["episode_id"],
                        "family_id": episode["family_id"],
                        "side": episode["side"],
                        "truth": episode["truth_state_id"],
                        "candidate_proposal": candidate_proposal,
                        "query_proposal": query_proposal,
                        "selected_query": plan["selected_action"],
                        "oracle_query": episode["oracle_query_id"],
                        "final": final_state,
                        "cost": config["policy"]["queryCost"] + decision_cost(episode["truth_state_id"], final_state, set(witness_config["knownIds"]), config),
                        "no_query_cost": plan["no_query_cost"],
                        "retained": output["authoritative_hypothesis_ids_retained"] == witness_config["outputIds"],
                    }
                )

    per_episode: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        per_episode.setdefault(row["episode_id"], []).append(row)
    representative = [rows[0] for rows in per_episode.values()]
    nonknown = [row for row in representative if row["truth"] not in witness_config["knownIds"]]
    irrelevant_outputs = []
    for episode in episodes:
        for query in catalog["queries"]:
            if query["query_id"] == episode["oracle_query_id"]:
                continue
            output = finalize_witness(None, "K41", witness_config)
            irrelevant_outputs.append(output)
    family_metrics = {}
    for family_id in sorted({row["family_id"] for row in representative}):
        subset = [row for row in representative if row["family_id"] == family_id]
        family_metrics[family_id] = {
            "episode_count": len(subset),
            "mean_cost": sum(row["cost"] for row in subset) / len(subset),
            "mean_no_query_cost": sum(row["no_query_cost"] for row in subset) / len(subset),
            "improvement": sum(row["no_query_cost"] - row["cost"] for row in subset) / len(subset),
        }
    metrics = {
        "group_count": len({row["group_id"] for row in episodes}),
        "episode_count": len(episodes),
        "candidate_proposal_count": len(candidate_proposals),
        "query_proposal_count": len(query_proposals),
        "policy_evaluation_count": len(records),
        "correct_query_selection_rate": sum(row["selected_query"] == row["oracle_query"] for row in representative) / len(representative),
        "final_exact_accuracy": sum(row["final"] == row["truth"] for row in representative) / len(representative),
        "candidate_and_query_proposal_invariance": sum(
            len({(row["selected_query"], row["final"]) for row in rows}) == 1
            for rows in per_episode.values()
        ) / len(per_episode),
        "sequential_mean_decision_cost": sum(row["cost"] for row in representative) / len(representative),
        "no_query_mean_decision_cost": sum(row["no_query_cost"] for row in representative) / len(representative),
        "improvement_over_no_query": sum(row["no_query_cost"] - row["cost"] for row in representative) / len(representative),
        "resolved_per_query": sum(row["final"] == row["truth"] for row in representative) / len(representative),
        "false_known_on_nonknown_truth": sum(row["final"] in witness_config["knownIds"] for row in nonknown) / len(nonknown),
        "safe_nonknown_on_nonknown_truth": sum(row["final"] not in witness_config["knownIds"] for row in nonknown) / len(nonknown),
        "irrelevant_query_fail_closed_rate": sum(not row["witness_valid"] and row["final_state_id"] == "A00" for row in irrelevant_outputs) / len(irrelevant_outputs),
        "authoritative_true_hypothesis_retention": sum(row["retained"] for row in records) / len(records),
        "truth_counts": dict(sorted(Counter(row["truth"] for row in representative).items())),
        "evaluation_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_or_score_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "actual_execution_count": 0,
    }
    gates = config["gates"]
    checks = {
        "population_counts": bool(
            metrics["group_count"] == config["population"]["groupCount"]
            and metrics["episode_count"] == gates["requiredEpisodeCount"]
            and metrics["policy_evaluation_count"] == gates["requiredPolicyEvaluationCount"]
        ),
        "correct_query_selection": metrics["correct_query_selection_rate"] == gates["requiredCorrectQuerySelectionRate"],
        "final_exact": metrics["final_exact_accuracy"] == gates["requiredFinalExactAccuracy"],
        "proposal_invariance": metrics["candidate_and_query_proposal_invariance"] == gates["requiredCandidateAndQueryProposalInvariance"],
        "sequential_cost": metrics["sequential_mean_decision_cost"] <= gates["maximumSequentialMeanDecisionCost"] + 1e-12,
        "improvement": metrics["improvement_over_no_query"] + 1e-12 >= gates["minimumImprovementOverNoQuery"],
        "evidence_efficiency": metrics["resolved_per_query"] == gates["requiredResolvedPerQuery"],
        "zero_false_known": metrics["false_known_on_nonknown_truth"] <= gates["maximumFalseKnownOnNonKnownTruth"],
        "safe_nonknown": metrics["safe_nonknown_on_nonknown_truth"] == gates["requiredSafeNonKnownOnNonKnownTruth"],
        "irrelevant_query_fail_closed": metrics["irrelevant_query_fail_closed_rate"] == gates["requiredIrrelevantQueryFailClosedRate"],
        "authoritative_retention": metrics["authoritative_true_hypothesis_retention"] == gates["requiredAuthoritativeTrueHypothesisRetention"],
        "zero_access_and_execution": bool(
            metrics["evaluation_language_read_count"] <= gates["maximumEvaluationLanguageReadCount"]
            and metrics["model_load_count"] <= gates["maximumModelLoadCount"]
            and metrics["model_generation_or_score_count"] <= gates["maximumModelGenerationOrScoreCount"]
            and metrics["API_call_count"] <= gates["maximumAPICallCount"]
            and metrics["training_run_count"] <= gates["maximumTrainingRunCount"]
            and metrics["actual_execution_count"] <= gates["maximumActualExecutionCount"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics, "family_metrics": family_metrics}


__all__ = ["build_episodes", "decision_cost", "evaluate", "exact_query_plan"]
