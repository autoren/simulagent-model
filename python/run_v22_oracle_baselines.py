"""Run exact lifted-schema and non-relational lookup baselines for V22 development."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from audit_v22_relational import read_records, reference_world, trace_for_induction
from run_v18_schema_baselines import balanced_accuracy, episode_summary
from v10_protocol import file_sha256
from v22_relational import (
    enumerate_program_hypotheses,
    execute_partial,
    rows_to_epistemic,
    trace_consistent_hypotheses,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def outcome_vocabulary(bits: int) -> list[str]:
    return [
        "transition_" + format(value, f"0{bits}b")
        for value in range(2 ** bits)
    ]


def prediction_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "queries": 0, "transition_set_exact_match": 0.0,
            "identifiability_accuracy": 0.0, "identifiability_balanced_accuracy": 0.0,
        }
    target_identifiable = [value["target_identifiable"] for value in rows]
    predicted_identifiable = [value["predicted_identifiable"] for value in rows]
    result = {
        "queries": len(rows),
        "transition_set_exact_match": sum(value["transition_set_exact"] for value in rows) / len(rows),
        "identifiability_accuracy": sum(
            target == predicted
            for target, predicted in zip(target_identifiable, predicted_identifiable, strict=True)
        ) / len(rows),
        "identifiability_balanced_accuracy": balanced_accuracy(
            target_identifiable, predicted_identifiable
        ),
    }
    for effect in ("fully_observed", "outcome_invariant", "outcome_sensitive"):
        selected = [value for value in rows if value["unknown_effect"] == effect]
        result[f"{effect}_transition_set_exact_match"] = (
            sum(value["transition_set_exact"] for value in selected) / len(selected)
            if selected else None
        )
    return result


def row(
    record: dict[str, Any], query: dict[str, Any], prediction: dict[str, Any], condition: str,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "episode_id": record["id"],
        "split": record["split"],
        "family": record["construction_family"],
        "axis": query["query_axis"],
        "query_id": query["id"],
        "entity_count": query["entity_count"],
        "unknown_effect": query["unknown_effect"],
        "target_identifiable": query["identifiable"],
        "predicted_identifiable": prediction["identifiable"],
        "transition_set_exact": prediction["possible_transition_codes"] == query["possible_transition_codes"],
        "predicted_transition_codes": prediction["possible_transition_codes"],
        "target_transition_codes": query["possible_transition_codes"],
    }


def metamorphic_consistency(
    records: Sequence[dict[str, Any]], predictions: dict[str, dict[str, Any]], axis: str,
) -> float:
    values = []
    for record in records:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for query in record["oracle_grounding"]["queries"]:
            if query["query_axis"] == axis and query.get("metamorphic_group"):
                groups[query["metamorphic_group"]].append(query)
        for queries in groups.values():
            if len(queries) != 2:
                values.append(False)
                continue
            left = predictions[queries[0]["id"]]["possible_transition_codes"]
            right = predictions[queries[1]["id"]]["possible_transition_codes"]
            values.append(left == right)
    return sum(values) / len(values) if values else 0.0


def evaluate(records: Sequence[dict[str, Any]], config: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if not audit["passed"]:
        raise RuntimeError("V22 structural audit must pass before oracle baselines")
    exact_rows = []
    lookup_rows = []
    schema_rows = []
    scaling_rows = []
    exact_predictions: dict[str, dict[str, Any]] = {}
    for record in records:
        bits = record["agent_input"]["dsl_contract"]["outcome_bits"]
        hypotheses = enumerate_program_hypotheses(bits)
        traces = [trace_for_induction(value) for value in record["oracle_grounding"]["support"]]
        tracemalloc.start()
        started = time.perf_counter()
        version_space = trace_consistent_hypotheses(hypotheses, traces, config)
        search_seconds = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        target_key = record["target"]["program_key"]
        target_retained = any(value.key == target_key for value in version_space)
        target_unique = len(version_space) == 1 and version_space[0].key == target_key
        schema_rows.append({
            "episode_id": record["id"], "split": record["split"],
            "family": record["construction_family"], "outcome_bits": bits,
            "initial_hypotheses": len(hypotheses), "surviving_hypotheses": len(version_space),
            "target_retained": target_retained, "target_unique": target_unique,
            "support_traces": len(traces),
        })
        scaling_rows.append({
            "episode_id": record["id"], "family": record["construction_family"],
            "outcome_bits": bits, "candidate_behaviors": len(hypotheses),
            "surviving_behaviors": len(version_space), "support_traces": len(traces),
            "candidate_trace_executions_upper_bound": len(hypotheses) * len(traces),
            "search_seconds": search_seconds, "peak_traced_bytes": peak,
        })
        lookup = {
            value["canonical_state_binding_hash"]: value["transition_code"]
            for value in record["oracle_grounding"]["support"]
        }
        for query in record["oracle_grounding"]["queries"]:
            state = rows_to_epistemic(query["epistemic_state"])
            exact = execute_partial(
                [value.program for value in version_space], config, query["entities"], state,
                query["action_binding"], config["limits"]["maximumUnknownAtomsPerQuery"],
            )
            exact_predictions[query["id"]] = exact
            exact_rows.append(row(record, query, exact, "exact_lifted_version_space"))
            if query["canonical_state_binding_hash"] in lookup and not query["unknown_atoms"]:
                possible = [lookup[query["canonical_state_binding_hash"]]]
            else:
                possible = outcome_vocabulary(bits)
            lookup_prediction = {
                "possible_transition_codes": possible,
                "identifiable": len(possible) == 1,
            }
            lookup_rows.append(row(record, query, lookup_prediction, "literal_graph_lookup"))

    exact = prediction_summary(exact_rows)
    lookup = prediction_summary(lookup_rows)
    exact_by_axis = {
        axis: {
            **prediction_summary([value for value in exact_rows if value["axis"] == axis]),
            **episode_summary([value for value in exact_rows if value["axis"] == axis]),
        }
        for axis in config["queryAxes"]
    }
    exact_by_family = {
        family: {
            **prediction_summary([value for value in exact_rows if value["family"] == family]),
            **episode_summary([value for value in exact_rows if value["family"] == family]),
            "schema_recovery": sum(
                value["target_unique"] for value in schema_rows if value["family"] == family
            ),
        }
        for family in config["constructionFamilies"]
    }
    exact_by_entity_count = {
        str(entity_count): prediction_summary([
            value for value in exact_rows if value["entity_count"] == entity_count
        ])
        for entity_count in config["queryEntityCounts"]
    }
    scaling_by_bits = {}
    for bits in config["outcomeBits"]:
        values = [value for value in scaling_rows if value["outcome_bits"] == bits]
        scaling_by_bits[str(bits)] = {
            "episodes": len(values),
            "candidate_behaviors": sorted({value["candidate_behaviors"] for value in values}),
            "median_surviving_behaviors": median(value["surviving_behaviors"] for value in values),
            "median_support_traces": median(value["support_traces"] for value in values),
            "median_search_seconds": median(value["search_seconds"] for value in values),
            "maximum_search_seconds": max(value["search_seconds"] for value in values),
            "median_peak_traced_bytes": median(value["peak_traced_bytes"] for value in values),
            "maximum_candidate_trace_executions_upper_bound": max(
                value["candidate_trace_executions_upper_bound"] for value in values
            ),
        }
    schema_recovery = sum(value["target_unique"] for value in schema_rows) / len(schema_rows)
    target_retention = sum(value["target_retained"] for value in schema_rows) / len(schema_rows)
    permutation = metamorphic_consistency(
        records, exact_predictions, "permutation_equivariance"
    )
    distractor = metamorphic_consistency(records, exact_predictions, "distractor_invariance")
    orientation_accuracy = exact_by_axis["relation_orientation"]["transition_set_exact_match"]
    gates = config["gates"]
    checks = {
        "oracle_schema_recovery": schema_recovery >= gates["minimumOracleSchemaRecovery"],
        "target_retention": target_retention == 1.0,
        "oracle_transition_set_exact": exact["transition_set_exact_match"] >= gates["minimumOracleTransitionSetExact"],
        "identifiability_accuracy": exact["identifiability_accuracy"] >= gates["minimumIdentifiabilityAccuracy"],
        "permutation_consistency": permutation >= gates["minimumPermutationConsistency"],
        "distractor_consistency": distractor >= gates["minimumDistractorConsistency"],
        "orientation_accuracy": orientation_accuracy >= gates["minimumOrientationAuditAccuracy"],
        "empirical_lookup_imperfect": lookup["transition_set_exact_match"] <= gates["maximumEmpiricalLookupTransitionSetExact"],
        "support_budget": max(value["support_traces"] for value in schema_rows) <= config["limits"]["maximumSupportTraces"],
        "zero_model_access": (
            config["limits"]["newModelForwardPassesPermitted"] == 0
            and config["limits"]["newLinearFitsPermitted"] == 0
            and config["limits"]["adapterTrainingRunsPermitted"] == 0
        ),
    }
    return {
        "schema_version": 22,
        "experiment": "v22_typed_relational_oracle_development_baselines",
        "condition": "oracle_relational_grounding",
        "passed": all(checks.values()),
        "decision": (
            "authorize_relational_language_grounding_development"
            if all(checks.values()) else "revise_v22_relational_foundation"
        ),
        "checks": checks,
        "metrics": {
            "episodes": len(records),
            "queries": len(exact_rows),
            "schema_recovery_rate": schema_recovery,
            "target_retention_rate": target_retention,
            "maximum_surviving_hypotheses": max(value["surviving_hypotheses"] for value in schema_rows),
            "exact_lifted_version_space": exact,
            "literal_graph_lookup": lookup,
            "exact_by_axis": exact_by_axis,
            "exact_by_family": exact_by_family,
            "exact_by_entity_count": exact_by_entity_count,
            "permutation_consistency": permutation,
            "distractor_consistency": distractor,
            "search_scaling_by_outcome_bits": scaling_by_bits,
            "search_scaling_episodes": scaling_rows,
        },
        "data_access": {
            "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "final_v22_records_created": 0,
        },
    }


def markdown(result: dict[str, Any], audit: dict[str, Any]) -> str:
    metrics = result["metrics"]
    exact = metrics["exact_lifted_version_space"]
    lookup = metrics["literal_graph_lookup"]
    lines = [
        "# V22 development results: typed relational oracle foundation",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "This is an open development result, not a sealed final evaluation. It authorizes work on",
        "the relational language-grounding interface only; it does not authorize final data, new",
        "model weights, or a matched neural challenger yet.",
        "",
        "## Oracle result",
        "",
        f"- {metrics['episodes']} mechanics and {metrics['queries']} relational queries;",
        f"- exact lifted schema recovery: {metrics['schema_recovery_rate']:.3f};",
        f"- exact transition-set match: {exact['transition_set_exact_match']:.3f};",
        f"- identifiability accuracy: {exact['identifiability_accuracy']:.3f};",
        f"- permutation consistency: {metrics['permutation_consistency']:.3f};",
        f"- distractor consistency: {metrics['distractor_consistency']:.3f}; and",
        f"- literal graph lookup transition-set match: {lookup['transition_set_exact_match']:.3f}.",
        "",
        "## Relational families",
        "",
        "| Family | Schemas recovered | Complete episodes | Queries | Exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, value in metrics["exact_by_family"].items():
        lines.append(
            f"| `{family}` | {value['schema_recovery']}/{value['episodes']} | "
            f"{value['complete_episodes']}/{value['episodes']} | {value['queries']} | "
            f"{value['transition_set_exact_match']:.3f} |"
        )
    lines.extend([
        "",
        "## Search scaling",
        "",
        "| Outcome bits | Candidates | Median support | Median search seconds | Maximum executions |",
        "|---:|---:|---:|---:|---:|",
    ])
    for bits, value in metrics["search_scaling_by_outcome_bits"].items():
        lines.append(
            f"| {bits} | {value['candidate_behaviors'][0]} | {value['median_support_traces']:.1f} | "
            f"{value['median_search_seconds']:.6f} | "
            f"{value['maximum_candidate_trace_executions_upper_bound']} |"
        )
    lines.extend([
        "",
        "## Audit and firewall",
        "",
        f"- structural/metamorphic audit: {'pass' if audit['passed'] else 'fail'};",
        "- complete false facts are distinct from explicit unknown facts;",
        "- fit/evaluation programs and registered structural query axes are disjoint;",
        "- no target program or structured oracle-graph field appears in agent inputs;",
        "- zero V21-final record or model-result reads; and",
        "- zero model forwards, linear fits, adapter runs, or V22-final constructions.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v22.json")
    parser.add_argument("--dataset", default="data/v22-relational-development")
    parser.add_argument("--audit", default="outputs/v22-relational-development/audit.json")
    parser.add_argument("--output", default="outputs/v22-relational-development/oracle-baselines.json")
    parser.add_argument("--markdown", default="docs/v22-results.md")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    audit_path = PROJECT_ROOT / args.audit
    audit = json.loads(audit_path.read_text())
    root = PROJECT_ROOT / args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    for path, expected in manifest["implementation_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V22 implementation changed after generation: {path}")
    result = evaluate(read_records(root), config, audit)
    result["source"] = {
        "dataset_manifest": str((root / "manifest.json").relative_to(PROJECT_ROOT)),
        "dataset_sha256": manifest["dataset_sha256"],
        "audit": args.audit,
        "audit_sha256": file_sha256(audit_path),
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (PROJECT_ROOT / args.markdown).write_text(markdown(result, audit))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
