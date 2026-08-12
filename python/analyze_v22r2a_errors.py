"""No-refit V22r2a error conditioning, gold ranks, and component-oracle replay."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import (
    integration_condition,
    load_npz,
    pair_features,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def feature_maps(arrays):
    return (
        {str(key): arrays["candidate_features"][i] for i, key in enumerate(arrays["candidate_ids"].tolist())},
        {str(key): arrays["evidence_features"][i] for i, key in enumerate(arrays["evidence_ids"].tolist())},
    )


def mean(values):
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v22r2a-evaluation-amendment-lock.json")
    parser.add_argument("--result", default="outputs/v22r2-relational-grounding/evaluation-v22r2a/result.json")
    parser.add_argument("--output", default="outputs/v22r2-relational-grounding/error-conditioning.json")
    parser.add_argument("--markdown", default="docs/v22r2-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    if result["amendment_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V22r2a result and amendment lock differ")
    original = json.loads((PROJECT_ROOT / lock["source"]["original_lock"]).read_text())
    metadata = json.loads((PROJECT_ROOT / lock["source"]["feature_metadata"]).read_text())
    arrays = load_npz(PROJECT_ROOT / metadata["feature_artifact"])
    heads = load_npz(PROJECT_ROOT / result["heads_artifact"])
    scenes = read_jsonl_directory(PROJECT_ROOT / original["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    records = read_jsonl_directory(PROJECT_ROOT / original["source"]["dataset"] / "records")
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    stored_predictions = {
        row["scene_id"]: row for row in [
            json.loads(line) for line in (PROJECT_ROOT / result["grounding_predictions"]).read_text().splitlines()
            if line.strip()
        ]
    }
    candidate_features, evidence_features = feature_maps(arrays)
    atom_coef = heads["atom_coef"][0]
    atom_intercept = float(heads["atom_intercept"][0])
    truth_classes = heads["truth_classes"].tolist()
    truth_coef = heads["truth_coef"]
    truth_intercept = heads["truth_intercept"]
    rank_rows = []
    variants = {name: {} for name in (
        "frozen", "oracle_assignment_frozen_truth",
        "frozen_assignment_oracle_truth", "oracle",
    )}
    for scene in scenes:
        prediction = stored_predictions[scene["id"]]
        target_by_evidence = {
            row["evidence_id"]: row for row in scene["target"]["atom_groundings"]
        }
        public_candidates = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
        candidate_x = np.stack([candidate_features[value] for value in public_candidates])
        frozen_rows = []
        oracle_assignment_rows = []
        oracle_truth_rows = []
        oracle_rows = []
        predicted_by_evidence = {row["evidence_id"]: row for row in prediction["rows"]}
        for evidence_row in scene["agent_input"]["evidence"]:
            evidence_id = evidence_row["id"]
            target = target_by_evidence[evidence_id]
            vector = evidence_features[evidence_id]
            scores = sigmoid(pair_features(vector[None, :], candidate_x) @ atom_coef + atom_intercept)
            gold_index = public_candidates.index(target["candidate_id"])
            atom_rank = 1 + int(np.sum(scores > scores[gold_index]))
            truth_scores = sigmoid(truth_coef @ vector + truth_intercept)
            truth_scores = truth_scores / truth_scores.sum()
            gold_truth = truth_classes.index(target["truth_label"])
            truth_rank = 1 + int(np.sum(truth_scores > truth_scores[gold_truth]))
            rank_rows.append({
                "split": scene["split"], "role": scene["role"],
                "surface_bank": target["surface_bank"],
                "predicate_kind": target["predicate_kind"],
                "relation_orientation": target["relation_orientation"],
                "truth_label": target["truth_label"],
                "semantic_operator": target["semantic_operator"],
                "atom_rank": atom_rank, "truth_rank": truth_rank,
            })
            frozen = predicted_by_evidence[evidence_id]
            frozen_rows.append({
                "evidence_id": evidence_id, "candidate_id": frozen["candidate_id"],
                "truth_label": frozen["truth_label"],
            })
            oracle_assignment_rows.append({
                "evidence_id": evidence_id, "candidate_id": target["candidate_id"],
                "truth_label": frozen["truth_label"],
            })
            oracle_truth_rows.append({
                "evidence_id": evidence_id, "candidate_id": frozen["candidate_id"],
                "truth_label": target["truth_label"],
            })
            oracle_rows.append({
                "evidence_id": evidence_id, "candidate_id": target["candidate_id"],
                "truth_label": target["truth_label"],
            })
        for name, rows in (
            ("frozen", frozen_rows),
            ("oracle_assignment_frozen_truth", oracle_assignment_rows),
            ("frozen_assignment_oracle_truth", oracle_truth_rows),
            ("oracle", oracle_rows),
        ):
            variants[name][scene["id"]] = {
                "epistemic_state": predicted_epistemic_rows(scene, rows)
            }

    def ranks_for(split, role=None):
        selected = [row for row in rank_rows if row["split"] == split and (role is None or row["role"] == role)]
        return {
            "atoms": len(selected),
            "gold_atom_top1": mean([row["atom_rank"] <= 1 for row in selected]),
            "gold_atom_top2": mean([row["atom_rank"] <= 2 for row in selected]),
            "gold_atom_top3": mean([row["atom_rank"] <= 3 for row in selected]),
            "gold_truth_top1": mean([row["truth_rank"] <= 1 for row in selected]),
            "gold_truth_top2": mean([row["truth_rank"] <= 2 for row in selected]),
        }

    v22_config = json.loads((PROJECT_ROOT / original["source"]["v22_config"]).read_text())
    component_replay = {}
    for name, lookup in variants.items():
        component_replay[name] = {
            "support_component_with_oracle_queries": integration_condition(
                evaluation_records, "frozen", "oracle", lookup, v22_config, original["config_payload"]
            ),
            "query_component_with_oracle_support": integration_condition(
                evaluation_records, "oracle", "frozen", lookup, v22_config, original["config_payload"]
            ),
        }
    by_operator = {}
    evaluation_rows = [row for row in rank_rows if row["split"] == "grounding_evaluation"]
    for operator in sorted({row["semantic_operator"] for row in evaluation_rows}):
        selected = [row for row in evaluation_rows if row["semantic_operator"] == operator]
        by_operator[operator] = {
            "atoms": len(selected),
            "gold_atom_top1": mean([row["atom_rank"] == 1 for row in selected]),
            "gold_truth_top1": mean([row["truth_rank"] == 1 for row in selected]),
        }
    output = {
        "schema_version": "22r2a",
        "experiment": "v22r2a_no_refit_error_conditioning",
        "source_result_sha256": file_sha256(result_path),
        "new_model_forward_passes": 0, "new_linear_fits": 0,
        "hyperparameter_selections": 0,
        "gold_rank_retention": {
            split: {role: ranks_for(split, role) for role in ("support", "query")}
            for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation")
        },
        "evaluation_by_semantic_operator": by_operator,
        "component_oracle_replay": component_replay,
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    eval_rank = output["gold_rank_retention"]["grounding_evaluation"]
    support_frozen = component_replay["frozen"]["support_component_with_oracle_queries"]
    support_oracle_assignment = component_replay["oracle_assignment_frozen_truth"]["support_component_with_oracle_queries"]
    support_oracle_truth = component_replay["frozen_assignment_oracle_truth"]["support_component_with_oracle_queries"]
    query_frozen = component_replay["frozen"]["query_component_with_oracle_support"]
    query_oracle_assignment = component_replay["oracle_assignment_frozen_truth"]["query_component_with_oracle_support"]
    query_oracle_truth = component_replay["frozen_assignment_oracle_truth"]["query_component_with_oracle_support"]
    lines = [
        "# V22r2 results: hard relational-language grounding",
        "",
        f"Decision: `{result['decision']}`. Development gates passed: `{str(result['passed']).lower()}`.",
        "",
        "V22r2 is an open development result. The V22 symbolic oracle remains exact, but the fixed",
        "hard language interface does not yet populate it reliably from held-out wording.",
        "",
        "## Registered result",
        "",
        "| Condition | Transition-set exact | Target retained | Empty version spaces |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["integration"].items():
        lines.append(f"| `{name}` | {row['transition_set_exact_match']:.3f} | {row['target_retention_rate']:.3f} | {row['empty_version_space_rate']:.3f} |")
    evaluation = result["grounding"]["by_split"]["grounding_evaluation"]
    lines.extend([
        "", "## Held-out grounding", "",
        f"- atom assignment: {evaluation['atom_assignment_accuracy']:.3f};",
        f"- ordered-relation assignment: {evaluation['relation_argument_order_accuracy']:.3f};",
        f"- truth status: {evaluation['truth_status_accuracy']:.3f};",
        f"- exact scene graph: {evaluation['exact_scene_graph']:.3f}; and",
        f"- all-support episodes exact: {evaluation['episodes_with_all_support_graphs_exact']:.3f}.",
        "", "## No-refit diagnostic", "",
        f"Evaluation support gold-atom retention rises from {eval_rank['support']['gold_atom_top1']:.3f} at top 1 to {eval_rank['support']['gold_atom_top2']:.3f} at top 2 and {eval_rank['support']['gold_atom_top3']:.3f} at top 3.",
        f"Evaluation query gold-atom retention rises from {eval_rank['query']['gold_atom_top1']:.3f} at top 1 to {eval_rank['query']['gold_atom_top2']:.3f} at top 2 and {eval_rank['query']['gold_atom_top3']:.3f} at top 3.",
        "", "Component-oracle downstream exact match:", "",
        "| Component condition | Support / oracle query | Oracle support / query |",
        "|---|---:|---:|",
        f"| Fully frozen | {support_frozen['transition_set_exact_match']:.3f} | {query_frozen['transition_set_exact_match']:.3f} |",
        f"| Oracle atom assignment, frozen truth | {support_oracle_assignment['transition_set_exact_match']:.3f} | {query_oracle_assignment['transition_set_exact_match']:.3f} |",
        f"| Frozen atom assignment, oracle truth | {support_oracle_truth['transition_set_exact_match']:.3f} | {query_oracle_truth['transition_set_exact_match']:.3f} |",
        "", "## Interpretation", "",
        "Support certainty is the immediate integration bottleneck, but both atom alignment and held-out",
        "truth phrasing contribute. A probabilistic support experiment is justified only as a controlled",
        "decomposition: it should preregister small top-k branch budgets and require target retention gains",
        "without accepting broad answer sets as success. Query grounding must remain a separate reported ceiling.",
        "No LoRA, final suite, grammar expansion, or joint neural challenger is authorized by this result.",
        "", "The first evaluation attempt aborted before any prediction because the installed scikit-learn",
        "required an explicit one-vs-rest wrapper for multiclass liblinear. V22r2a locked that",
        "nondiscretionary compatibility amendment before the single completed replacement run.", "",
    ])
    markdown_path = (PROJECT_ROOT / args.markdown).resolve()
    markdown_path.write_text("\n".join(lines))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
