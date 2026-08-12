"""Audit the unmaterialized V21 design using an ineligible test seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_v18_benchmark import read_records
from v10_protocol import file_sha256
from v21_final_suite import generate_suite, structural_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SEED = "V21_DESIGN_AUDIT_TEST_SEED_NEVER_FINAL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v21-multimechanic-final.json")
    parser.add_argument("--output", default="outputs/v21-design/pre-materialization-audit.json")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text())
    v19_lock_path = PROJECT_ROOT / "configs/v19-frozen-integration-lock.json"
    v19_lock = json.loads(v19_lock_path.read_text())
    v18_records = read_records(PROJECT_ROOT / v19_lock["source"]["v18_dataset"])
    excluded = {
        2: {tuple(value["target"]["behavioral_signature"]) for value in v18_records},
    }
    episodes, scenes = generate_suite(config, TEST_SEED, excluded)
    summary = structural_summary(episodes, scenes)
    final_signatures = {tuple(value["target"]["behavioral_signature"]) for value in episodes}
    target_overlap = len(final_signatures & excluded[2])
    pairing = {}
    for scene in scenes:
        key = (scene["episode_id"], scene["item_kind"], scene["source_item_id"])
        pairing.setdefault(key, set()).add(scene["view"])
    forbidden = (
        "executable_schema", "behavioral_signature", "relevant_determinants",
        "action_dependency_schema",
    )
    leaked_inputs = sum(
        any(name in json.dumps(value["agent_input"], sort_keys=True) for name in forbidden)
        for value in [*episodes, *scenes]
    )
    checks = {
        "exactly_40_episodes": summary["episodes"] == 40,
        "family_quota_exact": summary["family_counts"] == {
            family: values["episodes"] for family, values in config["constructionFamilies"].items()
        },
        "outcome_bits_balanced": summary["outcome_bit_counts"] == {"1": 20, "2": 20},
        "injectivity_balanced": summary["injectivity_counts"] == {
            "injective": 20, "non_injective": 20,
        },
        "unique_behavior_per_mechanic": len(final_signatures) == 40,
        "zero_v18_behavior_overlap": target_overlap == 0,
        "all_views_paired": all(views == {"supported", "novel_ontology"} for views in pairing.values()),
        "all_nine_surfaces_present": set(summary["surface_counts"]) == set(config["language"]["surfaceFamilies"]),
        "all_three_operators_present": set(summary["semantic_operator_counts"]) == set(config["language"]["semanticOperators"]),
        "all_unresolved_modes_present": set(summary["unresolved_mode_counts"]) == set(config["language"]["unresolvedModes"]),
        "zero_target_input_leaks": leaked_inputs == 0,
        "final_dataset_absent": not (PROJECT_ROOT / "data/v21-final").exists(),
        "final_seed_absent": not (PROJECT_ROOT / "outputs/v21-final/seed-draw.json").exists(),
        "v20_evaluation_absent": not (PROJECT_ROOT / "outputs/v20-probabilistic-interface/evaluation").exists(),
        "delayed_seed_policy": config["seedPolicy"]["kind"] == "delayed_os_csprng_after_execution_lock",
        "test_seed_ineligible": config["seedPolicy"]["testSeedsAreNeverEligible"] is True,
        "one_construction_and_no_retry": (
            config["limits"]["finalSuiteConstructionsPermitted"] == 1
            and config["limits"]["retriesPermitted"] == 0
        ),
    }
    report = {
        "schema_version": 21,
        "experiment": "v21_unmaterialized_design_audit",
        "config": args.config,
        "config_sha256": file_sha256(config_path),
        "test_seed": TEST_SEED,
        "test_seed_final_eligible": False,
        "summary": summary,
        "v18_behavior_overlap": target_overlap,
        "leaked_agent_inputs": leaked_inputs,
        "checks": checks,
        "passed": all(checks.values()),
        "data_access": {
            "v18_development_records_read": len(v18_records),
            "v20_results_read": 0,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "final_records_created_or_read": 0,
            "model_forward_passes": 0,
        },
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
