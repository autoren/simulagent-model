"""Lock the one-field V21 prompt-budget amendment before any final seed draw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_v19_compatibility import prompt_inventory
from v10_protocol import file_sha256
from v21_final_suite import generate_suite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = (
    "COUNT_A_NEVER_FINAL", "COUNT_B_NEVER_FINAL",
    "COUNT_C_NEVER_FINAL", "COUNT_D_NEVER_FINAL",
)


def only_budget_changed(original: dict, amended: dict) -> bool:
    original = json.loads(json.dumps(original))
    amended = json.loads(json.dumps(amended))
    old = original["limits"].pop("maximumNewModelForwardPasses")
    new = amended["limits"].pop("maximumNewModelForwardPasses")
    return original == amended and old == 2000 and new == 5200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-lock", default="configs/v21-multimechanic-design-lock.json")
    parser.add_argument("--config", default="configs/v21r2-multimechanic-final.json")
    parser.add_argument("--amendment", default="docs/v21r2-forward-budget-amendment.md")
    parser.add_argument("--output", default="configs/v21r2-multimechanic-design-lock.json")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise RuntimeError(f"V21r2 amendment lock already exists: {output}")
    if (PROJECT_ROOT / "data/v21-final").exists() or (PROJECT_ROOT / "outputs/v21-final/seed-draw.json").exists():
        raise RuntimeError("V21 final data or seed exists before the resource amendment")
    base_path = PROJECT_ROOT / args.base_lock
    base = json.loads(base_path.read_text())
    for path, expected in base["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"Original locked V21 implementation changed: {path}")
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text())
    if not only_budget_changed(base["config"], config):
        raise RuntimeError("V21r2 changes more than the prompt-inference budget")
    counts = {}
    for seed in TEST_SEEDS:
        _, scenes = generate_suite(config, seed)
        base_prompts, nli_prompts, _ = prompt_inventory(scenes)
        counts[seed] = {
            "base_prompts": len(base_prompts),
            "nli_prompts": len(nli_prompts),
            "total": len(base_prompts) + len(nli_prompts),
        }
    if {value["total"] for value in counts.values()} != {5136}:
        raise RuntimeError("V21 test-seed prompt inventory is not the amended fixed count")
    v20_result_path = PROJECT_ROOT / "outputs/v20-probabilistic-interface/evaluation/result.json"
    v20_result = json.loads(v20_result_path.read_text())
    lock = {
        "schema_version": 21,
        "experiment": "v21r2_prompt_budget_only_design_amendment",
        "base_design_lock": args.base_lock,
        "base_design_lock_sha256": file_sha256(base_path),
        "config": config,
        "config_path": args.config,
        "config_sha256": file_sha256(config_path),
        "amendment": args.amendment,
        "amendment_sha256": file_sha256(PROJECT_ROOT / args.amendment),
        "implementation": base["implementation"],
        "source": base["source"],
        "seed_policy": base["seed_policy"],
        "limits": config["limits"],
        "prompt_inventory_test_seeds": counts,
        "test_seeds_final_eligible": False,
        "changed_fields": {
            "limits.maximumNewModelForwardPasses": {"from": 2000, "to": 5200},
        },
        "v20_result_known_at_amendment": True,
        "v20_result_sha256": file_sha256(v20_result_path),
        "v20_decision": v20_result["decision"],
        "data_access_before_amendment": {
            "v20_results_read": 1,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "final_seed_draws": 0,
            "final_records_created_or_read": 0,
            "final_model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": args.output,
        "lock_sha256": file_sha256(output),
        "changed_fields": lock["changed_fields"],
        "prompt_inventory": counts,
        "final_records": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
