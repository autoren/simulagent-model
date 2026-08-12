import json
import unittest
from pathlib import Path

import numpy as np

from audit_v18_benchmark import read_records
from audit_v19_compatibility import prompt_inventory, read_scenes
from evaluate_v19_frozen_integration import evaluate_condition, frozen_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V19ProtocolTests(unittest.TestCase):
    def test_two_views_cover_identical_latent_items_without_agent_target_leakage(self) -> None:
        scenes = read_scenes(PROJECT_ROOT / "data/v19")
        by_view = {
            view: {(value["episode_id"], value["source_item_id"]) for value in scenes if value["view"] == view}
            for view in ("supported", "novel_ontology")
        }
        self.assertEqual(by_view["supported"], by_view["novel_ontology"])
        self.assertEqual(len(by_view["supported"]), 3456)
        for scene in scenes:
            rendered = json.dumps(scene["agent_input"], sort_keys=True)
            self.assertNotIn("allowed_values", rendered)
            self.assertNotIn("assignment", rendered)
            self.assertNotIn("executable_schema", rendered)

    def test_prompt_deduplication_matches_the_pre_extraction_gate(self) -> None:
        scenes = read_scenes(PROJECT_ROOT / "data/v19")
        base, nli, _ = prompt_inventory(scenes)
        self.assertEqual(len(base), 240)
        self.assertEqual(len(nli), 480)
        audit = json.loads(
            (PROJECT_ROOT / "outputs/v19-frozen-integration/pre-extraction-audit.json").read_text()
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["prompt_inventory"]["new_model_forward_passes"], 720)

    def test_frozen_binary_pipeline_uses_locked_scaler_and_class_order(self) -> None:
        payload = {
            "x_scaler_mean": np.asarray([1.0]),
            "x_scaler_scale": np.asarray([2.0]),
            "x_coef": np.asarray([[1.0]]),
            "x_intercept": np.asarray([0.0]),
            "x_classes": np.asarray([False, True]),
        }
        values = np.asarray([[-1.0], [3.0]])
        self.assertEqual(frozen_pipeline(payload, "x", values).tolist(), [False, True])
        self.assertEqual(frozen_pipeline(payload, "x", values, decision=True).tolist(), [-1.0, 1.0])

    def test_oracle_condition_reproduces_one_v18_episode_exactly(self) -> None:
        episodes = read_records(PROJECT_ROOT / "data/v18")
        episode = next(value for value in episodes if value["split"] == "development")
        result = evaluate_condition([episode], {}, "oracle", "oracle")
        self.assertEqual(result["episode_metrics"]["complete_episodes"], 1)
        self.assertEqual(result["schema_recovery"]["target_retention_rate"], 1.0)
        self.assertEqual(result["schema_recovery"]["empty_version_space_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
