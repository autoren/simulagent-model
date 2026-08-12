import json
import unittest
from pathlib import Path

from evaluate_v11_frozen_scale import scale_decision
from extract_v11_scale_features_mlx import evidence_from_base_text
from freeze_v11_frozen_scale import expected_layer


def gates(passed, oracle_failed=False, upstream_failed=False):
    return {
        "passed": passed,
        "checks": [
            {"name": "minimum_fold_oracle_polarity_accuracy", "passed": not oracle_failed},
            {"name": "minimum_fold_span_accuracy", "passed": not upstream_failed},
        ],
    }


class V11FrozenScaleTests(unittest.TestCase):
    def test_quarter_depth_maps_v10_layer_six_to_larger_layer_eight(self):
        rule = {
            "referenceLayers": 24,
            "referenceExtractionLayer": 6,
            "rounding": "nearest_integer",
        }
        self.assertEqual(expected_layer(24, rule), 6)
        self.assertEqual(expected_layer(32, rule), 8)

    def test_evidence_is_recovered_from_exact_v10_prompt_text(self):
        text = "Candidate action: act\nQueried determinant: {}\nEvidence excerpt: The report was rejected."
        self.assertEqual(evidence_from_base_text(text), "The report was rejected.")

    def test_scale_decision_requires_both_fixed_results(self):
        both_pass = {
            "qwen35_4b": {"primary_gates": gates(True)},
            "qwen35_9b": {"primary_gates": gates(True)},
        }
        self.assertEqual(scale_decision(both_pass), "transferable_polarity_emerges_by_4b_no_lora_authorized")
        only_nine = {
            "qwen35_4b": {"primary_gates": gates(False, oracle_failed=True)},
            "qwen35_9b": {"primary_gates": gates(True)},
        }
        self.assertEqual(scale_decision(only_nine), "transferable_polarity_emerges_at_9b_prefer_9b_grounding")
        neither = {
            "qwen35_4b": {"primary_gates": gates(False, oracle_failed=True)},
            "qwen35_9b": {"primary_gates": gates(False, oracle_failed=True)},
        }
        self.assertEqual(scale_decision(neither), "frozen_scale_insufficient_test_nonlinear_token_aware_readout")

    def test_config_pins_both_models_and_homologous_layers(self):
        config = json.loads(Path("configs/v11-frozen-scale.json").read_text())
        self.assertEqual(config["runOrder"], ["qwen35_4b", "qwen35_9b"])
        self.assertEqual({value["extractionLayer"] for value in config["models"].values()}, {8})
        self.assertTrue(all(len(value["revision"]) == 40 for value in config["models"].values()))


if __name__ == "__main__":
    unittest.main()
