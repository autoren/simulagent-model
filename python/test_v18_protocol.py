import json
import unittest
from pathlib import Path

from audit_v18_benchmark import audit, read_records
from run_v18_schema_baselines import (
    balanced_accuracy,
    conditional_support_union_answer,
    decision_tree_answers,
    empirical_lookup_answer,
    episode_summary,
    set_f1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V18ProtocolTests(unittest.TestCase):
    def test_generated_development_corpus_passes_firewall_and_topology_audit(self) -> None:
        dataset = PROJECT_ROOT / "data/v18"
        config = json.loads((PROJECT_ROOT / "configs/dataset.v18.json").read_text())
        manifest = json.loads((dataset / "manifest.json").read_text())
        result = audit(read_records(dataset), config, manifest, dataset)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["underidentified_schemas"], 0)
        self.assertEqual(result["agent_input_forbidden_keys"], {})
        self.assertEqual(result["training_signature_overlap"]["determinant_vocabulary"], 8)

    def test_lookup_is_conservative_on_an_unseen_assignment(self) -> None:
        answer = empirical_lookup_answer({0: "transition_00"}, [1], 2)
        self.assertFalse(answer["identifiable"])
        self.assertEqual(len(answer["possible_transition_codes"]), 4)

    def test_conditional_union_uses_compatible_observations_only(self) -> None:
        answer = conditional_support_union_answer(
            {0: "transition_00", 1: "transition_11"}, [1, 3], 2
        )
        self.assertTrue(answer["identifiable"])
        self.assertEqual(answer["possible_transition_codes"], ["transition_11"])

    def test_fixed_depth_tree_interpolates_visible_outcome_bits(self) -> None:
        support = [
            {"assignment": {"a": False, "b": False}, "transition_code": "transition_00"},
            {"assignment": {"a": True, "b": False}, "transition_code": "transition_10"},
            {"assignment": {"a": False, "b": True}, "transition_code": "transition_01"},
            {"assignment": {"a": True, "b": True}, "transition_code": "transition_11"},
        ]
        answers = decision_tree_answers(support, ("a", "b"), [[3]], 2)
        self.assertEqual(answers[0]["possible_transition_codes"], ["transition_11"])

    def test_episode_summary_uses_episode_as_the_independent_unit(self) -> None:
        rows = [
            {"episode_id": "a", "transition_set_exact": True},
            {"episode_id": "a", "transition_set_exact": False},
            {"episode_id": "b", "transition_set_exact": True},
        ]
        result = episode_summary(rows)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["complete_episodes"], 1)
        self.assertEqual(result["episode_macro_transition_set_exact_match"], 0.75)

    def test_metric_helpers_cover_imbalanced_labels_and_empty_sets(self) -> None:
        self.assertEqual(balanced_accuracy([False, False, True], [False, False, False]), 0.5)
        self.assertEqual(set_f1(set(), set()), 1.0)
        self.assertAlmostEqual(set_f1({"a", "b"}, {"b", "c"}), 0.5)


if __name__ == "__main__":
    unittest.main()
