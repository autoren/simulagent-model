import json
import unittest
from pathlib import Path

from v21_final_suite import (
    SEMANTIC_OPERATOR, SURFACES, generate_suite, structural_summary,
)


class V21FinalSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v21-multimechanic-final.json").read_text())
        cls.episodes, cls.scenes = generate_suite(cls.config, "TEST_SEED_NEVER_FINAL")

    def test_population_strata_and_unique_behaviors(self):
        summary = structural_summary(self.episodes, self.scenes)
        self.assertEqual(summary["episodes"], 40)
        self.assertEqual(summary["family_counts"], {
            "primitive_one_bit": 8,
            "composed_one_bit": 12,
            "factorized_two_bit": 12,
            "nested_two_bit": 8,
        })
        self.assertEqual(summary["outcome_bit_counts"], {"1": 20, "2": 20})
        self.assertEqual(summary["injectivity_counts"], {"injective": 20, "non_injective": 20})
        signatures = [tuple(value["target"]["behavioral_signature"]) for value in self.episodes]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_language_views_are_exactly_paired(self):
        by_key = {}
        for scene in self.scenes:
            key = (scene["episode_id"], scene["item_kind"], scene["source_item_id"])
            by_key.setdefault(key, {})[scene["view"]] = scene
        self.assertTrue(by_key)
        for pair in by_key.values():
            self.assertEqual(set(pair), {"supported", "novel_ontology"})
            supported = pair["supported"]
            novel = pair["novel_ontology"]
            self.assertEqual(
                [value["allowed_values"] for value in supported["target"]["determinant_grounding"]],
                [value["allowed_values"] for value in novel["target"]["determinant_grounding"]],
            )
            self.assertEqual(supported.get("observed_transition_code"), novel.get("observed_transition_code"))

    def test_all_registered_surfaces_and_uncertainty_modes_occur(self):
        summary = structural_summary(self.episodes, self.scenes)
        self.assertEqual(set(summary["surface_counts"]), set(SURFACES))
        self.assertEqual(set(summary["semantic_operator_counts"]), set(SEMANTIC_OPERATOR.values()))
        self.assertEqual(set(summary["unresolved_mode_counts"]), {"unknown", "stale", "conflicting"})

    def test_agent_inputs_do_not_expose_target_programs(self):
        forbidden = {
            "executable_schema", "behavioral_signature", "relevant_determinants",
            "action_dependency_schema", "target",
        }
        for value in [*self.episodes, *self.scenes]:
            serialized = json.dumps(value["agent_input"], sort_keys=True)
            self.assertFalse(any(key in serialized for key in forbidden))

    def test_one_step_scope_is_explicit(self):
        for episode in self.episodes:
            self.assertNotIn("next_state", json.dumps(episode["agent_input"]))
            self.assertIn(episode["agent_input"]["dsl_contract"]["outcome_bits"], (1, 2))


if __name__ == "__main__":
    unittest.main()
