import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from generate_v41_confirmation import build_population, old_program_keys
from v41_interface import assemble_epistemic_graph, compile_language_scene


class V41ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PROJECT_ROOT / "configs/v41-relational-mechanic-confirmation.json").read_text())
        cls.v22 = json.loads((PROJECT_ROOT / "configs/dataset.v22.json").read_text())
        cls.v32 = json.loads((PROJECT_ROOT / "configs/v32-factorized-semantics.json").read_text())
        cls.rows = build_population(cls.config, cls.v22, cls.v32)

    def test_population_is_forty_unseen_mechanics(self):
        self.assertEqual(len(self.rows), 40)
        keys = [row["target"]["program_key"] for row in self.rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(set(keys) & old_program_keys(self.config))

    def test_family_and_bit_quotas(self):
        for family in self.config["population"]["families"]:
            rows = [row for row in self.rows if row["construction_family"] == family]
            self.assertEqual(len(rows), 10)
            self.assertEqual(sum(row["oracle_metadata"]["outcome_bits"] == 1 for row in rows), 3)
            self.assertEqual(sum(row["oracle_metadata"]["outcome_bits"] == 2 for row in rows), 7)

    def test_first_scene_round_trip(self):
        row = self.rows[0]
        public = row["agent_input"]["support_traces"][0]
        reference = row["language_reference"]["support"][0]
        oracle = row["oracle_grounding"]["support"][0]
        compiled = compile_language_scene(public, self.v32)
        graph = assemble_epistemic_graph(public, compiled, reference["entity_alias_to_canonical"], self.v32)
        self.assertTrue(graph["complete"])
        self.assertEqual(graph["epistemic_state"], sorted(oracle["epistemic_state"], key=lambda item: item["atom"]))

    def test_agent_input_excludes_program_and_language_references(self):
        for row in self.rows:
            self.assertNotIn("target", row["agent_input"])
            self.assertNotIn("language_reference", row["agent_input"])


if __name__ == "__main__":
    unittest.main()
