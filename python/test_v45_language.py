import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v43_language import compile_state, operator_ontology, predicate_ontology, public_entities, render_state
from v43r1_measurement import graph_equal
from v45_language import (
    action_ontology,
    compile_action_sequence,
    episode_aliases,
    evaluate_safety_challenge,
    render_action_sequence,
    safety_challenges,
)


class V45LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((PROJECT_ROOT / "data/v44-deterministic-delayed-effects/development_fit.jsonl").read_text().splitlines()[0])
        cls.aliases = episode_aliases(cls.source)
        cls.predicate = predicate_ontology(f"v45|{cls.source['id']}")
        cls.operator, cls.operator_cues = operator_ontology(f"v45|{cls.source['id']}")
        cls.action, cls.action_cues = action_ontology(cls.source["id"])

    def test_state_round_trip_uses_canonical_graph_comparison(self):
        query = next(row for row in self.source["agent_input"]["queries"] if row["partial_initial_state"])
        entities = public_entities(query["entities"], self.aliases)
        public, reference = render_state(query["initial_state"], self.aliases, self.predicate, self.operator_cues, "v45-test-state")
        compiled = compile_state(public, entities, self.predicate, self.operator)
        self.assertEqual("ok", compiled["status"])
        self.assertTrue(graph_equal(compiled["epistemic_state"], list(reversed(reference["epistemic_state"]))))

    def test_action_round_trip_preserves_wait_placement(self):
        query = self.source["agent_input"]["queries"][0]
        entities = public_entities(query["entities"], self.aliases)
        public, reference = render_action_sequence(query["actions"], self.aliases, self.action_cues, "v45-test-action")
        compiled = compile_action_sequence(public, entities, self.action)
        self.assertEqual("ok", compiled["status"])
        self.assertEqual(reference["actions"], compiled["actions"])
        self.assertEqual(reference["command_kinds"], compiled["command_kinds"])
        self.assertIn("wait", compiled["command_kinds"])

    def test_wait_with_arguments_abstains(self):
        entities = [{"id": value, "entity_type": "unit"} for value in sorted(self.aliases.values())]
        text = f"Step 1: {entities[0]['id']} performs {self.action_cues['wait']} toward {entities[1]['id']}."
        self.assertNotEqual("ok", compile_action_sequence({"action_sequence_text": text}, entities, self.action)["status"])

    def test_all_registered_safety_classes_fail_closed(self):
        entities = [{"id": value, "entity_type": "unit"} for value in sorted(self.aliases.values())][:2]
        challenges = safety_challenges(entities, self.predicate, self.operator, self.operator_cues, self.action, self.action_cues)
        self.assertEqual(9, len(challenges))
        self.assertTrue(all(evaluate_safety_challenge(row, entities, self.predicate, self.operator, self.action) for row in challenges))


if __name__ == "__main__":
    unittest.main()
