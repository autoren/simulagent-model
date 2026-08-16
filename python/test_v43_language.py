import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v43_language import (
    action_ontology,
    compile_action_sequence,
    compile_state,
    episode_aliases,
    operator_ontology,
    predicate_ontology,
    public_entities,
    render_action_sequence,
    render_state,
    safety_challenges,
    evaluate_safety_challenge,
)


class V43LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((PROJECT_ROOT / "data/v42-sequential-state-foundation/development_fit.jsonl").read_text().splitlines()[0])
        cls.aliases = episode_aliases(cls.source)
        cls.predicate = predicate_ontology(cls.source["id"])
        cls.operator, cls.operator_cues = operator_ontology(cls.source["id"])
        cls.action, cls.action_cues = action_ontology(cls.source["id"])

    def test_state_round_trip_including_unknown(self):
        query = next(row for row in self.source["agent_input"]["queries"] if row["partial_initial_state"])
        entities = public_entities(query["entities"], self.aliases)
        public, reference = render_state(query["initial_state"], self.aliases, self.predicate, self.operator_cues, "test-state")
        compiled = compile_state(public, entities, self.predicate, self.operator)
        self.assertEqual("ok", compiled["status"])
        self.assertEqual(reference["epistemic_state"], compiled["epistemic_state"])

    def test_action_round_trip_preserves_order(self):
        query = self.source["agent_input"]["queries"][0]
        entities = public_entities(query["entities"], self.aliases)
        public, reference = render_action_sequence(query["actions"], self.aliases, self.action_cues, "test-action")
        compiled = compile_action_sequence(public, entities, self.action)
        self.assertEqual("ok", compiled["status"])
        self.assertEqual(reference["actions"], compiled["actions"])

    def test_safety_challenges_fail_closed(self):
        catalog = [{"id": value, "entity_type": "unit"} for value in sorted(self.aliases.values())]
        challenges = safety_challenges(catalog[:2], self.aliases, self.predicate, self.operator, self.operator_cues, self.action, self.action_cues)
        self.assertEqual(7, len(challenges))
        self.assertTrue(all(evaluate_safety_challenge(row, catalog[:2], self.predicate, self.operator, self.action) for row in challenges))

    def test_fresh_episode_lexicons_differ(self):
        self.assertNotEqual(predicate_ontology("episode-a"), predicate_ontology("episode-b"))
        self.assertNotEqual(action_ontology("episode-a"), action_ontology("episode-b"))


if __name__ == "__main__":
    unittest.main()
