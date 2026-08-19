from __future__ import annotations

import unittest

from v113_known_disagreement_rescue import apply_rule, enumerate_rules


CONFIG = {
    "thresholdGrids": {
        "proposalScore": [0.0, 0.5], "scoreGap": [0.1, 1.01],
        "confidence": [0.75, 1.01],
    },
    "registeredRuleFamilies": [
        "no_rescue", "accept_all", "same_scenario_only", "proposal_score_at_least",
        "proposal_score_and_gap", "proposal_score_and_same_scenario",
        "confidence_at_least", "confidence_and_same_scenario",
        "proposal_score_confidence_and_same_scenario",
    ],
}


class V113RescueTests(unittest.TestCase):
    def test_rules_are_unique_and_cover_exact_families(self) -> None:
        rules = enumerate_rules(CONFIG)
        self.assertEqual({row["family"] for row in rules}, set(CONFIG["registeredRuleFamilies"]))
        self.assertEqual(len({str(sorted(row.items())) for row in rules}), len(rules))

    def test_only_eligible_disagreements_can_be_rescued(self) -> None:
        rule = {"family": "accept_all", "complexity": 1}
        feature = {
            "eligible": False, "proposed_intent_score": 1.0, "score_gap": 0.0,
            "llm_confidence": 1.0, "proposed_and_nearest_same_scenario": True,
        }
        self.assertFalse(apply_rule(rule, feature))

    def test_conjunctive_rule_requires_every_condition(self) -> None:
        rule = {
            "family": "proposal_score_confidence_and_same_scenario",
            "minimum_score": 0.5, "minimum_confidence": 0.75, "complexity": 3,
        }
        feature = {
            "eligible": True, "proposed_intent_score": 0.6, "score_gap": 0.1,
            "llm_confidence": 0.9, "proposed_and_nearest_same_scenario": True,
        }
        self.assertTrue(apply_rule(rule, feature))
        feature["proposed_and_nearest_same_scenario"] = False
        self.assertFalse(apply_rule(rule, feature))


if __name__ == "__main__":
    unittest.main()
