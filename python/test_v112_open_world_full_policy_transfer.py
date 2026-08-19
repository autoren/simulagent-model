from __future__ import annotations

import unittest

from v112_open_world_full_policy_transfer import (
    novelty_evidence_metrics, policy_prediction, population_gates, select_fresh_population,
)


CONFIG = {
    "freshPopulation": {
        "sourcePartition": "validation", "baseSalt": "test", "classes": ["known", "novel"],
        "recordCountPerClass": 2, "requiredScenarioCoverage": {"known": 2, "novel": 1},
    },
    "frozenPolicy": {
        "knownActionConfidence": 0.9, "unsupportedActionConfidence": 0.8,
        "abstainActionConfidence": 0.0, "positiveNovelEvidenceProbability": 0.75,
        "negativeNovelEvidenceProbability": 0.15,
    },
}


class V112FullPolicyTests(unittest.TestCase):
    def test_fresh_population_excludes_prior_and_covers_scenarios(self) -> None:
        rows = []
        for index, (label, scenario) in enumerate((
            ("known", "a"), ("known", "b"), ("known", "a"),
            ("novel", "c"), ("novel", "c"), ("novel", "c"),
        )):
            rows.append({
                "candidate_id": f"c{index}", "source_id": str(index), "partition": "validation",
                "class_label": label, "scenario": scenario, "intent": f"i{index}",
                "current_utterance_intent_overlap_count": 0, "slot_type_count": 0,
            })
        excluded = {"selected_population": [{"candidate_id": "c0"}]}
        population = select_fresh_population({"candidate_index": rows}, excluded, CONFIG)
        self.assertTrue(all(population_gates(population, CONFIG).values()))
        self.assertNotIn("c0", {row["candidate_id"] for row in population["selected_population"]})

    def test_policy_treats_abstain_as_evidence_but_still_asks(self) -> None:
        direct = {"status": "ABSTAIN", "known_intent": None, "novel_scenario": None, "confidence": 0.9}
        action, evidence = policy_prediction(direct, {"nearest_intent": "x"}, CONFIG)
        self.assertEqual(action["status"], "ABSTAIN")
        self.assertTrue(evidence["novel_candidate"])
        self.assertFalse(evidence["capability_defined"])
        self.assertFalse(evidence["executable"])

    def test_known_requires_exact_retrieval_agreement(self) -> None:
        direct = {"status": "KNOWN", "known_intent": "x", "novel_scenario": None, "confidence": 0.9}
        accepted, _ = policy_prediction(direct, {"nearest_intent": "x"}, CONFIG)
        rejected, _ = policy_prediction(direct, {"nearest_intent": "y"}, CONFIG)
        self.assertEqual(accepted["status"], "KNOWN")
        self.assertEqual(rejected["status"], "ABSTAIN")

    def test_novelty_metrics_are_separate_from_action(self) -> None:
        records = [
            {"record_id": "n", "class_label": "novel_valid"},
            {"record_id": "k", "class_label": "known_familiar"},
        ]
        evidence = {
            "n": {"novel_candidate": True, "novel_evidence_probability": 0.75},
            "k": {"novel_candidate": False, "novel_evidence_probability": 0.15},
        }
        metrics = novelty_evidence_metrics(records, evidence)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
