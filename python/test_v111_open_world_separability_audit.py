from __future__ import annotations

import unittest

from v111_open_world_separability_audit import (
    apply_rule, binary_metrics, enumerate_rules, select_rule,
)


CONFIG = {
    "thresholdGrids": {"score": [0.0, 0.5, 1.01], "margin": [0.0, 0.2], "proposal": [0.0, 0.5]},
    "registeredRuleFamilies": [
        "top_score_band", "low_intent_margin", "top_score_band_and_low_margin",
        "llm_known_proposal_score_below", "llm_known_retrieval_disagreement",
        "llm_nonknown_top_score_band", "not_llm_unsupported_top_score_band",
        "llm_known_top_score_band_and_low_margin", "top_score_band_or_known_disagreement",
        "llm_abstain_only",
    ],
    "calibrationSelection": {
        "minimumNovelPrecisionForFeasibleCandidate": 0.7,
        "minimumNovelRecallForFeasibleCandidate": 0.5,
        "maximumNonNovelFalsePositiveRateForFeasibleCandidate": 0.1,
    },
}


def feature(identifier: str, novel: bool, score: float, margin: float) -> dict:
    return {
        "record_id": identifier, "class_label": "novel_valid" if novel else "known_familiar",
        "is_novel": novel, "top_intent_score": score, "intent_margin": margin,
        "proposed_intent_score": score, "llm_is_known": not novel,
        "llm_is_abstain": novel, "llm_is_unsupported": False,
        "llm_retrieval_intent_disagree": False,
    }


class V111SeparabilityTests(unittest.TestCase):
    def test_registered_rule_families_enumerate_exactly(self) -> None:
        rules = enumerate_rules(CONFIG)
        self.assertEqual({row["family"] for row in rules}, set(CONFIG["registeredRuleFamilies"]))
        self.assertEqual(len({str(sorted(row.items())) for row in rules}), len(rules))

    def test_band_rule_and_binary_metrics(self) -> None:
        rows = [feature("n", True, 0.3, 0.1), feature("k", False, 0.8, 0.4)]
        rule = {"family": "top_score_band", "low": 0.0, "high": 0.5, "complexity": 2}
        self.assertTrue(apply_rule(rule, rows[0]))
        self.assertFalse(apply_rule(rule, rows[1]))
        self.assertEqual(binary_metrics(rows, rule)["f1"], 1.0)

    def test_selection_uses_calibration_feasibility_first(self) -> None:
        rows = [
            feature("n1", True, 0.3, 0.1), feature("n2", True, 0.4, 0.1),
            feature("k1", False, 0.8, 0.4), feature("k2", False, 0.9, 0.4),
        ]
        selected = select_rule(enumerate_rules(CONFIG), rows, CONFIG)
        self.assertGreater(selected["feasible_candidate_count"], 0)
        self.assertTrue(selected["selected"]["feasible"])


if __name__ == "__main__":
    unittest.main()
