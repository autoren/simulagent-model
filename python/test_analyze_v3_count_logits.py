import unittest

from analyze_v3_count_logits import analyze


class V3LogitDiagnosticTests(unittest.TestCase):
    def test_detects_perfect_ranking_even_when_argmax_is_always_one(self) -> None:
        gold = {"ambiguous": True, "identifiable": False}
        rows = [
            {
                "id": "ambiguous",
                "candidate_logits": {"1": 2.0, "2": 1.9, "3": 0.0, "4": 0.0, "5": 0.0},
            },
            {
                "id": "identifiable",
                "candidate_logits": {"1": 2.0, "2": 1.0, "3": 0.0, "4": 0.0, "5": 0.0},
            },
        ]
        report = analyze(gold, rows)
        self.assertEqual(report["roc_auc"], 1.0)
        self.assertEqual(
            report["posthoc_best_same_validation_threshold"][
                "balanced_identifiability_accuracy"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
