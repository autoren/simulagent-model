import unittest

from evaluate_v5_challenge_mlx import evidence_contrasts, surface_invariance


def row(
    pair: str,
    surface: str,
    gold: bool,
    score: float,
    evidence_pair: str | None = None,
) -> dict:
    return {
        "surface_pair_id": pair,
        "surface_variant": surface,
        "gold_ambiguous": gold,
        "score": score,
        "evidence_pair_id": evidence_pair,
    }


class V5ChallengeMetricTests(unittest.TestCase):
    def test_surface_invariance_counts_prediction_agreement(self) -> None:
        rows = [
            row("a", "canonical", False, -1.0),
            row("a", "entity_renamed", False, -0.5),
            row("a", "paraphrased", False, 0.5),
            row("b", "canonical", True, 1.0),
            row("b", "entity_renamed", True, 0.5),
            row("b", "paraphrased", True, 2.0),
        ]
        report = surface_invariance(rows, 0.0)
        self.assertEqual(report["surface_pairs"], 2)
        self.assertEqual(
            report["transformations"]["entity_renamed"]["prediction_agreement"], 1.0
        )
        self.assertEqual(
            report["transformations"]["paraphrased"]["prediction_agreement"], 0.5
        )

    def test_evidence_contrasts_measure_cross_label_ordering(self) -> None:
        rows = [
            row("a", "canonical", True, 2.0, "evidence-a"),
            row("b", "canonical", False, 1.0, "evidence-a"),
            row("c", "canonical", False, 3.0, "evidence-a"),
            row("a", "paraphrased", True, -2.0, "evidence-a"),
        ]
        report = evidence_contrasts(rows, 0.0)
        self.assertEqual(report["groups"], 1)
        self.assertEqual(report["cross_label_comparisons"], 2)
        self.assertEqual(report["directional_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
