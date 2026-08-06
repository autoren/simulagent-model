import unittest

from select_v3_count_checkpoints import apply_gate, select


def metric(step: int, balanced: float, f1: float = 0.4) -> dict:
    return {
        "checkpoint_step": step,
        "balanced_identifiability_accuracy": balanced,
        "ambiguity_detection": {"f1": f1, "recall": 0.5, "tn": 10},
        "macro_accuracy_by_observed_gold_count": 0.3,
        "predicted_count_distribution": {"1": 10, "2": 10},
    }


class V3CheckpointSelectionTests(unittest.TestCase):
    def test_selects_balanced_accuracy_before_loss_independent_ties(self) -> None:
        selected = select([metric(100, 0.53), metric(200, 0.57), metric(300, 0.55)])
        self.assertEqual(selected["checkpoint_step"], 200)

    def test_gate_requires_reproducible_multiseed_signal(self) -> None:
        self.assertTrue(apply_gate([metric(100, 0.56), metric(200, 0.57), metric(300, 0.55)])["passed"])
        self.assertFalse(apply_gate([metric(100, 0.60), metric(200, 0.50), metric(300, 0.49)])["passed"])


if __name__ == "__main__":
    unittest.main()
