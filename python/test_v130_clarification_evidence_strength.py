import json
import unittest
from pathlib import Path

from v130_clarification_evidence_strength import joint_answer_distribution, reliability_grid, run_audit


class V130EvidenceStrengthTests(unittest.TestCase):
    def test_grid_is_exact(self):
        config = json.loads(Path("configs/v130-clarification-evidence-strength.json").read_text())
        grid = reliability_grid(config["singleAnswerReliabilityGrid"])
        self.assertEqual((len(grid), grid[0], grid[-1]), (101, 0.95, 1.0))

    def test_joint_channel_preserves_marginals(self):
        ids = ["a", "b", "c"]
        joint = joint_answer_distribution("a", "b", 0.9, "symmetric", 2, 0.5, ids, "c", 0.75)
        self.assertAlmostEqual(sum(joint.values()), 1.0)
        self.assertAlmostEqual(sum(value for answers, value in joint.items() if answers[0] == "a"), 0.9)
        self.assertAlmostEqual(sum(value for answers, value in joint.items() if answers[1] == "a"), 0.9)

    def test_audit_is_aggregate_and_complete(self):
        config = json.loads(Path("configs/v130-clarification-evidence-strength.json").read_text())
        catalog = json.loads(Path(config["choiceCatalog"]).read_text())
        baseline = json.loads(Path(config["baselineConfig"]).read_text())
        result = run_audit(catalog, baseline, config)
        self.assertTrue(all(result["audit_checks"].values()))
        self.assertEqual(len(result["single_answer_thresholds"]), 9)
        self.assertEqual(len(result["multi_answer_conditions"]), 27)
        self.assertNotIn("pairs", result)


if __name__ == "__main__": unittest.main()
