import json
import unittest
from pathlib import Path

from v116_typed_clarification_voi import choice_maps
from v119_asymmetric_adaptive_evidence import joint_distribution, truths


class V119AdaptiveEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(Path("outputs/v109-open-world-typed-choice/interface/choice-catalog.json").read_text())
        cls.config = json.loads(Path("configs/v119-asymmetric-adaptive-evidence.json").read_text())
        _, cls.by_id, _ = choice_maps(cls.catalog)

    def test_truths(self):
        self.assertEqual(truths("K00", "K00", self.by_id), ("CONFIRM", "MATCH", "DECLARED"))
        self.assertEqual(truths("K01", "K00", self.by_id), ("REJECT", "MISMATCH", "DECLARED"))
        self.assertEqual(truths("U00", "K00", self.by_id), ("REJECT", "MISMATCH", "OUTSIDE_VISIBLE"))

    def test_all_joint_channels_normalize(self):
        for hypothesis in self.by_id:
            for reliability in (0.90, 0.95, 1.0):
                for rho in (0.0, 0.5, 1.0):
                    distribution = joint_distribution(hypothesis, "K00", reliability, rho, self.by_id, self.config)
                    self.assertAlmostEqual(sum(distribution.values()), 1.0)
                    self.assertTrue(all(value >= 0.0 for value in distribution.values()))


if __name__ == "__main__": unittest.main()
