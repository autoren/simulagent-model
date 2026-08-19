import unittest

from v118_evidence_identifiability_audit import (
    posterior_threshold_envelopes, required_bayes_factor, supplemental_unsupported_factor,
)


class V118EvidenceIdentifiabilityTests(unittest.TestCase):
    def test_threshold_envelopes(self):
        thresholds = posterior_threshold_envelopes()
        self.assertAlmostEqual(thresholds["exact_candidate"]["minimum"], 17 / 19)
        self.assertAlmostEqual(thresholds["exact_candidate"]["maximum"], 10 / 11)
        self.assertAlmostEqual(thresholds["unsupported"]["minimum"], 5 / 7)
        self.assertAlmostEqual(thresholds["unsupported"]["maximum"], 5 / 6)

    def test_uniform_candidate_bayes_factor(self):
        self.assertAlmostEqual(required_bayes_factor(1 / 17, 17 / 19), 136.0)
        self.assertAlmostEqual(required_bayes_factor(1 / 17, 10 / 11), 160.0)

    def test_supplemental_factor_is_one_when_unsupported_already_dominates(self):
        self.assertEqual(supplemental_unsupported_factor({
            "KNOWN": 0.10, "NOVEL": 0.02, "UNSUPPORTED": 0.87, "ABSTAIN": 0.01,
        }), 1.0)


if __name__ == "__main__":
    unittest.main()
