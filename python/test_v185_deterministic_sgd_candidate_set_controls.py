from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v185_deterministic_sgd_candidate_set_controls import normalize_text, threshold_set


class V185DeterministicCandidateSetControlsTest(unittest.TestCase):
    def test_threshold_set_is_complete_below_threshold_and_set_valued_on_tie(self) -> None:
        scores = {"K01": 0.2, "K02": 0.2, "K03": 0.1}
        self.assertEqual(threshold_set(scores, 0.3, 0.0), ["K01", "K02", "K03"])
        self.assertEqual(threshold_set(scores, 0.1, 0.0), ["K01", "K02"])

    def test_normalization_is_stable(self) -> None:
        self.assertEqual(normalize_text("  BUY—Tickets! "), "buy tickets")

    def test_model_and_protected_access_are_closed(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/v185-deterministic-sgd-candidate-set-controls.json").read_text()
        )
        self.assertTrue(config["deterministicViews"]["allCandidateSetsAreShadowOnly"])
        self.assertTrue(config["trustedClarificationPolicy"]["retrievalNeverDeterminesTerminalState"])
        self.assertFalse(config["decisionRule"]["passAuthorizesImmediateModelRun"])
        self.assertFalse(config["decisionRule"]["passAuthorizesProtectedAccess"])


if __name__ == "__main__":
    unittest.main()
