from __future__ import annotations

import unittest

from v216r1_negative_outcome_verification_repair import failed_scientific_checks, negative_outcome_matches


class V216R1NegativeOutcomeRepairTests(unittest.TestCase):
    def test_failed_checks_are_exact_and_sorted(self) -> None:
        self.assertEqual(["b"], failed_scientific_checks({"checks": {"a": True, "b": False}}))

    def test_negative_match_does_not_turn_failure_into_pass(self) -> None:
        invariant = {
            "expectedV216ScientificPassed": False,
            "expectedV216Branch": "NEG",
            "expectedV216Decision": "freeze-negative",
            "expectedFailedScientificChecks": ["size"],
            "expectedOlderTermCount": 1,
            "expectedNewerTermCount": 2,
            "expectedEligibleRecordCount": 3,
            "expectedDevelopmentGroupCount": 2,
            "expectedProtectedGroupCount": 1,
        }
        summary = {
            "audit": {"passed": False, "branch": "NEG", "decision": "freeze-negative", "checks": {"size": False, "hash": True}},
            "metrics": {"older_term_count": 1, "newer_term_count": 2, "eligible_record_count": 3, "development_group_count": 2, "protected_group_count": 1},
        }
        result = {"passed": False, "branch": "NEG", "decision": "freeze-negative"}
        self.assertTrue(negative_outcome_matches(summary, result, invariant))
        result["passed"] = True
        self.assertFalse(negative_outcome_matches(summary, result, invariant))


if __name__ == "__main__":
    unittest.main()

