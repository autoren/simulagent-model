from __future__ import annotations

import unittest

from v223r1_outcome_verification_repair import failed_checks, positive_outcome_matches


class V223R1OutcomeVerificationRepairTests(unittest.TestCase):
    def test_failed_checks_is_exact_and_sorted(self) -> None:
        self.assertEqual(["a", "c"], failed_checks({"checks": {"c": False, "b": True, "a": False}}))

    def test_positive_outcome_match(self) -> None:
        invariant = {
            "expectedV223ScientificPassed": True,
            "expectedV223Branch": "PASS",
            "expectedSelectedCandidateIds": ["X"],
            "expectedSourceUnitCount": 4,
            "expectedFrozenUrlAttemptCount": 21,
            "expectedSuccessfulUrlCount": 21,
            "expectedEligibleCandidateCount": 1,
            "expectedFormalTaskRecordBodyReadCount": 0,
        }
        summary = {
            "audit": {"passed": True, "branch": "PASS"},
            "metrics": {
                "source_unit_count": 4,
                "frozen_url_attempt_count": 21,
                "successful_url_count": 21,
                "eligible_source_specific_candidate_count": 1,
                "formal_task_record_body_read_count": 0,
            },
        }
        result = {"passed": True, "branch": "PASS", "selected_source_specific_candidate_ids": ["X"]}
        self.assertTrue(positive_outcome_matches(summary, result, invariant))
        result["selected_source_specific_candidate_ids"] = ["Y"]
        self.assertFalse(positive_outcome_matches(summary, result, invariant))


if __name__ == "__main__":
    unittest.main()

