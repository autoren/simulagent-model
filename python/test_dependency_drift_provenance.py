from __future__ import annotations

import unittest

from diagnose_dependency_drift import diagnose


class DependencyDriftProvenanceTests(unittest.TestCase):
    def test_all_drift_is_accounted_for_without_guessing(self) -> None:
        addendum = diagnose()
        self.assertEqual(addendum["finding_count"], 8)
        self.assertTrue(addendum["all_outcome_payloads_valid"])
        self.assertEqual(addendum["exact_frozen_dependency_recovery_count"], 0)
        self.assertEqual(addendum["current_dirty_dependency_count"], 0)
        self.assertEqual(
            {row["classification"] for row in addendum["findings"]},
            {"narrative_results_document", "executable_outcome_verifier", "executable_auditor"},
        )
        for row in addendum["findings"]:
            self.assertEqual(row["resolution"], "do_not_guess_or_overwrite_preserve_append_only_addendum")
            self.assertEqual(len(row["reachable_or_reflog_versions"]), 1)
            self.assertFalse(row["reachable_or_reflog_versions"][0]["matches_frozen_expected_sha256"])
            self.assertTrue(row["content_diff_diagnosis"]["current_matches_every_reachable_or_reflog_version"])
            self.assertEqual(row["content_diff_diagnosis"]["current_vs_reachable_changed_byte_count"], 0)
            self.assertEqual(row["content_diff_diagnosis"]["current_vs_reachable_changed_line_count"], 0)
            self.assertFalse(row["content_diff_diagnosis"]["frozen_expected_content_available_for_direct_diff"])


if __name__ == "__main__":
    unittest.main()
