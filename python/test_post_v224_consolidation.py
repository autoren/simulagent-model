from __future__ import annotations

import unittest

from cross_track_evidence_audit import payload_hash
from post_v224_consolidation import build_consolidation


class PostV224ConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = build_consolidation()

    def test_all_frozen_gates_pass(self) -> None:
        self.assertTrue(self.bundle["result"]["passed"])
        self.assertTrue(all(self.bundle["result"]["gates"].values()))

    def test_provenance_is_append_only(self) -> None:
        drift = self.bundle["result"]["dependency_drift"]
        self.assertEqual(drift["finding_count"], 8)
        self.assertEqual(drift["recovered_count"], 0)
        self.assertEqual(drift["dirty_current_count"], 0)

    def test_architecture_and_navigation_are_complete(self) -> None:
        architecture = self.bundle["result"]["reference_architecture"]
        navigation = self.bundle["result"]["navigation"]
        self.assertEqual(len(architecture["component_ids"]), 8)
        self.assertEqual(architecture["source_outcome_lock_count"], 18)
        self.assertTrue(architecture["integration_passed"])
        self.assertEqual(navigation["roadmap_count"], 28)
        self.assertEqual(navigation["historical_roadmap_count"], 27)
        self.assertEqual(navigation["canonical_roadmap"], "docs/research-roadmap-after-v224.md")

    def test_no_forbidden_access_and_deterministic_reconstruction(self) -> None:
        self.assertTrue(all(value == 0 for value in self.bundle["access"].values()))
        second = build_consolidation()
        self.assertEqual(payload_hash(self.bundle), payload_hash(second))


if __name__ == "__main__":
    unittest.main()
