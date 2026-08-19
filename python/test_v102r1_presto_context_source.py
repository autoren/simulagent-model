from __future__ import annotations

import unittest

from test_v102_presto_context_source import CONFIG, row
from v102r1_presto_context_source import (
    build_repaired_presto_context_inventory,
    evaluate_repaired_presto_source_gates,
    tolerant_context_surfaces,
)


class V102r1RepairTests(unittest.TestCase):
    def test_non_string_optional_leaves_are_ignored_not_coerced(self) -> None:
        metadata = row("d", "dev", "alpha project", seeded=False)["metadata"]
        metadata["seeded_notes"] = [{"name": None, "text": 42}]
        metadata["seeded_contacts"] = [None, 99]
        surfaces, ignored = tolerant_context_surfaces(metadata)
        self.assertGreaterEqual(ignored, 4)
        self.assertFalse(any("42" in value or "99" in value for _, value in surfaces))

    def test_repair_preserves_scientific_eligibility_and_text_free_output(self) -> None:
        development = row("d", "dev", "alpha project", seeded=False)
        development["metadata"]["seeded_notes"] = [{"name": None, "text": 42}]
        protected = row("t", "test", "beta task", seeded=True)
        protected["metadata"]["seeded_contacts"] = [None]
        inventory = build_repaired_presto_context_inventory([
            ("presto_dev.jsonl", development),
            ("presto_test.jsonl", protected),
        ], CONFIG)
        self.assertEqual(inventory["eligible_candidate_count"], 2)
        self.assertGreater(inventory["ignored_non_string_optional_context_leaf_count"], 0)
        self.assertTrue(all(evaluate_repaired_presto_source_gates(inventory, CONFIG).values()))
        self.assertNotIn("alpha project", str(inventory))
        self.assertNotIn("beta task", str(inventory))

    def test_null_containers_are_empty(self) -> None:
        metadata = row("d", "dev", "alpha project", seeded=False)["metadata"]
        metadata["seeded_notes"] = None
        metadata["seeded_contacts"] = None
        _, ignored = tolerant_context_surfaces(metadata)
        self.assertEqual(ignored, 2)


if __name__ == "__main__":
    unittest.main()
