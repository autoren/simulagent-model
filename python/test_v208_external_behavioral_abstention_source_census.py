from __future__ import annotations

import unittest

from v208_external_behavioral_abstention_source_census import marker_counts


class V208TreeMarkerTest(unittest.TestCase):
    def test_marker_counts_only_paths(self) -> None:
        groups = {"pair": ["pair_id"], "gold": ["should_abstain"], "phase": ["pre_execution"]}
        counts = marker_counts(
            ["schemas/pair_id.py", "labels/should_abstain.json", "tasks/pre_execution/index.json", "README.md"],
            groups,
        )
        self.assertEqual(counts, {"pair": 1, "gold": 1, "phase": 1})


if __name__ == "__main__":
    unittest.main()
