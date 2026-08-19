from __future__ import annotations

import unittest

from v200_transformed_char_last_controls import _jaccard


class V200MetricTest(unittest.TestCase):
    def test_jaccard_is_contract_set_based(self) -> None:
        self.assertEqual(_jaccard({"A", "B", "C"}, {"C", "B", "A"}), 1.0)
        self.assertEqual(_jaccard({"A", "B", "C"}, {"A", "D", "E"}), 0.2)


if __name__ == "__main__":
    unittest.main()
