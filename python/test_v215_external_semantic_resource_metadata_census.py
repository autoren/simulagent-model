from __future__ import annotations

import unittest

from v215_external_semantic_resource_metadata_census import _finite, _rate


class V215MetadataCensusTests(unittest.TestCase):
    def test_empty_rate_is_zero(self) -> None:
        self.assertEqual(0.0, _rate([]))

    def test_finite_recurses(self) -> None:
        self.assertTrue(_finite({"a": [0.0, 1.0]}))
        self.assertFalse(_finite({"a": [float("inf")]}))


if __name__ == "__main__":
    unittest.main()
