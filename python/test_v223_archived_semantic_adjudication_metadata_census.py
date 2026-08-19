from __future__ import annotations

import unittest

from v223_archived_semantic_adjudication_metadata_census import finite, metadata_only_url, rate


class V223MetadataCensusTests(unittest.TestCase):
    def test_rate_and_finite_helpers(self) -> None:
        self.assertEqual(0.0, rate([]))
        self.assertEqual(0.5, rate([True, False]))
        self.assertTrue(finite({"a": [0.0, 1.0]}))
        self.assertFalse(finite({"a": [float("inf")]}))

    def test_metadata_firewall_accepts_frozen_workflow_sources(self) -> None:
        self.assertTrue(
            metadata_only_url(
                "https://raw.githubusercontent.com/example/repo/0123456789abcdef0123456789abcdef01234567/README.md"
            )
        )
        self.assertTrue(
            metadata_only_url(
                "https://www.wikidata.org/w/index.php?title=Template:Property_proposal&oldid=123"
            )
        )

    def test_metadata_firewall_rejects_record_endpoints(self) -> None:
        self.assertFalse(metadata_only_url("https://api.github.com/repos/example/repo/issues/42"))
        self.assertFalse(
            metadata_only_url(
                "https://www.wikidata.org/wiki/Wikidata:Property_proposal/Archive/1"
            )
        )


if __name__ == "__main__":
    unittest.main()

