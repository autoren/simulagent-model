from __future__ import annotations

import unittest

from v224_graphql_queries import NODE_QUERY, RECORD_QUERY, RELEASE_QUERY, forbidden_selected_fields
from v224_mondo_record_disposition_metadata_census import mondo_ids_from_patch, normalize_label, selection_key


class V224MetadataCensusTests(unittest.TestCase):
    def test_graphql_queries_select_no_forbidden_language_fields(self) -> None:
        self.assertEqual([], forbidden_selected_fields(RECORD_QUERY))
        self.assertEqual([], forbidden_selected_fields(NODE_QUERY))
        self.assertEqual([], forbidden_selected_fields(RELEASE_QUERY))

    def test_patch_filter_retains_only_exact_added_ids(self) -> None:
        patch = " label: secret\n+id: MONDO:1234567\n+name: hidden\n-id: MONDO:0000001\n"
        self.assertEqual(["MONDO:1234567"], mondo_ids_from_patch(patch))

    def test_normalization_and_selection_are_stable(self) -> None:
        self.assertEqual("new term request", normalize_label(" New   Term Request "))
        config = {"samplingContract": {"seed": "fixed"}}
        self.assertEqual(selection_key(42, config), selection_key(42, config))
        self.assertNotEqual(selection_key(42, config), selection_key(43, config))


if __name__ == "__main__":
    unittest.main()

