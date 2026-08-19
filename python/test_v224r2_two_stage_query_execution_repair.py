from __future__ import annotations

import unittest

from v224_graphql_queries import NODE_QUERY, RECORD_QUERY, forbidden_selected_fields, selected_field_tokens
from v224r2_graphql_queries import DEEP_NODE_QUERY, RELEASE_QUERY, THIN_RECORD_QUERY


class V224R2TwoStageRepairTests(unittest.TestCase):
    def test_thin_query_is_safe_subset_and_deep_query_is_exact(self) -> None:
        self.assertEqual([], forbidden_selected_fields(THIN_RECORD_QUERY))
        self.assertEqual([], forbidden_selected_fields(DEEP_NODE_QUERY))
        self.assertEqual([], forbidden_selected_fields(RELEASE_QUERY))
        self.assertEqual(NODE_QUERY, DEEP_NODE_QUERY)
        allowed_new_shape_tokens = {"query", "string"}
        self.assertTrue(
            selected_field_tokens(THIN_RECORD_QUERY)
            <= selected_field_tokens(RECORD_QUERY) | allowed_new_shape_tokens
        )

    def test_thin_query_omits_deep_pull_metadata(self) -> None:
        self.assertNotIn("mergedBy", THIN_RECORD_QUERY)
        self.assertNotIn("reviews(first", THIN_RECORD_QUERY)
        self.assertNotIn("files(first", THIN_RECORD_QUERY)


if __name__ == "__main__":
    unittest.main()

