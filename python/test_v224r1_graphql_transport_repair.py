from __future__ import annotations

import unittest

from v224r1_graphql_transport_repair import graphql_request_payload, transport_preserves_document_and_variables


class V224R1GraphQLTransportRepairTests(unittest.TestCase):
    def test_document_and_same_named_variable_are_separate(self) -> None:
        query = "query($query: String!) { search(query: $query, type: ISSUE, first: 1) { issueCount } }"
        variables = {"query": "repo:owner/repo is:issue", "after": None}
        raw = graphql_request_payload(query, variables)
        self.assertTrue(transport_preserves_document_and_variables(raw, query, variables))
        self.assertEqual(1, raw.count(b'"variables"'))
        self.assertEqual(2, raw.count(b'"query":'))


if __name__ == "__main__":
    unittest.main()
