from __future__ import annotations

from v224_graphql_queries import NODE_QUERY as DEEP_NODE_QUERY, RELEASE_QUERY


THIN_RECORD_QUERY = r"""
query($query: String!, $after: String) {
  search(query: $query, type: ISSUE, first: 100, after: $after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Issue {
        id number createdAt updatedAt closedAt lastEditedAt state stateReason
        author { __typename login }
        labels(first: 100) {
          totalCount pageInfo { hasNextPage }
          nodes { name }
        }
        closedByPullRequestsReferences(first: 1) {
          totalCount pageInfo { hasNextPage }
          nodes { id number }
        }
        timelineItems(
          first: 100,
          itemTypes: [LABELED_EVENT, UNLABELED_EVENT, CLOSED_EVENT, REOPENED_EVENT,
                      MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT]
        ) {
          totalCount pageInfo { hasNextPage endCursor }
          nodes {
            __typename
            ... on LabeledEvent { createdAt actor { __typename login } label { name } }
            ... on UnlabeledEvent { createdAt actor { __typename login } label { name } }
            ... on ClosedEvent { createdAt stateReason actor { __typename login } }
            ... on ReopenedEvent { createdAt stateReason actor { __typename login } }
            ... on MarkedAsDuplicateEvent {
              createdAt actor { __typename login } isCrossRepository
              canonical {
                __typename
                ... on Issue { id number createdAt repository { nameWithOwner } }
              }
              duplicate {
                __typename
                ... on Issue { id number createdAt repository { nameWithOwner } }
              }
            }
            ... on UnmarkedAsDuplicateEvent {
              createdAt actor { __typename login } isCrossRepository
              canonical {
                __typename
                ... on Issue { id number repository { nameWithOwner } }
              }
              duplicate {
                __typename
                ... on Issue { id number repository { nameWithOwner } }
              }
            }
          }
        }
      }
    }
  }
}
"""

