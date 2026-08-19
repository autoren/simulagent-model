from __future__ import annotations

import re


RECORD_QUERY = r"""
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
        closedByPullRequestsReferences(first: 20) {
          totalCount pageInfo { hasNextPage }
          nodes {
            id number state merged mergedAt
            author { __typename login }
            mergedBy { __typename login }
            mergeCommit { oid committedDate }
            reviews(first: 100) {
              totalCount pageInfo { hasNextPage }
              nodes { state submittedAt author { __typename login } }
            }
            files(first: 100) {
              totalCount pageInfo { hasNextPage }
              nodes { path additions deletions changeType }
            }
          }
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


NODE_QUERY = RECORD_QUERY.replace(
    "query($query: String!, $after: String)", "query($id: ID!)"
).replace(
    "search(query: $query, type: ISSUE, first: 100, after: $after) {\n    issueCount\n    pageInfo { hasNextPage endCursor }\n    nodes {\n      ... on Issue {",
    "node(id: $id) {\n      ... on Issue {",
).replace("\n      }\n    }\n  }\n}\n", "\n      }\n  }\n}\n")


RELEASE_QUERY = r"""
query($after: String) {
  repository(owner: "monarch-initiative", name: "mondo") {
    releases(first: 100, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes {
        tagName publishedAt isDraft isPrerelease
        releaseAssets(first: 100) {
          totalCount pageInfo { hasNextPage }
          nodes { name downloadUrl size updatedAt }
        }
      }
    }
  }
}
"""


FORBIDDEN_SELECTED_FIELDS = {
    "title", "body", "bodytext", "bodyhtml", "comments", "comment", "text",
    "message", "description", "descriptionhtml", "reviewsbody", "reactiongroups",
}


def selected_field_tokens(query: str) -> set[str]:
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', query)
    without_comments = re.sub(r"#[^\n]*", "", without_strings)
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", without_comments))
    return {token.lower() for token in tokens}


def forbidden_selected_fields(query: str) -> list[str]:
    return sorted(selected_field_tokens(query) & FORBIDDEN_SELECTED_FIELDS)

