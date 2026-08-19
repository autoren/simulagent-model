from __future__ import annotations

import json
from typing import Any


def graphql_request_payload(query: str, variables: dict[str, Any]) -> bytes:
    return json.dumps(
        {"query": query, "variables": variables}, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def transport_preserves_document_and_variables(
    raw: bytes, query: str, variables: dict[str, Any]
) -> bool:
    parsed = json.loads(raw)
    return parsed == {"query": query, "variables": variables}

