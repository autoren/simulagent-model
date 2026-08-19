#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import Request, urlopen

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: retrieve_v216_bounded_external_artifacts.py LOCK")
    lock_path = Path(sys.argv[1]).resolve()
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V216 design lock or dependency hash mismatch")
    if not lock["authorization"]["retrieve_exactly_three_frozen_payloads_once"]:
        raise RuntimeError("V216 payload retrieval is not authorized")
    config = lock["config_payload"]
    manifest_path = PROJECT_ROOT / config["artifacts"]["retrievalManifest"]
    if manifest_path.exists() or any((PROJECT_ROOT / payload["rawPath"]).exists() for payload in config["payloads"]):
        raise RuntimeError("V216 retrieval output already exists")
    contract = config["retrievalContract"]
    attempts = []
    total_bytes = 0
    for payload in config["payloads"]:
        target = PROJECT_ROOT / payload["rawPath"]
        temporary = target.with_suffix(target.suffix + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        byte_count = 0
        row = {
            "payload_id": payload["payloadId"],
            "role": payload["role"],
            "url": payload["url"],
            "success": False,
            "status": None,
            "final_url": None,
            "content_type": None,
            "byte_count": 0,
            "sha256": None,
            "raw_path": payload["rawPath"],
            "error": None,
        }
        try:
            request = Request(payload["url"], headers={"User-Agent": contract["userAgent"]})
            with urlopen(request, timeout=contract["requestTimeoutSeconds"]) as response, temporary.open("wb") as output:
                row["status"] = response.status
                row["final_url"] = response.geturl()
                row["content_type"] = response.headers.get_content_type()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    total_bytes += len(chunk)
                    if byte_count > payload["expectedByteCount"] or total_bytes > contract["maximumTotalPayloadBytes"]:
                        raise RuntimeError("frozen byte budget exceeded")
                    digest.update(chunk)
                    output.write(chunk)
            observed_sha = digest.hexdigest()
            if byte_count != payload["expectedByteCount"]:
                raise RuntimeError(f"expected {payload['expectedByteCount']} bytes, observed {byte_count}")
            if payload["declaredSha256"] is not None and observed_sha != payload["declaredSha256"]:
                raise RuntimeError("declared SHA-256 mismatch")
            temporary.replace(target)
            row.update({"success": True, "byte_count": byte_count, "sha256": observed_sha})
        except Exception as error:
            if temporary.exists():
                temporary.unlink()
            row.update({"byte_count": byte_count, "sha256": digest.hexdigest() if byte_count else None, "error": f"{type(error).__name__}: {error}"})
        attempts.append(row)
    manifest = {
        "schema_version": "216-bounded-external-artifact-retrieval-manifest",
        "experiment": config["experiment"],
        "attempts": attempts,
        "total_byte_count": sum(row["byte_count"] for row in attempts if row["success"]),
        "unlisted_network_request_count": 0,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not all(row["success"] for row in attempts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

