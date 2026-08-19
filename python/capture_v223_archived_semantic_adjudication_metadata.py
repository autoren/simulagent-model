#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

from v223_archived_semantic_adjudication_metadata_census import metadata_only_url


class LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, maximum: int) -> None:
        super().__init__()
        self.maximum = maximum
        self.count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.count += 1
        if self.count > self.maximum:
            raise urllib.error.HTTPError(newurl, code, "redirect limit exceeded", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists() or args.snapshot_directory.exists():
        raise RuntimeError("V223 metadata capture output already exists")
    config = json.loads(args.config.read_text())
    contract = config["evidenceContract"]
    frozen_urls = [url for unit in config["sourceUnits"] for url in unit["urls"]]
    if not all(metadata_only_url(url) for url in frozen_urls):
        raise RuntimeError("V223 frozen URL violates the task-language firewall")
    args.snapshot_directory.mkdir(parents=True)
    attempts = []
    total_successful_bytes = 0
    for unit in config["sourceUnits"]:
        for url in unit["urls"]:
            key = hashlib.sha256(f"{unit['unitId']}|{url}".encode()).hexdigest()[:20]
            snapshot = args.snapshot_directory / f"{key}.snapshot"
            row = {
                "unit_id": unit["unitId"],
                "url": url,
                "success": False,
                "status": None,
                "final_url": None,
                "content_type": None,
                "snapshot_path": None,
                "sha256": None,
                "byte_count": 0,
                "error": None,
            }
            try:
                handler = LimitedRedirectHandler(contract["maximumRedirects"])
                opener = urllib.request.build_opener(handler)
                request = urllib.request.Request(url, headers={"User-Agent": contract["userAgent"]})
                with opener.open(request, timeout=contract["requestTimeoutSeconds"]) as response:
                    remaining = contract["maximumTotalSuccessfulSnapshotBytes"] - total_successful_bytes
                    data = response.read(remaining + 1)
                    if len(data) > remaining:
                        raise RuntimeError("V223 total successful snapshot byte limit exceeded")
                    row.update(
                        {
                            "success": 200 <= response.status < 300,
                            "status": response.status,
                            "final_url": response.geturl(),
                            "content_type": response.headers.get_content_type(),
                        }
                    )
                    if row["success"]:
                        snapshot.write_bytes(data)
                        row["snapshot_path"] = str(snapshot)
                        row["sha256"] = hashlib.sha256(data).hexdigest()
                        row["byte_count"] = len(data)
                        total_successful_bytes += len(data)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            attempts.append(row)
    manifest = {
        "schema_version": "223-archived-semantic-adjudication-metadata-retrieval-manifest",
        "experiment": config["experiment"],
        "retrieval_date": contract["retrievalDate"],
        "formal_task_record_body_read_count": 0,
        "issue_proposal_comment_pull_or_archive_record_request_count": 0,
        "attempts": attempts,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
