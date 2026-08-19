#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


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


def title_from_html(data: bytes) -> str | None:
    match = re.search(rb"<title[^>]*>(.*?)</title>", data, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1).decode("utf-8", errors="replace")).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists() or args.snapshot_directory.exists():
        raise RuntimeError("V217A metadata capture output already exists")
    config = json.loads(args.config.read_text())
    contract = config["evidenceContract"]
    args.snapshot_directory.mkdir(parents=True)
    attempts = []
    total_bytes = 0
    for unit in config["sourceUnits"]:
        for url in unit["urls"]:
            key = hashlib.sha256(f"{unit['unitId']}|{url}".encode()).hexdigest()[:20]
            row = {
                "unit_id": unit["unitId"], "url": url, "success": False, "status": None,
                "final_url": None, "content_type": None, "title": None, "snapshot_path": None,
                "sha256": None, "byte_count": 0, "error": None,
            }
            try:
                handler = LimitedRedirectHandler(contract["maximumRedirects"])
                opener = urllib.request.build_opener(handler)
                request = urllib.request.Request(url, headers={"User-Agent": contract["userAgent"]})
                with opener.open(request, timeout=contract["requestTimeoutSeconds"]) as response:
                    data = response.read(contract["maximumMetadataBytesPerURL"] + 1)
                    if len(data) > contract["maximumMetadataBytesPerURL"]:
                        raise RuntimeError("per-URL metadata byte ceiling exceeded")
                    total_bytes += len(data)
                    if total_bytes > contract["maximumTotalMetadataBytes"]:
                        raise RuntimeError("total metadata byte ceiling exceeded")
                    content_type = response.headers.get_content_type()
                    suffix = ".json" if content_type in {"application/json", "application/vnd.github+json"} else ".html"
                    snapshot = args.snapshot_directory / f"{key}{suffix}"
                    snapshot.write_bytes(data)
                    row.update({
                        "success": 200 <= response.status < 300,
                        "status": response.status,
                        "final_url": response.geturl(),
                        "content_type": content_type,
                        "title": title_from_html(data),
                        "snapshot_path": str(snapshot.resolve()),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "byte_count": len(data),
                    })
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            attempts.append(row)
    manifest = {
        "schema_version": "217a-official-source-event-metadata-retrieval-manifest",
        "experiment": config["experiment"],
        "retrieval_date": contract["retrievalDate"],
        "total_metadata_bytes": sum(row["byte_count"] for row in attempts if row["success"]),
        "attempts": attempts,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

