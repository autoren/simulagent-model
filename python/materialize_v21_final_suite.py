"""Draw the one delayed seed and atomically materialize the sealed V21 suite."""

from __future__ import annotations

import argparse
import json
import secrets
import tempfile
from pathlib import Path

from audit_v18_benchmark import read_records
from v10_protocol import file_sha256
from v21_final_suite import canonical_json, generate_suite, sha256_text, structural_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v21-multimechanic-execution-lock.json")
    parser.add_argument("--output-dir", default="data/v21-final")
    parser.add_argument("--seed-ledger", default="outputs/v21-final/seed-draw.json")
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    output_dir = PROJECT_ROOT / args.output_dir
    seed_ledger = PROJECT_ROOT / args.seed_ledger
    if output_dir.exists() or seed_ledger.exists():
        raise RuntimeError("V21 seed or final suite already exists; retry forbidden")
    if lock["limits"]["finalSuiteConstructionsPermitted"] != 1:
        raise RuntimeError("Execution lock does not authorize exactly one construction")
    if lock["limits"]["seedDrawsPermitted"] != 1 or lock["limits"]["retriesPermitted"] != 0:
        raise RuntimeError("Execution lock seed/retry policy is invalid")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V21 locked implementation changed: {path}")
    seed = secrets.token_hex(32)
    seed_ledger.parent.mkdir(parents=True, exist_ok=True)
    seed_ledger.write_text(json.dumps({
        "schema_version": 21,
        "draw_number": 1,
        "entropy_bits": 256,
        "source": "python_secrets_system_csprng",
        "seed": seed,
        "seed_sha256": sha256_text(seed),
        "execution_lock": args.lock,
        "execution_lock_sha256": file_sha256(lock_path),
        "retry_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    v18_records = read_records(PROJECT_ROOT / lock["source"]["v18_dataset"])
    excluded = {2: {tuple(value["target"]["behavioral_signature"]) for value in v18_records}}
    episodes, scenes = generate_suite(lock["config"], seed, excluded)
    summary = structural_summary(episodes, scenes)
    if summary["episodes"] != 40:
        raise RuntimeError("V21 delayed seed did not produce exactly 40 mechanics")
    with tempfile.TemporaryDirectory(prefix="v21-final-", dir=output_dir.parent) as temporary:
        root = Path(temporary)
        artifacts = {}
        payloads = {
            "episodes.jsonl": episodes,
            "supported.jsonl": [value for value in scenes if value["view"] == "supported"],
            "novel_ontology.jsonl": [value for value in scenes if value["view"] == "novel_ontology"],
        }
        for name, values in payloads.items():
            content = "".join(canonical_json(value) + "\n" for value in values)
            (root / name).write_text(content)
            artifacts[name] = sha256_text(content)
        dataset_sha256 = sha256_text("".join(
            f"{name}\n{(root / name).read_text()}" for name in sorted(payloads)
        ))
        manifest = {
            "schema_version": 21,
            "experiment": lock["config"]["experiment"],
            "execution_lock": args.lock,
            "execution_lock_sha256": file_sha256(lock_path),
            "design_lock": lock["design_lock"],
            "design_lock_sha256": lock["design_lock_sha256"],
            "seed_ledger": args.seed_ledger,
            "seed_ledger_sha256": file_sha256(seed_ledger),
            "seed": seed,
            "seed_sha256": sha256_text(seed),
            "summary": summary,
            "artifact_sha256": artifacts,
            "dataset_sha256": dataset_sha256,
            "source": {
                "v18_dataset_sha256": lock["source"]["v18_dataset_sha256"],
                "v18_records_read_for_exclusion_only": len(v18_records),
                "v17_records_read": 0,
                "v17_model_results_read": 0,
            },
            "construction_number": 1,
            "retry_authorized": False,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        Path(temporary).replace(output_dir)
    print(json.dumps({
        "output_dir": args.output_dir,
        "seed_ledger": args.seed_ledger,
        "seed_sha256": sha256_text(seed),
        "dataset_sha256": dataset_sha256,
        "summary": summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
