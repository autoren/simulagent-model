from __future__ import annotations

import json
import unittest
from pathlib import Path

from cross_track_evidence_audit import (
    CONFIG_PATH,
    ROOT,
    audit_repository,
    classify_status,
    is_repair,
    outcome_signals,
    payload_hash,
    valid_lock,
    version_number,
    version_token,
)


class CrossTrackEvidenceAuditTests(unittest.TestCase):
    def test_lock_hash_contract(self) -> None:
        lock = {"schema_version": "test.v1", "value": 3}
        lock["lock_payload_sha256"] = payload_hash(lock)
        self.assertTrue(valid_lock(lock))
        lock["value"] = 4
        self.assertFalse(valid_lock(lock))

    def test_version_and_repair_parsing(self) -> None:
        self.assertEqual(version_number("configs/v224r2-example-outcome-lock.json"), 224)
        self.assertEqual(version_token("v217a-example-outcome-lock.json"), "v217a")
        self.assertTrue(is_repair("v224r2-example-outcome-lock.json"))
        self.assertFalse(is_repair("v224-example-outcome-lock.json"))

    def test_status_signals_do_not_descend_into_row_lists(self) -> None:
        lock = {
            "outcome": {
                "decision": "freeze_negative_branch",
                "scientific_passed": False,
                "rows": [{"exact_decision": True}] * 5,
            }
        }
        signals = outcome_signals(lock)
        self.assertEqual(len(signals["decisions"]), 1)
        self.assertEqual(classify_status(signals, False), "negative_or_boundary")

    def test_repository_coverage_and_safety(self) -> None:
        audit = audit_repository(ROOT, CONFIG_PATH)
        reproducibility = audit["reproducibility_audit"]
        self.assertEqual(reproducibility["outcome_lock_count"], 198)
        self.assertEqual(reproducibility["payload_valid_count"], 198)
        self.assertEqual(reproducibility["payload_invalid"], [])
        self.assertEqual(reproducibility["versions_without_frozen_outcome"], [58, 76, 77, 93, 99, 222])
        self.assertEqual(len(audit["family_ledger"]), 17)
        self.assertEqual(len(audit["critical_chain"]), 18)
        self.assertEqual(audit["stopping_decision"]["authorized_next_experiment_count"], 0)
        config = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
        skipped_suffixes = set(config["dependency_hash_policy"]["skip_extensions"])
        forbidden_components = set(config["dependency_hash_policy"]["skip_path_components"])
        for experiment in audit["experiment_ledger"]:
            for row in experiment["dependency_audit"]["verified"]:
                path = Path(row["path"])
                self.assertNotIn(path.suffix.lower(), skipped_suffixes)
                self.assertFalse({part.lower() for part in path.parts} & forbidden_components)


if __name__ == "__main__":
    unittest.main()
