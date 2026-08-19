from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v168_fixed_ontology_reversible_sandbox import initial_state
from v171_stateful_sandbox_sequence_confirmation import compose_config, run_sequence


class V171StatefulSandboxSequenceConfirmationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        design = json.loads(
            (PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation.json").read_text()
        )
        source_lock = json.loads(
            (PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-lock.json").read_text()
        )
        cls.config = compose_config(design, source_lock["config_payload"])

    def check_unit_sequence(self, scenario: str, variant: int) -> dict:
        result = run_sequence(
            {
                "sequence_id": f"v171-unit-{scenario}-{variant}",
                "split": "implementation_unit_only",
                "scenario": scenario,
                "variant": variant,
                "initial_state": initial_state(variant),
            },
            self.config,
        )
        self.assertTrue(result["exact_oracle_final_state"])
        self.assertTrue(result["invariants_preserved"])
        self.assertTrue(result["zero_unauthorized_retained_mutation"])
        return result

    def test_revision_race_rejects_stale_preview_and_continues(self) -> None:
        result = self.check_unit_sequence("revision_race", 999)
        self.assertTrue(result["revision_race_rejected"])
        self.assertTrue(result["post_recovery_continuation"])

    def test_verified_crash_finalizes_and_continues(self) -> None:
        result = self.check_unit_sequence("crash_after_verify_before_finalize", 998)
        self.assertTrue(result["crash_recovered"])
        self.assertTrue(result["post_recovery_continuation"])

    def test_provenance_tamper_fails_closed(self) -> None:
        result = self.check_unit_sequence("provenance_tamper_detection", 997)
        self.assertTrue(result["provenance_tamper_detected"])


if __name__ == "__main__":
    unittest.main()
