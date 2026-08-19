import json
import tempfile
import unittest
from pathlib import Path

from prospective_language_phase3 import (
    eligible_clarification_records,
    initialize_or_load_phase3,
    lock_clarification_response,
    next_clarification_record_id,
    validate_phase3_config,
    verify_phase3_bundle,
)
from prospective_language_pilot import PilotProtocolError, load_study_config, sha256_json


ROOT = Path(__file__).resolve().parents[1]
PHASE3_CONFIG = json.loads(
    (ROOT / "configs/prospective-language-pilot-v1-phase3.json").read_text(encoding="utf-8")
)
STUDY_CONFIG = load_study_config(ROOT / "configs/prospective-language-pilot-v1.json")


def controller_rows():
    rows = []
    for position, scenario in enumerate(STUDY_CONFIG["scenarios"], start=1):
        clarify = position <= 11
        rows.append(
            {
                "record_id": scenario["record_id"],
                "display_position": position,
                "route": "CLARIFY" if clarify else "DEFER",
                "clarification_questions": (
                    ["What information can you provide?", "Which outcome matters most?"]
                    if clarify
                    else []
                ),
                "sandbox_plan": None,
                "defer_message": None if clarify else "Insufficient information.",
                "used_safe_fallback": False,
                "controller_payload_sha256": f"hash-{position}",
            }
        )
    return rows


class ProspectiveLanguagePhase3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.participant_dir = Path(self.temp.name) / "P001"
        self.rows = controller_rows()
        lock_payload = {"schema_version": "test-lock", "interaction_authorized": True}
        self.lock = {**lock_payload, "lock_payload_sha256": sha256_json(lock_payload)}
        lock_path = self.participant_dir / "audit" / "phase3_clarification_lock.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(self.lock), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_config_preserves_failed_phase2_and_zero_generation(self):
        validate_phase3_config(PHASE3_CONFIG)
        self.assertFalse(PHASE3_CONFIG["phase2"]["required_qualified_for_clarification_batch"])
        self.assertFalse(PHASE3_CONFIG["collection"]["assistant_generation_enabled"])

    def test_only_valid_nonfallback_clarifications_are_eligible(self):
        eligible = eligible_clarification_records(PHASE3_CONFIG, self.rows)
        self.assertEqual(11, len(eligible))
        self.assertTrue(all(row["route"] == "CLARIFY" for row in eligible))

    def test_complete_collection_is_immutable_and_separated(self):
        state = initialize_or_load_phase3(
            PHASE3_CONFIG, STUDY_CONFIG, self.participant_dir, self.rows, self.lock
        )
        first = next_clarification_record_id(state)
        while (record_id := next_clarification_record_id(state)) is not None:
            lock_clarification_response(
                PHASE3_CONFIG,
                STUDY_CONFIG,
                self.participant_dir,
                self.rows,
                state,
                record_id=record_id,
                answer="This is my natural clarification response.",
                unable_reason=None,
                unable_note="",
                participant_attestation=True,
            )
        self.assertEqual("phase_3_complete_waiting_for_terminal_run", state["phase"])
        manifest = json.loads(
            (self.participant_dir / "audit" / "phase3_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(11, manifest["locked_response_count"])
        self.assertEqual(0, manifest["assistant_generation_count_during_phase3"])
        self.assertFalse(manifest["terminal_run_authorized"])
        report = verify_phase3_bundle(PHASE3_CONFIG, self.participant_dir)
        self.assertEqual("pass", report["verification"])
        self.assertEqual(11, report["locked_response_count"])
        public_text = (
            self.participant_dir / "public" / "phase3_clarification_answers.jsonl"
        ).read_text(encoding="utf-8")
        self.assertNotIn("participant_card", public_text)
        self.assertNotIn("private_goal", public_text)
        with self.assertRaises(PilotProtocolError):
            lock_clarification_response(
                PHASE3_CONFIG,
                STUDY_CONFIG,
                self.participant_dir,
                self.rows,
                state,
                record_id=first,
                answer="Replacement answer.",
                unable_reason=None,
                unable_note="",
                participant_attestation=True,
            )


if __name__ == "__main__":
    unittest.main()
