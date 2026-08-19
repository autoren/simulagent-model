import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from prospective_language_pilot import (
    PilotProtocolError,
    completed_count,
    deterministic_scenario_order,
    initialize_or_load_session,
    load_study_config,
    lock_initial_response,
    next_incomplete_record_id,
    phase_1_is_complete,
    study_config_hash,
    verify_phase_1_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "prospective-language-pilot-v1.json"


class ProspectiveLanguagePilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_study_config(CONFIG_PATH)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_frozen_population_is_diverse_and_opaque(self) -> None:
        scenarios = self.config["scenarios"]
        self.assertEqual(16, len(scenarios))
        self.assertEqual(16, len({scenario["record_id"] for scenario in scenarios}))
        families = {
            family
            for scenario in scenarios
            for family in scenario["research_metadata"]["families"]
        }
        self.assertTrue(
            {
                "everyday_life",
                "fantastical",
                "mystery",
                "art_design",
                "faith_christian",
                "emotional",
                "logistics",
            }.issubset(families)
        )
        self.assertFalse(self.config["phase_1"]["assistant_generation_enabled"])
        self.assertFalse(self.config["phase_1"]["assistant_run_authorized"])

    def test_order_is_deterministic_and_participant_specific(self) -> None:
        first = deterministic_scenario_order(self.config, "P001")
        second = deterministic_scenario_order(self.config, "P001")
        other = deterministic_scenario_order(self.config, "P002")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(16, len(first))

    def test_lock_creates_separate_public_private_and_audit_files(self) -> None:
        participant_dir, state = initialize_or_load_session(
            self.config, "P001", self.storage_root
        )
        record_id = next_incomplete_record_id(state)
        self.assertIsNotNone(record_id)
        lock_initial_response(
            self.config,
            participant_dir,
            state,
            record_id=record_id,
            initial_request="Please help me work through this situation naturally.",
            unable_reason=None,
            unable_note="",
            participant_attestation=True,
        )

        public_path = participant_dir / "public" / "phase1_initial_requests.jsonl"
        private_path = participant_dir / "private" / "phase1_private_records.jsonl"
        audit_path = participant_dir / "audit" / "phase1_manifest.json"
        public_text = public_path.read_text(encoding="utf-8")
        private_text = private_path.read_text(encoding="utf-8")
        manifest = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertIn("initial_request", public_text)
        self.assertNotIn("participant_card", public_text)
        self.assertNotIn("private_goal", public_text)
        self.assertNotIn("research_metadata", public_text)
        self.assertIn("participant_card", private_text)
        self.assertEqual(1, manifest["locked_record_count"])
        self.assertEqual(0, manifest["assistant_generation_count"])
        self.assertFalse(manifest["assistant_run_authorized"])

    def test_locked_record_cannot_be_replaced(self) -> None:
        participant_dir, state = initialize_or_load_session(
            self.config, "P001", self.storage_root
        )
        record_id = next_incomplete_record_id(state)
        lock_initial_response(
            self.config,
            participant_dir,
            state,
            record_id=record_id,
            initial_request="This is my first natural request for the scenario.",
            unable_reason=None,
            unable_note="",
            participant_attestation=True,
        )
        with self.assertRaises(PilotProtocolError):
            lock_initial_response(
                self.config,
                participant_dir,
                state,
                record_id=record_id,
                initial_request="I am trying to replace my locked response.",
                unable_reason=None,
                unable_note="",
                participant_attestation=True,
            )

    def test_explicit_unable_response_is_valid_but_blank_is_not(self) -> None:
        participant_dir, state = initialize_or_load_session(
            self.config, "P001", self.storage_root
        )
        record_id = next_incomplete_record_id(state)
        with self.assertRaises(PilotProtocolError):
            lock_initial_response(
                self.config,
                participant_dir,
                state,
                record_id=record_id,
                initial_request="",
                unable_reason=None,
                unable_note="",
                participant_attestation=True,
            )
        response = lock_initial_response(
            self.config,
            participant_dir,
            state,
            record_id=record_id,
            initial_request="",
            unable_reason="scenario_unclear",
            unable_note="I cannot tell what I would naturally ask.",
            participant_attestation=True,
        )
        self.assertEqual("unable_to_respond", response["response_status"])

    def test_all_records_complete_phase_without_assistant_generation(self) -> None:
        participant_dir, state = initialize_or_load_session(
            self.config, "P001", self.storage_root
        )
        while (record_id := next_incomplete_record_id(state)) is not None:
            lock_initial_response(
                self.config,
                participant_dir,
                state,
                record_id=record_id,
                initial_request=f"My natural first request for record {record_id}.",
                unable_reason=None,
                unable_note="",
                participant_attestation=True,
            )

        self.assertTrue(phase_1_is_complete(self.config, state))
        self.assertEqual(16, completed_count(state))
        self.assertEqual("phase_1_complete_waiting_for_assistant_run", state["phase"])
        manifest = json.loads(
            (participant_dir / "audit" / "phase1_manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(manifest["phase_1_complete"])
        self.assertEqual(16, manifest["locked_record_count"])
        self.assertEqual(0, manifest["assistant_generation_count"])
        report = verify_phase_1_bundle(self.config, participant_dir)
        self.assertEqual("pass", report["verification"])
        self.assertEqual(16, report["locked_record_count"])

    def test_verifier_rejects_tampered_public_projection(self) -> None:
        participant_dir, state = initialize_or_load_session(
            self.config, "P001", self.storage_root
        )
        while (record_id := next_incomplete_record_id(state)) is not None:
            lock_initial_response(
                self.config,
                participant_dir,
                state,
                record_id=record_id,
                initial_request=f"My natural first request for record {record_id}.",
                unable_reason=None,
                unable_note="",
                participant_attestation=True,
            )
        public_path = participant_dir / "public" / "phase1_initial_requests.jsonl"
        public_path.write_text(
            public_path.read_text(encoding="utf-8").replace("natural", "altered", 1),
            encoding="utf-8",
        )
        with self.assertRaises(PilotProtocolError):
            verify_phase_1_bundle(self.config, participant_dir)

    def test_frozen_config_change_prevents_resume(self) -> None:
        initialize_or_load_session(self.config, "P001", self.storage_root)
        changed = deepcopy(self.config)
        changed["scenarios"][0]["title"] = "Changed after collection began"
        self.assertNotEqual(study_config_hash(self.config), study_config_hash(changed))
        with self.assertRaises(PilotProtocolError):
            initialize_or_load_session(changed, "P001", self.storage_root)


if __name__ == "__main__":
    unittest.main()
