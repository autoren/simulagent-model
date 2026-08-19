import copy
import unittest

from v92_architecture_audit import (
    build_audit,
    load_json,
    payload_sha256,
    summarize_model_access,
    verify_internal_lock,
)


class V92ArchitectureAuditTests(unittest.TestCase):
    def test_payload_hash_is_order_independent(self) -> None:
        self.assertEqual(payload_sha256({"a": 1, "b": 2}), payload_sha256({"b": 2, "a": 1}))

    def test_every_evidence_lock_is_internally_valid(self) -> None:
        design = load_json("configs/v92-structured-llm-architecture-design.json")
        for path in design["evidenceLocks"]:
            self.assertIsInstance(verify_internal_lock(path), dict)

    def test_cumulative_model_access_is_exact(self) -> None:
        design = load_json("configs/v92-structured-llm-architecture-design.json")
        locks = {}
        for path in design["evidenceLocks"]:
            name = path.split("/")[-1].split("-")[0]
            if name == "v88r1":
                locks["v88r1"] = verify_internal_lock(path)
            elif name in {"v80", "v81", "v82", "v85", "v90", "v91"}:
                locks[name] = verify_internal_lock(path)
        self.assertEqual(
            summarize_model_access(locks),
            {
                "model_load_count": 11,
                "model_generation_count": 401,
                "LLM_API_call_count": 0,
                "adapter_training_run_count": 0,
                "external_side_effect_count": 0,
                "real_tool_or_service_call_count": 0,
            },
        )

    def test_full_audit_passes(self) -> None:
        audit = build_audit(load_json("configs/v92-structured-llm-architecture-design.json"))
        self.assertTrue(audit["passed"])
        self.assertFalse(any(audit["learned_role_qualification"].values()))
        self.assertTrue(all(audit["control_coverage"].values()))

    def test_tampered_lock_is_detected_by_payload_hash(self) -> None:
        payload = verify_internal_lock("configs/v91-rank-only-outcome-lock.json")
        tampered = copy.deepcopy(payload)
        tampered["authorization"]["use_local_model_as_candidate_generator_or_search_scheduler"] = True
        expected = tampered.pop("lock_payload_sha256")
        self.assertNotEqual(payload_sha256(tampered), expected)


if __name__ == "__main__":
    unittest.main()
