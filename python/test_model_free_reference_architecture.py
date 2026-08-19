from __future__ import annotations

import unittest

from cross_track_evidence_audit import payload_hash
from model_free_reference_architecture import run_reference_architecture


class ModelFreeReferenceArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = run_reference_architecture()

    def test_all_frozen_integration_gates_pass(self) -> None:
        self.assertTrue(self.bundle["audit"]["passed"])
        self.assertTrue(all(self.bundle["audit"]["gates"].values()))
        self.assertTrue(all(row["payload_lock_valid"] for row in self.bundle["result"]["source_lock_integrity"]))

    def test_one_corruption_is_decoded_without_version_space_divergence(self) -> None:
        state = self.bundle["result"]["typed_version_space"]
        self.assertEqual(state["clean_decoded"], state["one_corruption_decoded"])
        self.assertEqual(state["raw_robust_survivors"], state["clean_decoded_survivors"])
        self.assertEqual(state["routed_decision"], "alias")

    def test_other_defers_without_sandbox_entry_or_hypothesis_loss(self) -> None:
        result = self.bundle["result"]
        other = result["other_defer"]
        self.assertEqual(other["decision"], "defer")
        self.assertTrue(other["version_space_preserved"])
        self.assertEqual(other["candidate_ids_after"], result["typed_version_space"]["candidate_ids"])
        self.assertEqual(other["sandbox_entry_count"], 0)

    def test_sandbox_and_terminal_settlement_invariants(self) -> None:
        sandbox = self.bundle["result"]["reversible_sandbox"]
        semantic = self.bundle["result"]["outside_semantic_terminal_planner"]
        self.assertTrue(sandbox["committed"])
        self.assertTrue(sandbox["exact_final_target_state"])
        self.assertTrue(sandbox["provenance_chain_valid"])
        self.assertTrue(semantic["oracle_audit_passed"])
        self.assertEqual(semantic["root_action"], "calibrate")
        self.assertEqual(semantic["action_after_green"], "defer")
        self.assertEqual(semantic["horizon_escape_path_count"], 0)
        self.assertEqual(semantic["terminal_audit"]["unsettled_paths"], 0)

    def test_access_boundary_and_determinism(self) -> None:
        access = self.bundle["access"]
        exempt = {"simulated_sandbox_transaction_count", "model_free_oracle_evaluation_count"}
        self.assertTrue(all(value == 0 for key, value in access.items() if key not in exempt))
        second = run_reference_architecture()
        self.assertEqual(payload_hash(self.bundle), payload_hash(second))


if __name__ == "__main__":
    unittest.main()
