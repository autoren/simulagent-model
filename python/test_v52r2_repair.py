from __future__ import annotations

import json
import unittest
from decimal import Decimal, localcontext

from v22r2_grounding import PROJECT_ROOT
from v51_sbc import distribution_tv, sequence_tv
from v52_particle import mechanic_registry, particle_inference as base_inference
from v52r2_particle import particle_inference as repaired_inference


class V52R2JointNormalizationRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (PROJECT_ROOT / "configs/v52-implementation-lock.json").read_text()
        )["config_payload"]
        cls.registry = mechanic_registry()
        cls.record = next(
            json.loads(line)
            for line in (
                PROJECT_ROOT
                / "data/v52-rao-blackwellized-particle-filtering/sbc.jsonl"
            ).read_text().splitlines()
            if json.loads(line)["id"] == "sbc_00006"
        )
        args = (
            cls.registry, cls.record["supports"], cls.record["query"],
            cls.config["particleBudgets"]["primaryBudget"],
            cls.config["population"]["particleSeed"], "sbc", cls.record["id"],
            cls.config["particleBudgets"]["primaryRepeatOnSbc"],
            cls.config["algorithm"]["resamplingEssThresholdFraction"],
        )
        cls.base = base_inference(*args)
        cls.repaired = repaired_inference(*args)

    def test_filtering_paths_and_likelihoods_are_unchanged(self):
        self.assertEqual(
            self.base["support_log_evidence_by_program"],
            self.repaired["support_log_evidence_by_program"],
        )
        self.assertEqual(
            self.base["query_log_weight_by_program"],
            self.repaired["query_log_weight_by_program"],
        )
        self.assertEqual(
            self.base["record_log_evidence"], self.repaired["record_log_evidence"]
        )
        self.assertEqual(
            self.base["support_diagnostics"], self.repaired["support_diagnostics"]
        )
        self.assertEqual(
            self.base["query_diagnostics"], self.repaired["query_diagnostics"]
        )

    def test_repair_is_below_preregistered_substantive_tv_limit(self):
        values = [
            sequence_tv(self.base["support_program"], self.repaired["support_program"]),
            sequence_tv(self.base["query_program"], self.repaired["query_program"]),
            distribution_tv(self.base["probability"], self.repaired["probability"]),
            distribution_tv(self.base["joint"], self.repaired["joint"]),
            distribution_tv(self.base["configuration"], self.repaired["configuration"]),
            distribution_tv(self.base["suffix"], self.repaired["suffix"]),
        ]
        self.assertLessEqual(max(values), 1e-25)

    def test_every_repaired_marginal_meets_frozen_tolerance(self):
        with localcontext() as context:
            context.prec = 100
            for values in (
                self.repaired["support_program"], self.repaired["query_program"],
                self.repaired["probability"].values(),
                self.repaired["joint"].values(),
                self.repaired["configuration"].values(),
                self.repaired["suffix"].values(),
            ):
                self.assertLess(
                    abs(sum(values, Decimal(0)) - 1), Decimal("1e-80")
                )


if __name__ == "__main__":
    unittest.main()
