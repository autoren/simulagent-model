from __future__ import annotations

import unittest

from v203_independent_confirmation_feasibility import (
    _partition_dialogue,
    _support_census,
)


class V203FeasibilityTest(unittest.TestCase):
    def test_partition_dialogue_key_preserves_split(self) -> None:
        candidate = "sgd::test::99_00001::004::Alarm_1::AddAlarm"
        self.assertEqual(_partition_dialogue(candidate), "test::99_00001")

    def test_support_census_requires_exact_contract_and_excludes_consumed_dialogue(self) -> None:
        rows = [
            {"candidate_id": "sgd::test::d1::000::S1::I1", "partition": "test", "service": "S1", "intent": "I1"},
            {"candidate_id": "sgd::test::d1::002::S1::I1", "partition": "test", "service": "S1", "intent": "I1"},
            {"candidate_id": "sgd::test::d2::000::S1::I1", "partition": "test", "service": "S1", "intent": "I1"},
            {"candidate_id": "sgd::dev::d3::000::S1::I1", "partition": "dev", "service": "S1", "intent": "I1"},
            {"candidate_id": "sgd::test::d4::000::S2::I2", "partition": "test", "service": "S2", "intent": "I2"},
        ]
        census = _support_census(
            rows,
            {"test": {"S1::I1": "C1", "S2::I2": "OTHER"}, "dev": {"S1::I1": "C1"}},
            {"C1", "C2"},
            set(),
            {"test::d1"},
            {"test"},
        )
        by_contract = {row["capability_contract_id"]: row for row in census["records"]}
        self.assertEqual(by_contract["C1"]["eligible_source_record_count"], 1)
        self.assertEqual(by_contract["C1"]["eligible_partition_dialogue_count"], 1)
        self.assertEqual(by_contract["C2"]["eligible_partition_dialogue_count"], 0)
        self.assertEqual(census["exact_contract_coverage"], 1)


if __name__ == "__main__":
    unittest.main()
