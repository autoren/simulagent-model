from __future__ import annotations

import json
import unittest
from pathlib import Path

from v211_deterministic_residual_baselines import select_v210_residual, split_residual, tokenize


CONFIG = json.loads(Path("configs/v211-deterministic-residual-baselines.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class V211ProtocolTest(unittest.TestCase):
    def test_identifier_only_split_reconstructs_locked_hashes(self) -> None:
        surfaces = read_jsonl(Path(CONFIG["population"]["inputDevelopmentSurface"]))
        projections = read_jsonl(Path(CONFIG["population"]["inputDevelopmentProjection"]))
        residual = select_v210_residual(surfaces, projections, CONFIG)
        split = split_residual(residual, CONFIG)
        self.assertEqual(len(residual), 180)
        for key in ("calibrationGroupIdHash", "evaluationGroupIdHash", "calibrationRecordIdHash", "evaluationRecordIdHash"):
            snake = {"calibrationGroupIdHash": "calibration_group_id_hash", "evaluationGroupIdHash": "evaluation_group_id_hash", "calibrationRecordIdHash": "calibration_record_id_hash", "evaluationRecordIdHash": "evaluation_record_id_hash"}[key]
            self.assertEqual(split[snake], CONFIG["population"][key])
        self.assertFalse(set(split["calibration_group_ids"]) & set(split["evaluation_group_ids"]))

    def test_tokenizer_is_casefolded_and_underscore_free(self) -> None:
        self.assertEqual(tokenize("Signal DAX_applies."), ("signal", "dax", "applies"))


if __name__ == "__main__":
    unittest.main()
