import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from train_v8_action_conditioned_head import (
    ActionConditionedHead,
    gate_report,
    make_loss,
)


class V8StructuredHeadGateTests(unittest.TestCase):
    def test_structured_head_loss_and_gradient_step_are_finite(self):
        mx.random.seed(7)
        records = 6
        input_dims = 8
        model = ActionConditionedHead(input_dims, width=4)
        optimizer = optim.Adam(learning_rate=0.001)
        inputs = (
            mx.random.normal((records, input_dims)),
            mx.random.normal((records, input_dims)),
            mx.random.normal((records, input_dims)),
            mx.random.normal((records, 7, input_dims)),
            mx.random.normal((records, 7, input_dims)),
        )
        record_targets = mx.array([1, 0, 1, 0, 1, 0], dtype=mx.float32)
        status_targets = mx.array(
            np.arange(records * 7, dtype=np.int32).reshape(records, 7) % 5
        )
        loss_fn = make_loss(
            {"lossWeights": {
                "determinant": 1.0,
                "pointwise": 0.5,
                "pairwise": 2.0,
                "surface": 0.1,
            }},
            record_targets,
            status_targets,
            mx.ones((records, 7), dtype=mx.float32),
            mx.ones((records,), dtype=mx.float32),
            mx.array([[0, 1], [2, 3], [4, 5]], dtype=mx.int32),
            mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int32),
        )
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        (loss, parts), gradients = loss_and_grad(model, *inputs)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss, parts)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(np.isfinite(float(value)) for value in parts.values()))

    def test_absolute_structured_and_direction_gates_are_all_hard(self):
        surface = {
            "pointwise": {"balanced_accuracy": 0.8},
            "pair_direction": {"accuracy": 0.9},
            "structured": {
                "status_macro_f1": 0.64,
                "decisive_determinant_accuracy": 0.9,
            },
        }
        folds = {"a": {"by_surface": {"canonical": surface}}, "b": {"by_surface": {"canonical": surface}}}
        report = gate_report(folds, {
            "minimumEveryFoldSurfaceBalancedAccuracy": 0.65,
            "minimumMeanFoldSurfaceBalancedAccuracy": 0.75,
            "minimumEveryFoldSurfacePairDirection": 0.85,
            "minimumEveryFoldSurfaceStatusMacroF1": 0.65,
            "minimumEveryFoldSurfaceDecisiveDeterminantAccuracy": 0.75,
        })
        self.assertFalse(report["passed"])
        self.assertEqual(
            [check["name"] for check in report["checks"] if not check["passed"]],
            ["minimum_fold_surface_status_macro_f1"],
        )


if __name__ == "__main__":
    unittest.main()
