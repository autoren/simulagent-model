import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from train_v8_relational_head import QueryConditionedRelationalHead, make_loss


class V8RelationalHeadTests(unittest.TestCase):
    def test_forward_shapes_and_gradient_step(self):
        mx.random.seed(11)
        records, dims, width = 6, 8, 4
        model = QueryConditionedRelationalHead(dims, width)
        inputs = (
            mx.random.normal((records, dims)),
            mx.random.normal((records, dims)),
            mx.random.normal((records, dims)),
            mx.random.normal((records, 7, dims)),
            mx.random.normal((records, 7, dims)),
        )
        status_targets_np = np.arange(records * 7, dtype=np.int32).reshape(records, 7) % 5
        sensitivity_np = (status_targets_np == 2).astype(np.float32)
        config = {"lossWeights": {
            "determinant": 0.5,
            "rowSensitivity": 1.0,
            "pointwise": 1.0,
            "pairwise": 2.0,
            "surface": 0.1,
        }}
        loss_fn = make_loss(
            config,
            mx.array([1, 0, 1, 0, 1, 0], dtype=mx.float32),
            mx.array(status_targets_np),
            mx.array(sensitivity_np),
            mx.ones((records, 7), dtype=mx.float32),
            mx.ones((records, 7), dtype=mx.float32),
            mx.ones((records,), dtype=mx.float32),
            mx.array([[0, 1], [2, 3], [4, 5]], dtype=mx.int32),
            mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int32),
        )
        optimizer = optim.Adam(learning_rate=0.001)
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        (loss, parts), gradients = loss_and_grad(model, *inputs)
        optimizer.update(model, gradients)
        status, rows, records_out = model(*inputs)
        mx.eval(model.parameters(), optimizer.state, loss, parts, status, rows, records_out)
        self.assertEqual(status.shape, (records, 7, 5))
        self.assertEqual(rows.shape, (records, 7))
        self.assertEqual(records_out.shape, (records,))
        self.assertTrue(np.isfinite(float(loss)))


if __name__ == "__main__":
    unittest.main()
