# V31 API-binding repair amendment

## Trigger

The first call of the first frozen-readout seed failed before the loss was evaluated and before any
optimizer update or parameter artifact was produced. `mlx.nn.value_and_grad(model, fn)` captures
the module and calls `fn` with only the user-supplied tensor arguments. The locked V31 closures
instead declared that captured module as an explicit first argument. The resulting exception was:

```text
TypeError: make_loss.<locals>.loss_fn() missing 1 required positional argument: 'entity_mask'
```

The attempt ledger and empty output directory are preserved under
`outputs/v31-signed-fact-adaptation/failed-attempts/frozen-api-binding/`. At failure time there were
zero completed seeds, zero optimizer updates, zero saved parameters, zero evaluation records read,
and zero evaluation representations or predictions.

## Authorized repair

The amendment changes only the call binding. A wrapper replaces `value_and_grad(model, fn)` with
an equivalent adapter that calls the locked closure as `fn(model, *tensor_arguments)`. The locked
corpus, fit and calibration populations, feature artifact, model, trainable-parameter boundary,
head, losses, class weights, order, seeds, optimizer, learning rate, accumulation, clipping,
epochs, gates, evaluation implementation, and integration implementation remain byte-identical.

The failed call is not counted as a training run because no loss, gradient, or optimizer step was
completed. The amendment authorizes the originally registered three frozen and three LoRA runs,
not an additional seed or repeat. A repaired trained-system lock must hash this amendment before
sealed evaluation can begin.
