# V209r1 dynamic regime-shape validation repair

V209 passed its design audit and all six unit tests, then stopped before a scientific result while constructing the closed-world comparator. The shared kernel validator required the full `(3, 2, 3)` regime/state/observation shape. That is correct for the full environment but incompatible with the deliberately preregistered two-regime closed-world and one-regime point-model comparators.

V209r1 changes only that implementation invariant. It infers a positive regime count from the reference array, requires the target array to have the same shape, and continues to require exactly two task states and three semantic observations. Normalization, finite values, common positive support, cost shapes, history anchors, and every other validation remain unchanged.

The original V209 config payload is used directly from its immutable design lock. No scientific parameter or gate is copied or editable in V209r1. After a separate repair lock, unit tests may run and exactly one repaired oracle evaluation may occur. A miss is frozen without further repair unless another independently auditable software failure prevents a scientific result.
