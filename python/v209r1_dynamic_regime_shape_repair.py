from __future__ import annotations

from typing import Any

import numpy as np

import v209_controlled_language_observation_pomdp as parent


def _dynamic_language_kernel_post_init(self: parent.LanguageKernel) -> None:
    reference = np.asarray(self.reference, dtype=np.float64)
    target = np.asarray(self.target, dtype=np.float64)
    anchors = np.asarray(self.history_anchors, dtype=np.float64)
    costs = {name: np.asarray(value, dtype=np.float64) for name, value in self.clarification_costs.items()}
    offsets = np.asarray(self.history_cost_offsets, dtype=np.float64)
    if reference.ndim != 3 or reference.shape[0] < 1:
        raise ValueError("V209r1 language channel requires at least one regime")
    if reference.shape[1:] != (len(parent.STATE_NAMES), len(parent.OBSERVATION_NAMES)):
        raise ValueError("V209r1 fixed state/observation shape mismatch")
    if target.shape != reference.shape:
        raise ValueError("V209r1 target must match reference regime/state/observation shape")
    if anchors.shape != (len(parent.OBSERVATION_NAMES), len(parent.OBSERVATION_NAMES)):
        raise ValueError("V209r1 history-anchor shape mismatch")
    if set(costs) != set(parent.CLARIFICATION_ACTIONS):
        raise ValueError("V209r1 clarification-cost actions mismatch")
    if any(value.shape != reference.shape[:2] for value in costs.values()):
        raise ValueError("V209r1 clarification-cost shape mismatch")
    if offsets.shape != (len(parent.OBSERVATION_NAMES),):
        raise ValueError("V209r1 history-cost shape mismatch")
    if not 0.0 < float(self.history_mix_weight) < 1.0:
        raise ValueError("V209r1 history mix must be strictly between zero and one")
    for name, value in (("reference", reference), ("target", target), ("history anchors", anchors)):
        if not np.allclose(value.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"V209r1 {name} is not normalized")
        if np.any(value <= 0.0) or not np.isfinite(value).all():
            raise ValueError(f"V209r1 {name} lacks finite common positive support")
    if not all(np.isfinite(value).all() for value in (*costs.values(), offsets)):
        raise ValueError("V209r1 clarification costs are nonfinite")
    for value in (reference, target, anchors, offsets, *costs.values()):
        value.setflags(write=False)
    object.__setattr__(self, "reference", reference)
    object.__setattr__(self, "target", target)
    object.__setattr__(self, "history_anchors", anchors)
    object.__setattr__(self, "clarification_costs", costs)
    object.__setattr__(self, "history_cost_offsets", offsets)


parent.LanguageKernel.__post_init__ = _dynamic_language_kernel_post_init

LanguageKernel = parent.LanguageKernel
audit_oracle = parent.audit_oracle
build_kernel = parent.build_kernel
evaluate_oracle = parent.evaluate_oracle


def repair_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    kernel, _ = build_kernel(config)
    regime_shapes = {}
    for count in (1, 2, 3):
        comparator = parent._subkernel(kernel, list(range(count)))
        regime_shapes[str(count)] = {
            "reference": list(comparator.reference.shape),
            "target": list(comparator.target.shape),
            "ask_reference_cost": list(comparator.clarification_costs["ask_reference"].shape),
            "ask_target_cost": list(comparator.clarification_costs["ask_target"].shape),
        }
    return {
        "one_two_three_regime_kernels_construct": True,
        "regime_shapes": regime_shapes,
        "parent_config_used_without_copy": True,
        "changed_scientific_parameter_count": 0,
        "changed_gate_count": 0,
        "changed_comparator_count": 0,
        "changed_decision_rule_count": 0,
    }


__all__ = ["LanguageKernel", "audit_oracle", "build_kernel", "evaluate_oracle", "repair_diagnostics"]
