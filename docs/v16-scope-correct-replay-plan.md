# V16 preregistration: scope-correct V15 gate replay

V16 corrects one evaluation-topology defect identified after the locked V15 result. It performs no feature extraction, model fit, prediction, threshold change, or data access.

The first thirteen V15 gate checks are copied byte-for-value from the locked result. The complete-intervention-group check keeps its original 0.50 threshold but is computed only over non-context transfer folds whose evaluation mask itself contains at least one complete six-record intervention group. Folds with zero complete in-mask groups report this metric as not applicable. No evaluation mask may be expanded to manufacture complete groups.

The replay must find exactly 15 applicable mechanic/surface folds and 11 non-applicable lexicon/operator/combined folds, matching the post-result topology audit. It verifies the V15 protocol, feature, result, head-artifact, and scope-audit hashes before producing a decision.

If all fourteen scope-correct checks pass, V16 authorizes design of a separately locked final-mechanic evaluation using the unchanged frozen V15 architecture. V16 does not access a final mechanic, protected record, adapter, alternate feature, or model.
