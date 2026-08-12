# V16 results: scope-correct V15 gate replay

V16 passes all fourteen scope-correct gates without fitting a model, running inference, recomputing a prediction, changing a threshold, or accessing new data.

The complete-intervention-group gate applies to 15 transfer folds whose own evaluation masks contain complete six-record groups. Its worst value is 0.577 on `surface:current_observation`, above the unchanged 0.50 threshold. Eleven lexicon/operator/combined folds correctly report the group metric as not applicable because their evaluation masks contain zero complete groups.

Decision: `authorize_separately_locked_final_mechanic_evaluation`.

This authorizes only the design of a separately locked final-mechanic evaluation using the unchanged frozen V15 architecture. No final mechanic has been accessed, and LoRA remains unauthorized.

- V16 lock: `cbdd17f8327ef15438e437b2f1cc312357accbce35243a84545f1fb6443826e2`;
- V16 result: `5ea5ef2b0d375a223eadac6ff325f62d8badfd1325a36935510b9b8a47e86e08`;
- new fits / forward passes / predictions / threshold changes: 0 / 0 / 0 / 0.
