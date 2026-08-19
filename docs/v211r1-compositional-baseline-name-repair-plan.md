# V211r1 compositional baseline name repair

V211 passed its unit tests and completed the protected-free prediction firewall, but failed the exact prediction-count gate. The compositional response-span predictions were emitted under the obsolete key `CONTEXT_CONTRAST`; the scorer correctly expected the preregistered name `COMPOSITIONAL_RESPONSE_SPAN`. Consensus had consumed those same values and was complete and correct, but the named comparator received zero scored rows.

V211r1 changes only the emitted key. It reuses the frozen learned lexicon, evaluation surface, sealed evaluation truth, split, metrics, gates, and decision-impact implementation. It does not refit, reread calibration, or alter any prediction value. A separate prediction worker receives only the frozen learned lexicon and evaluation surface. Its output is frozen before the existing sealed evaluation truth is opened for rescoring.
