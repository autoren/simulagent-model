# V17r2 preregistration amendment: visible action outcome

V17r2 inherits the complete design, training rule, gates, firewall, claim boundary, and one-shot limits in `docs/v17-final-mechanic-plan.md`, with one construction-only correction made before any final record, feature, prediction, or score existed.

The simulator exposes both a normalized state-change summary (`actualOutcome`) and the visible text returned by the chosen action (`outcome`). A read-only diagnostic changes the visible observation without changing the action surface, so the initial V17 transition identity collapsed all assignments and aborted before writing data. V17r2 defines a transition by the tuple:

`(visible action outcome, normalized actual outcome, action-surface delta)`.

This is the complete observable transition for the candidate action. The console readout independently reports mirror, generator, and tone status, so the same locked assertion still requires exactly eight assignments and eight distinct codes. The constructor remains forbidden to write any record if that invariant fails.

The initial failed invocation disclosed only that the old normalized identity had one code. It disclosed no returned text, target record, feature, prediction, or score. The archived abort is documented in `docs/v17-construction-abort.md`. V17r2 receives a new construction lock before a second constructor is invoked; it remains the only final corpus and one-shot model evaluation.
