# V115 Two-Pass Contrastive Catalog-Fit Development Plan

V114 showed that typed abstention is a borderline and unstable novelty discriminator and that a static
known-intent rescue can increase false-known cost. V115 therefore changes the evidence, not a threshold.
On 192 new record-disjoint MASSIVE records, the pinned local model first uses the exact V112 typed-choice
interface. A second no-retry pass is then shown the first known proposal—or deterministic nearest intent
when pass one did not propose one—and must explicitly challenge it against the complete declared catalog.

The second pass chooses exactly one fixed verdict: confirm the reviewed candidate, select another declared
candidate, identify a coherent valid-but-undeclared capability, reject the request as outside every visible
scenario, or report insufficient evidence. It also emits a bounded novelty probability. Hidden intent
names, descriptions, and labels are never shown. Invalid output becomes zero-confidence abstention with
zero novelty probability. The declared catalog necessarily includes its public intent labels and slot types;
the hidden record-level truth class, intent, and scenario are never supplied to either pass.

The same 240 fixtures receive exactly two generations in one model load: 192 observed requests and 48
missing-observation controls. There are no retries. The primary evidence question is whether explicit
valid-but-undeclared verdicts meet the frozen V112 novelty precision, recall, false-positive, and calibration
gates while the review also preserves known and unsupported discrimination. A deterministic shadow policy
accepts a known request only when pass two confirms the identical pass-one known proposal, accepts
unsupported only when both passes agree, and asks in every other case. Pass-two novelty always asks and can
never define a capability.

V115 is development evidence from the same source distribution. Even a full pass authorizes only a search
for a genuinely independent confirmation source. It cannot open V101's protected language, begin typed
induction or richer planning, call an API, train an adapter, prune hypotheses, or grant the model belief,
action, capability, tool, or execution authority.
