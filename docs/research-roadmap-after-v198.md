# Research roadmap after V198

## Confirmed result and exact claim boundary

V195's development result now has a dialogue-isolated protected confirmation. V198 reused the complete frozen V195
policy without prompt, model, budget, parser, controller, cost, gate, or fallback changes. On 113 observed protected
SGD records plus 12 missing controls, the local `Qwen3.8-27B-4bit` ranker achieved:

- primary top-3 recall `0.95625`;
- balanced top-3 recall `0.9557522124`;
- primary top-3 controller cost `0.20875`;
- structural validity `0.9911504425`;
- target retention and trusted exact completion `1.0`; and
- zero final-phase truncation, retry, API use, authority, action, or execution.

On the identical protected records, unchanged `CHAR_LAST` cost `0.2322186147`. The local model's paired primary
improvement was `0.0234686147`, clearing the prospective `0.01` incremental gate. The independently reconstructed
outcome is frozen at
`configs/v198-protected-language-menu-ranker-confirmation-outcome-lock.json` with SHA-256
`acb48d8b2c904fc1bedc435823770ef36089d9d26e5a67aef427460f318091c3`.

The supported claim is therefore no longer development-only:

> For this finite 14-contract SGD benchmark, the fixed bounded local model can non-authoritatively rank a complete
> clarification menu more economically than a strong character-retrieval comparator on both development and a
> dialogue-isolated protected population, while the trusted controller retains every target and all terminal
> authority.

This is not unrestricted open-world recognition. It does not show that the model can decide whether an arbitrary
request belongs outside the catalog, create a correct ontology entry, identify hidden service versions, answer its
own clarification questions reliably, preserve performance under distribution shift, or choose safe delayed actions.

## Runtime finding

Low reasoning effort did not make the Qwen reasoning channel self-limiting. Every V195 and V198 reasoning phase used
all 48 tokens and none naturally closed. The separate mechanical close and reserved 64-token final phase prevented
final truncation. That two-phase harness is part of the confirmed condition. A future run may compare a different
reasoning budget only as a separately preregistered ablation; it may not silently replace the confirmed policy.

## Active Track B: controlled robustness and shift

The next question is whether the confirmed gain reflects stable semantic ranking or sensitivity to incidental menu
presentation. Start with exact, model-free transformations before introducing synthetic language or another model.

### B1 — exact menu-presentation invariance

1. Freeze a family of deterministic, bijective option-order and opaque-option-ID permutations using hashes of public
   record identities and fixed salts. Do not read utterance text when assigning transformations.
2. Prove for every variant that the 14 semantic descriptions are unchanged, the option-to-contract map remains a
   bijection, the hidden target remains expressible exactly once, and the trusted `OTHER` fallback is unchanged.
3. Define paired contract-level invariance, top-3 recall, controller-cost, structural-validity, and fail-closed gates
   before any transformed model generation.
4. Evaluate `CHAR_LAST` and the unchanged V195/V198 local policy on development language first. Model outputs remain
   proposals only; all IDs are mapped back through the frozen bijection before scoring.
5. Open a paired protected robustness role only if the development policy and gates were fixed before transformed
   protected generations. Reusing V198 records would be paired secondary evidence, not a new independent
   confirmation population.

This is the cleanest first robustness test because semantic equivalence and target retention can be proved exactly;
it does not require a human to judge whether a paraphrase preserved meaning.

### B2 — controlled menu expansion and distractors

Proceed only after B1. Candidate distractors must be sourced and locked without looking at model errors. A model-free
collision audit must establish that no added option is another valid gold target for the selected record under the
benchmark contract semantics. Report results by expansion size and hardness. Keep the complete authoritative target
set and `OTHER` fallback available.

### B3 — vocabulary, paraphrase, and ambiguity shifts

Natural external language is preferred to post-result paraphrase generation. If deterministic or model-generated
transformations are used, label them synthetic stress tests and preregister preservation checks. Without independent
human validation they cannot establish natural-language equivalence or human ambiguity. Deliberately ambiguous cases
must be scored against a set of admissible contracts or a clarification requirement, not a forced single hidden
label.

## Later tracks

### Track C — shadow ontology acquisition

Defer until the finite-menu interface survives Track B. New concepts may enter only a reversible shadow ledger.
Deterministic type, precondition, effect, identifiability, collision, and conflict checks must precede sandbox use.
Model confidence never registers, prunes, merges, or deletes a concept.

### Track D — clarification reliability and human factors

Independent participants remain unavailable. Source-oracle, scripted, or model-generated answers may be used as
explicit simulations, but cannot support claims about human accuracy, effort, or behavior.

### Track E — richer partially observable decisions

After Track B, connect semantic ambiguity to delayed state-dependent consequences. Compare posterior-aware
clarification and control with MAP, retrieval-only, always-ask, and always-defer policies. The key outcome is expected
decision loss under retained semantic uncertainty, not standalone classification accuracy.

### Track F — additional model or API

Not currently justified. V198 answered the capacity/usefulness question for the fixed local condition. A second model
or API needs a separate preregistered scientific contrast, fixed accounting, and no fallback or selection after
observing results. It should not precede the robustness question.

## Durable controls

- Keep the model non-authoritative and map malformed, truncated, duplicate, unknown, or extra-key output to
  `INSUFFICIENT`.
- Never prune the authoritative contract universe from model output.
- Do not tune on V198 protected errors or inspect raw protected responses; normalized outputs are sufficient.
- Freeze every transformation, target mapping, metric, and gate before transformed scoring.
- Distinguish development, paired protected robustness, fresh confirmation, and synthetic stress evidence.
- No API, training, ontology mutation, service call, external side effect, action, or execution follows from V198.

