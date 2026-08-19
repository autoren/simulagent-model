# V77 local-first clarification benchmark preregistration

## Purpose

V77 begins a new development-only LLM-interface program after the V76
active-sensing source deferral. Its first stage contains no language-model
inference. It asks whether an exactly enumerable task can distinguish candidate
generation, candidate weighting, and posterior-aware decision making before a
local model is allowed to propose any interpretation.

The benchmark models the synthetic instruction `Send the final report to
Jordan.` using four supported interpretations and one explicit
`none_of_the_above` interpretation. Supported interpretations cross two
documents (`q2_report`, `annual_report`) with two recipients (`jordan_lee`,
`jordan_patel`). The unknown interpretation has no correct irreversible send.
It can be exposed by clarification evidence and handled through a reversible
`safe_draft` or `abstain` action.

This is a project-authored mechanism and software benchmark. It is not human
language evidence, an external benchmark, a safety guarantee, or evidence that
an LLM understands the instruction.

## Exact interaction model

The hidden interpretation persists for the episode. The observable process has
an active state and an absorbing terminal state. Three information actions are
available:

- `ask_report`, returning a noisy Q2, annual, or other report label;
- `ask_recipient`, returning a noisy Lee, Patel, or other recipient label;
- `ask_full_details`, returning one of the four complete supported
  interpretations or `full_other`.

Four reversible candidate-specific draft actions remain active and return a
noisy approval or rejection. Four irreversible send actions terminate. A
`safe_draft` terminates the simulated task with a reversible conservative
outcome, and `abstain` terminates without acting. No action touches an external
tool or real recipient.

Question channels use fixed symmetric noise. All hidden interpretations have
identical observation support for every action, so neither MAP nor any later
candidate-set control can rely on an off-support fallback. Draft approval is
more likely only for the matching supported interpretation. The positive
reward profile gives a large penalty to the wrong irreversible send; the
dominant-control profile makes `safe_draft` preferable regardless of the hidden
interpretation.

## Fail-closed certification

The posterior-aware planner may select an irreversible send only when both
conditions hold under its complete current belief:

1. the probability of that exact supported interpretation is at least `0.90`;
2. the `none_of_the_above` probability is at most `0.10`.

Information actions, reversible drafts, `safe_draft`, and `abstain` remain
available. Point controls are planned under their collapsed beliefs and are
evaluated only in shadow under the complete environment belief; their actions
never receive execution authority. This deliberately measures the consequence
of epistemic collapse without weakening the deployed fail-closed boundary.

## Frozen fixtures

Four complete priors are registered before any planner implementation or
policy value is computed:

1. `ambiguous_send`: four supported meanings share most mass and
   `none_of_the_above` retains nonzero mass. Information should precede an
   irreversible send.
2. `clear_send`: `q2_report__jordan_lee` has probability `0.995`. The certified
   send should dominate unnecessary clarification.
3. `unknown_heavy`: `none_of_the_above` has probability `0.40`. Clarification
   must be able to route an unknown-indicating history to a reversible or
   abstaining outcome.
4. `dominant_safe_draft`: it reuses the ambiguous prior but changes only the
   reward profile so that `safe_draft` is dominant. This is the uncertainty-
   without-decision-value control.

All fixtures use horizon three and undiscounted bounded reward.

## Model-free controls

The first immutable census compares:

- exact posterior-aware planning with complete belief certification;
- MAP interpretation planning, evaluated under the complete environment;
- persistent posterior sampling, evaluated under the complete environment;
- an act-immediately policy over terminal controls;
- an ask-always policy that repeatedly requests full details before a final
  certified terminal control;
- the best open-loop action sequence;
- an oracle-interpretation ceiling that knows the persistent interpretation.

The primary outcomes are expected action regret under the exact environment,
not action agreement alone. Every reachable posterior must normalize, every
policy branch must be total on its evaluation support, and all control outputs
remain simulated.

## Noncompensatory gates

The benchmark authorizes a frozen local-model candidate-generation protocol
only if every registered structural, control, and access gate passes. In
particular:

- `ambiguous_send` must begin with focused clarification, later use both a
  report and recipient question on reachable policy paths, and materially
  outperform MAP and act-immediately control;
- `clear_send` must send `q2_report` to `jordan_lee` immediately, while the
  ask-always control incurs material regret;
- `unknown_heavy` must clarify first and contain a reachable
  unknown-indicating history whose continuation is `safe_draft` or `abstain`,
  never an irreversible send;
- `dominant_safe_draft` must select `safe_draft` immediately and give MAP and
  posterior sampling zero material regret;
- all observation kernels must normalize and retain identical support across
  interpretations;
- the design and outcome stages must record zero model forward passes, API
  calls, adapter runs, human records, and external side effects.

Any failed gate freezes a model-free benchmark-design failure. Parameters,
priors, thresholds, costs, and fixture membership may not be tuned using the
failed census. A successor must be separately named and preregistered.

## Authorization boundary

A complete pass authorizes only a fresh protocol for frozen local-model
candidate generation on development-only synthetic language. It does not
authorize API access, learned evidence likelihoods, LoRA, LLM-guided action
pruning, execution authority, V58 human-language work, or any claim beyond this
finite project-authored benchmark.
