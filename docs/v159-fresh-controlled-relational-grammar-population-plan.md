# V159 fresh controlled relational-grammar population plan

## Purpose

V158 showed that a large lexical score and a large top-two margin do not certify semantic sufficiency. Five
fresh development requests attracted the wrong specific question even though their margins were high. V159
therefore creates a new population in which topical retrieval and explicit relation structure are separately
observable.

This is a population-design branch only. It does not run or score a grammar policy, retrieval policy, LLM,
API, learned classifier, calibration routine, or planner.

## Prospective language contract

The interaction catalog registers four specific binary questions and one generic route question. Each
specific question exposes:

- ordinary retrieval metadata;
- exactly two normalized relation aliases; and
- two typed closed-answer options.

Relation aliases map only to query IDs. They do not map to state IDs, truth labels, witnesses, effects,
preconditions, tools, or executable capabilities.

Each complete group contains four request strata:

1. `lexical_control`: explicit catalog language identifies one specific question;
2. `grammar_unique`: one quoted registered relation alias occurs in a controlled relation construction;
3. `grammar_conflict`: two aliases belonging to different specific questions occur as unresolved
   alternatives; and
4. `insufficient`: no registered relation alias is supplied.

The remaining four fixtures are trusted closed answers: a family route, an unclear route, and the two
specific typed answers. Generic routing is non-semantic and cannot create a witness. Only a valid specific
closed answer can create a typed witness.

## Population and split

- 4 fresh families;
- 8 slot variants per family;
- 32 complete groups;
- 8 fixtures per group;
- 256 fixtures total;
- first 4 variants per family: development;
- last 4 variants per family: evaluation.

The evaluation half is generated and sealed with the design but is not available to the successor policy.

## Noncompensatory design gates

The branch closes unless all of the following hold:

- exact choice, query, family, group, stage, stratum, and split counts;
- eight unique normalized relation aliases;
- every conflict names aliases belonging to two distinct registered queries;
- no public truth, oracle, compatibility, grammar-cardinality, witness, or state-proposal field;
- exact specific-answer witness routing;
- valid generic routes with zero semantic witnesses;
- request, route-only, malformed, and unclear evidence fail closed to `A00`;
- complete authoritative hypothesis retention;
- zero exact conversation overlap with prior controlled populations;
- zero policy scores, model/API calls, training, side effects, action, or execution.

## Claim boundary

A passing V159 establishes only that a fresh, project-authored controlled-language asset is internally
coherent and suitable for a separately preregistered model-free development policy. Because relation aliases
and constructions are supplied by the benchmark, this is grammar-controlled feasibility—not unrestricted
open-world language understanding and not external transfer evidence.

