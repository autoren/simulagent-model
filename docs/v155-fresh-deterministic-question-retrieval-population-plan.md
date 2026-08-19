# V155 fresh deterministic question-retrieval population plan

## Purpose

V154 showed that a local LLM can order registered clarification questions usefully, but it did not meet the strict top-1 and reciprocal-rank gates. Bounded low reasoning made performance worse. V155 therefore begins a scientifically distinct branch: deterministic retrieval from explicit registered question metadata.

V155 only constructs and freezes a fresh synthetic population. It does not run or score a retrieval policy and does not access a model. Separating population construction from V156 policy preregistration prevents observed retrieval outcomes from influencing the benchmark design.

## Population

The catalog has six known states, generic non-executable novel-candidate `N60`, unsupported `U60`, and insufficient-evidence `A00`. Six registered binary clarification questions cover:

- community-garden versus wildlife-rehabilitation permits;
- conference badges versus makerspace credentials;
- greenhouse sensors versus marine buoys;
- theater tickets versus museum-storage visits;
- active archive-copy revision versus certified-record erasure;
- solar-rebate tracking versus certified-grant-record erasure.

Each question provides visible, versioned retrieval metadata: anchor phrases, primary terms, and secondary terms. This metadata describes the registered question; it does not contain fixture IDs, truth labels, oracle query IDs, or state proposals.

Each of 48 groups contains four pre-answer requests and two trusted closed-answer records. The 288 fixtures are evenly split between synthetic development and evaluation. Public rows hide family, stage, truth, compatibility, oracle query, witness, and variant metadata.

## Gates

The design must prove exact group and split completeness, typed-answer routing, pre-answer and malformed-event fail-closure, complete hypothesis retention, zero candidate-proposal surface, and zero exact conversation overlap with prior controlled populations. Automated overlap comparison may read prior files only to compare canonical conversation strings; it must not print or persist their language.

The design stage permits no retrieval policy scoring, evaluation-policy access, model/tokenizer load, model generation or scoring, API, training, action, or execution.

Passing authorizes only a separately preregistered model-free policy on the V155 development split. Evaluation language, an LLM or hybrid condition, calibration, induction, authority, action, and execution remain closed.
