# V102 PRESTO Human-Context Source Plan

## Purpose

V102 tests whether official PRESTO v1 can supply a within-record evidence-sufficiency control without
using synthetic language or dataset identity as the class cue. It freezes the 415,990,813-byte archive
identity and the mechanical dependency rule before downloading the payload.

## Eligibility

Parse only the official `presto_dev.jsonl` and `presto_test.jsonl` members. A candidate must be `en-US`,
carry `metadata.context == "human"`, and contain a target argument delimited by `«` and `»` that:

1. normalizes to at least three characters and no more than eight tokens;
2. is not one of the frozen trivial constants;
3. is absent as a contiguous normalized token sequence from the current input; and
4. occurs as a contiguous normalized token sequence in a previous human turn or the record's seeded
   lists, notes, or contacts.

For each admitted source record, the eventual full-context and context-ablated conditions must share the
same example identifier, current input, and target. Only the evidence supplied to the parser changes.
The feasibility artifact emits identifiers and structural dependency-source kinds, never input text,
target text, target arguments, context text, normalized tokens, or seeded values.

## Gates

Require at least 64 eligible candidates independently in development and protected test, 256 overall,
64 whose missing value is supported by a previous turn, and 64 supported by seeded state. Require at
least two dependency-source kinds and eight distinct target root functions. Development and protected-
test identifiers must be disjoint and the synthetic-context candidate count must be exactly zero.

## Boundary

Passing establishes only that an independently checkable paired insufficiency population can be
constructed. It authorizes a separate text-free population preregistration, not language extraction,
manual inspection, model access, API use, training, posterior integration, planning, or execution.
