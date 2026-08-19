# V103 PRESTO Target-Syntax Census Plan

## Purpose

V102r1 found zero candidates under the README-documented guillemet target-argument parser. V103 asks,
without emitting target or context language, whether the failure occurs because literals use another
delimiter or because literal values are not exact copies of context.

## Frozen diagnostics

For en-US human-context records in the persisted PRESTO dev/test archive, count six literal families:
double guillemets, single guillemets, ASCII double quotes, curly double quotes, ASCII single quotes, and
square brackets. The first five are candidate-eligible string delimiters; square brackets are diagnostic
only because they may encode structure.

For each family, count records at five successive stages: delimiter present, quality-filtered literal,
literal absent from the current input, literal present in admissible context, and both absent from input
and present in context. Also count aggregate target punctuation features. No literal, target, input,
context, normalized token, identifier, root-function name, or hash derived from language may be emitted.

## Decision gate

Union candidate-eligible families by source record. A viable successor requires at least 64 candidates
in development, 64 in protected test, 256 overall, 64 previous-turn-dependent, 64 seeded-state-dependent,
two context-source kinds, and eight semantic root functions. Passing authorizes only a new preregistered
scientific construction. Failure closes PRESTO as the paired insufficiency source and leaves MASSIVE as
the core open-set benchmark.
