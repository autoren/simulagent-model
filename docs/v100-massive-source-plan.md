# V100 MASSIVE Core Source Plan

## Purpose

V100 tests whether the official MASSIVE 1.1 `en-US` source can support the four core controlled open-set
classes with typed scenario, intent, and slot ground truth. The official archive identity and all role,
count, partition, and access gates are frozen before download.

## Structural partition

After automatic validation of the en-US records, scenarios and intents are filtered only by annotation
counts. One eligible scenario is hash-selected and withheld completely as unsupported. Three different
eligible scenarios are hash-selected as the catalog. One sufficiently supported intent is hidden from
each of two hash-selected catalog scenarios; every other sufficiently supported catalog intent is
declared known, with at least three declared intents remaining.

Role assignment occurs before utterance-derived overlap is computed. Known examples are familiar when
the current utterance shares a normalized non-stopword token with its intent identifier and unfamiliar
when it shares none. Novel-valid examples use hidden intents inside catalog scenarios. Unsupported
examples use the completely withheld scenario.

## Partition gates

Each class must have at least 64 records overall and at least 16 independently in MASSIVE's validation
(`dev`) and test partitions. Familiar and unfamiliar known examples must each cover at least two catalog
scenarios; novel examples must cover exactly two hidden scenarios; unsupported examples must cover one
withheld scenario. The structural ontology must contain at least 18 scenarios, 60 intents, and 20 slot
types before selection.

## Boundary

The source census may tokenize utterances and parse slot markup automatically, but may not emit raw or
annotated utterances, slot values, normalized tokens, or prompts. Passing authorizes only preregistration
of a hash-selected validation/test population before selected-language extraction. It does not authorize
manual inspection, model access, API use, training, posterior integration, planning, or execution.
