# V100 MASSIVE Core Source Feasibility Result

## Outcome

V100 passed every prospectively locked source gate. The official MASSIVE 1.1 archive matched its
40,251,390-byte S3 identity, and the automatically located `1.1/data/en-US.jsonl` member contained
16,521 valid records across the expected 18 scenarios, 60 intents, and 55 slot types.

The structural, hash-selected roles were fixed before any utterance-derived overlap was computed:

- catalog scenarios: `iot`, `recommendation`, and `calendar`;
- fully withheld unsupported scenario: `email`;
- hidden novel-valid intents: `calendar::calendar_query` and `iot::iot_hue_lightoff`;
- 12 other sufficiently supported catalog intents remained declared known.

## Four-class census

The text-free inventory contains 5,424 candidates:

| Class | Total | Validation | Test | Scenario coverage |
|---|---:|---:|---:|---:|
| Known familiar | 1,080 | 118 | 198 | 3 |
| Known unfamiliar | 1,956 | 225 | 346 | 3 |
| Novel valid | 1,007 | 119 | 169 | 2 |
| Unsupported | 1,381 | 157 | 271 | 1 |

Every class exceeds the locked minimum of 64 overall and 16 separately in validation and test. Both
known classes cover all three catalog scenarios, the novel class covers exactly the two independently
hidden-intent scenarios, and the unsupported class comes only from the one completely withheld scenario.

## Access and integrity boundary

The archive payload SHA-256 is
`4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`; the text-free candidate-index
SHA-256 is `d436b27715aa3daa6d03116428d41a401c51a5a051792fdf5673aea107c9b46f`.

All 16,521 source records were parsed and tokenized automatically. No raw or annotated utterance,
normalized token set, or slot value was emitted or manually inspected. There were zero model loads,
model generations, API calls, adapter-training runs, real service calls, and external side effects.

## Interpretation and authorization

V100 establishes source feasibility only. It does not measure novelty detection, abstention,
calibration, posterior quality, planning, or LLM capability. The positive result authorizes only a
fresh preregistration that hash-selects validation and test populations before selected-language
extraction. It does not authorize language extraction yet, any model access, API use, training,
posterior integration, planning, or execution.
